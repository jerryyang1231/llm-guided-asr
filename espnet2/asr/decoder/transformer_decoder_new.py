# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any, List, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import (
    BatchScorerInterface,
    MaskParallelScorerInterface,
)


class BaseTransformerDecoder(
    AbsDecoder, BatchScorerInterface, MaskParallelScorerInterface
):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self._output_size_bf_softmax = attention_dim
        # Must set by the inheritance
        self.decoders = None
        self.batch_ids = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            return_hs: (bool) whether to return the last hidden output
                                  before output layer
            return_all_hs: (bool) whether to return all the hidden intermediates
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        x = self.embed(tgt)
        intermediate_outs = []
        for layer_idx, decoder_layer in enumerate(self.decoders):
            x, tgt_mask, memory, memory_mask = decoder_layer(
                x, tgt_mask, memory, memory_mask
            )
            if return_all_hs:
                intermediate_outs.append(x)
        if self.normalize_before:
            x = self.after_norm(x)
        if return_hs:
            hidden = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        if return_hs:
            return (x, hidden), olens
        elif return_all_hs:
            return (x, intermediate_outs), olens
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if return_hs:
            hidden = y
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return (y, hidden), new_cache
        return y, new_cache

    def score(self, ys, state, x, return_hs=False):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if return_hs:
            (logp, hs), state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), hs, state
        else:
            logp, state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).


        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        if return_hs:
            (logp, hs), states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=return_hs
            )
        else:
            logp, states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=return_hs
            )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        if return_hs:
            return (logp, hs), state_list
        return logp, state_list

    def forward_partially_AR(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_lengths: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (n_mask * n_beam, maxlen_out)
            tgt_mask: input token mask,  (n_mask * n_beam, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            tgt_lengths: (n_mask * n_beam, )
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)  # (n_mask * n_beam, maxlen_out, D)
        new_cache = []
        if cache is None:
            cache = [None] * len(self.decoders)

        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, tgt_lengths, memory, memory_mask = (
                decoder.forward_partially_AR(
                    x, tgt_mask, tgt_lengths, memory, None, cache=c
                )
            )
            new_cache.append(x)

        if self.batch_ids is None or len(self.batch_ids) < x.size(0):
            self.batch_ids = torch.arange(x.size(0), device=x.device)

        if self.normalize_before:
            y = self.after_norm(
                x[self.batch_ids[: x.size(0)], tgt_lengths.unsqueeze(0) - 1].squeeze(0)
            )
        else:
            y = x[self.batch_ids, tgt_lengths.unsqueeze(0) - 1].squeeze(0)

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, torch.stack(new_cache)

    def batch_score_partially_AR(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        yseq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        # merge states
        if states[0] is None:
            batch_state = None
        else:
            # reshape state of [mask * batch, layer, 1, D]
            # into [layer, mask * batch, 1, D]
            batch_state = states.transpose(0, 1)

        # batch decoding
        tgt_mask = (~make_pad_mask(yseq_lengths)[:, None, :]).to(xs.device)
        m = subsequent_mask(tgt_mask.size(-1), device=xs.device).unsqueeze(0)
        tgt_mask = tgt_mask & m

        logp, states = self.forward_partially_AR(
            ys, tgt_mask, yseq_lengths, xs, cache=batch_state
        )

        # states is torch.Tensor, where shape is (layer, n_mask * n_beam, yseq_len, D)
        # reshape state to [n_mask * n_beam, layer, yseq_len, D]
        state_list = states.transpose(0, 1)
        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn
            except Exception:
                use_flash_attn = False

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    True,
                    False,
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    True,
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )


class LightweightConvolutionTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class LightweightConvolution2DTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolutionTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolution2DTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class TransformerMDDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        use_speech_attn: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
                (
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    )
                    if use_speech_attn
                    else None
                ),
            ),
        )

        self.use_speech_attn = use_speech_attn

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        speech: torch.Tensor = None,
        speech_lens: torch.Tensor = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        if speech is not None:
            speech_mask = (~make_pad_mask(speech_lens, maxlen=speech.size(1)))[
                :, None, :
            ].to(speech.device)
        else:
            speech_mask = None

        x = self.embed(tgt)
        if self.use_speech_attn:
            x, tgt_mask, memory, memory_mask, _, speech, speech_mask = self.decoders(
                x, tgt_mask, memory, memory_mask, None, speech, speech_mask
            )
        else:
            x, tgt_mask, memory, memory_mask = self.decoders(
                x, tgt_mask, memory, memory_mask
            )
        if self.normalize_before:
            x = self.after_norm(x)
            if return_hs:
                hs_asr = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)

        if return_hs:
            return x, olens, hs_asr

        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        speech: torch.Tensor = None,
        speech_mask: torch.Tensor = None,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            speech: encoded speech, float32  (batch, maxlen_in, feat)
            speech_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            if self.use_speech_attn:
                x, tgt_mask, memory, memory_mask, _, speech, speech_mask = decoder(
                    x,
                    tgt_mask,
                    memory,
                    memory_mask,
                    cache=c,
                    pre_memory=speech,
                    pre_memory_mask=speech_mask,
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, memory_mask, cache=c
                )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if return_hs:
            h_asr = y

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return y, h_asr, new_cache
        return y, new_cache

    def score(self, ys, state, x, speech=None):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            speech=speech.unsqueeze(0) if speech is not None else None,
            cache=state,
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        speech: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(
            ys, ys_mask, xs, speech=speech, cache=batch_state
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

class LLMGuidedTransformerDecoder(BaseTransformerDecoder):
    # @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        ctc_vocab_path: Optional[str] = None,
        train_biasing_words_path="/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/train_sp/output.txt",
        dev_biasing_words_path="/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/dev/output.txt",
        test_biasing_words_path="/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/test/output.txt",
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        self.output_size = encoder_output_size
        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )

        # v defined later
        self.llm = None
        self.embed = None
        self.ctc = None
        self.ctc_tokenizer = None
        self.ctc_token_id_converter = None
        if ctc_vocab_path is not None:
            from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
            from espnet2.text.token_id_converter import TokenIDConverter
            self.ctc_tokenizer = SentencepiecesTokenizer(
                ctc_vocab_path + "/bpe.model"
            )
            self.ctc_token_id_converter = TokenIDConverter(
                ctc_vocab_path + "/tokens.txt"
            )

        self.use_cache = False

        self.biasing_words_dict = {}
        for prompt_file in [train_biasing_words_path, dev_biasing_words_path, test_biasing_words_path]:
            if prompt_file is None:
                raise ValueError(f"missing {prompt_file}")
            with open(prompt_file, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue

                    # 如果這一行只有一個字串（也就是沒有任何空白可以拆分），就跳過或設成空字串
                    if len(line.split()) == 1:
                        # 這裡代表只有 utt_id 而已，沒有後續 biasing words
                        utt_id = line
                        biasing_words = ""  # 或直接 continue, 視需求決定
                    else:
                        # 這一行至少有一個空白，能拆出兩部分
                        utt_id, biasing_words = line.split(None, 1)
                    self.biasing_words_dict[utt_id] = biasing_words


    def set_utt_ids(self, utt_ids: List[str]):
        """
        將批次的 utt_id 清單傳入解碼器內部，以便後續 forward／forward_one_step 使用。
        utt_ids: List[str]，長度 = batch size，順序對應於該批次樣本順序。
        """
        self.utt_ids = utt_ids

    def lookup_bias_words(self, utt_id: str) -> str:
        return self.biasing_words_dict.get(utt_id, "")

    def forward(
        self,
        hs_pad: torch.Tensor,      # (B, T_enc, D): encoder output
        hlens: torch.Tensor,       # (B,): 每筆 utterance 的 encoder 長度
        ys_in_pad: torch.Tensor,   # (B, T_dec): decoder 輸入（已 pad）
        ys_in_lens: torch.Tensor,  # (B,): decoder 輸入長度
        # biasing_words: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = hs_pad.size(0)

        # 1. CTC 解碼：先把 encoder 輸出( hs_pad )丟進 ctc.argmax 取得每 frame 最有機率的 id 序列
        lpz = self.ctc.argmax(hs_pad).data  # (B, T_enc)

        # 2. 依序去除重複與 blank，並轉成字串
        hyps_ctc = []
        ctc_lengths = []
        for l in lpz:                       # l: shape (T_enc,)
            y_hat = torch.unique_consecutive(l)   # 去掉連續重疊相同 token
            y = y_hat[y_hat != 0]                # 刪掉 CTC blank token（假設 blank_id = 0）
            if self.ctc_tokenizer is not None:
                # 如果有 tokenizer，就依次把 id -> token -> 文字
                hyps_ctc.append(
                    self.ctc_tokenizer.tokens2text(
                        self.ctc_token_id_converter.ids2tokens(y)
                    )
                )
                ctc_lengths.append(-1)
            else:
                hyps_ctc.append(y)
                ctc_lengths.append(y.size(0))
        ctc_lengths = hlens.new(ctc_lengths)  # 轉成跟 hlens 同型態的 tensor

        # 嘗試從屬性取得 utt_id
        utt_ids = getattr(self, "utt_id", None)
        if utt_ids is not None:
            biasing_batch = []
            for uid in utt_ids:
                biasing_batch.append(self.biasing_words_dict.get(uid, ""))  # default = ""
        else:
            biasing_batch = None

        # LLM 呼叫：hyps_ctc 代表「CTC 解碼文字」；hyps_bias 代表「偏向詞字串」
        hyps_bias = biasing_batch if biasing_batch is not None else [""] * B

        # 統一令所有要丟給 LLM 的長度設為 -1，讓 LLM 內部依「字串長度」自己算
        bias_lengths = hlens.new([-1] * B)  # 偏向詞字串的長度向量（placeholder）

        # 假設你的 self.llm.forward 已經改成接受兩個字串 list：
        #   llm(hyps_ctc: List[str], ctc_lengths: Tensor,
        #       hyps_bias: List[str], bias_lengths: Tensor,
        #       ys_in_pad: Tensor, ys_in_lens: Tensor)
        llm_out, llm_out_lengths = self.llm(
            hyps_ctc,    # CTC decode 出來的文字
            ctc_lengths, # 讓 LLM 自己判斷 CTC 送入長度
            hyps_bias,   # 偏向詞列表( human readable )
            bias_lengths,# 讓 LLM 自己判斷這裡的文字長度
            ys_in_pad,   # ASR decoder 先前加 SOS/EOS 過的 input token id
            ys_in_lens,  # ASR decoder input 的長度
        )

        # --- 4. 建立 attention mask，做 transformer decoder 部分（和你原本程式不變） ---
        tgt = llm_out  # (B, T_dec, D_llm)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)  # (B, 1, T_dec)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        tgt_mask = tgt_mask & m  # (B, T_dec, T_dec)

        memory = hs_pad  # (B, T_enc, D_enc)
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(memory.device)

        # 5. embed → decoder layers → output projection
        x = self.embed(tgt)  # 把 LLM 的 hidden vector 投影到 encoder output size
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)  # 投影到 ASR vocab 大小的 logit

        return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # 1. 嘗試從屬性取得 utt_id
        utt_id = getattr(self, "utt_id", None)
        if utt_id is not None:
            # 用 utt_id 查出對應的偏置詞字串
            biasing = self.biasing_words_dict.get(utt_id, "")

            # 將偏置詞用於 LLM 輸入
            # 具體方式依你的 LLM 接口而定，以下示意如何傳入
            hyps_bias = [biasing] * tgt.size(0)  # tgt.size(0) = batch size (通常為 1)
        else: # 用不到?
            hyps_bias = [""] * tgt.size(0)
        bias_lengths = tgt.new([-1] * tgt.size(0)) # -1 讓 LLM 自行用字串長度

        # 2. 原本的 CTC 解碼
        if tgt.size(1) == 1:
            lpz = self.ctc.argmax(memory).data  # (B, T_enc)

            y_hat = torch.unique_consecutive(lpz[0])
            self.hyp = y_hat[y_hat != 0]

            if self.ctc_tokenizer is not None:
                self.hyp = self.ctc_tokenizer.tokens2text(
                    self.ctc_token_id_converter.ids2tokens(self.hyp)
                )
        hyps_ctc = [self.hyp] * tgt.size(0)
        if self.ctc_tokenizer is not None:
            ctc_lengths = tgt.new([-1] * tgt.size(0))
        else:
            ctc_lengths = tgt.new([self.hyp.size(0)] * tgt.size(0))

        tgt[:, 0] = self.llm.start_of_response_token_id # Unecessary?

        # 呼叫 LLM，將 hyps_bias 傳進去
        llm_out, llm_out_lengths = self.llm.forward_inference(
            hyps_ctc,        # CTC 解碼文字
            ctc_lengths,     # CTC 假設長度
            hyps_bias,       # biasing words
            bias_lengths,
            tgt,             # 解碼器輸入 tokens
            tgt.new([tgt.size(1)] * tgt.size(0))  # 輸入長度
        )

        # 3. decoder cache 準備
        if cache is None:
            cache = [[None] * tgt.size(0)] * len(self.decoders)

        # 4. 把 LLM 輸出嵌入、跑 Transformer decoder
        x = self.embed(llm_out)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, None
        )

        # 5. 最後的投影與 softmax
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        if self.use_cache:
            return self.batch_score_cached(ys, states, xs)

        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        # 1. 把多個樣本各自的 decoder hidden state 合併成 batch_state（這裡直接跳過，因為 cache= None）
        batch_state = None

        # batch decoding
        # 2. 建立下一步輸入的遮罩 (subsequent_mask)
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        #    ys: (B, T_dec) → mask: (1, T_dec, T_dec)

        # 3. 呼叫 forward_one_step，計算這個 prefix（ys）在 encoder 輸出 xs 上，一步的 log-prob & new state
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)
        #    logp: (B, V)   new_states: list of per-layer hidden state

        # transpose state of [layer, batch] into [batch, layer]
        # 4. 把 new_states 從「[layer][batch, ...]」轉成「[batch][layer, ...]」
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]

        # 5. 回傳下一步的 log-prob (B×V) 和各樣本新的 decoder state
        return logp, state_list

    def forward_one_step_cached(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache_llm: List[Any] = None,
        cache_dec: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Any], List[torch.Tensor]]:
        # CTC decoding
        # when sos
        if tgt.size(1) == 1:
            lpz = self.ctc.argmax(memory).data

            y_hat = torch.unique_consecutive(lpz[0])
            hyp = y_hat[y_hat != 0]

            if self.ctc_tokenizer is not None:
                hyp = self.ctc_tokenizer.tokens2text(
                    self.ctc_token_id_converter.ids2tokens(hyp)
                )
                hyps_lengths = tgt.new([-1] * tgt.size(0))
            else:
                hyps_lengths = tgt.new([hyp.size(0)] * tgt.size(0))

            hyps = [hyp] * tgt.size(0)
        else:
            hyps, hyps_lengths = None, None

        tgt[:, 0] = self.llm.start_of_response_token_id # Unecessary?
        llm_out, new_cache_llm = self.llm.forward_inference_cached(
            hyps,
            hyps_lengths,
            tgt,
            tgt.new([tgt.size(1)] * tgt.size(0)),
            cache=cache_llm,
        )

        if cache_dec is None:
            cache_dec = [None] * (len(self.decoders) + 1)
        new_cache_dec = []

        x = self.embed(llm_out)
        if cache_dec[0] is not None:
            x = torch.cat([cache_dec[0], x], dim=1)
        new_cache_dec.append(x)

        for c, decoder in zip(cache_dec[1:], self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache_dec.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache_llm, new_cache_dec

    def batch_score_cached(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        # merge states
        n_batch = len(ys)
        n_dec_layers = len(self.decoders) + 1 # include embed layer
        n_llm_layers = self.llm.lm.config.num_hidden_layers

        if states[0] is None:
            batch_state_dec = None
            batch_state_llm = None
        else:
            batch_state_dec = [
                torch.stack([states[b][0][i] for b in range(n_batch)])
                for i in range(n_dec_layers)
            ]
            batch_state_llm = [
                (
                    torch.stack([states[b][1][i][0] for b in range(n_batch)]),
                    torch.stack([states[b][1][i][1] for b in range(n_batch)])
                )
                for i in range(n_llm_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states_llm, states_dec = self.forward_one_step_cached(
            ys, ys_mask, xs, cache_llm=batch_state_llm, cache_dec=batch_state_dec
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_dec_list = [
            [states_dec[i][b] for i in range(n_dec_layers)] for b in range(n_batch)
        ]
        state_llm_list = [
            [(states_llm[i][0][b], states_llm[i][1][b]) for i in range(n_llm_layers)]
            for b in range(n_batch)
        ]
        state_list = [
            (state_dec_list[b], state_llm_list[b]) for b in range(n_batch)
        ]
        # self.states_llm = states_llm
        return logp, state_list
