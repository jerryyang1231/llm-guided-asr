#!/usr/bin/env python3

"""Hugging Face Transformers PLM."""
import logging
from typing import Any, List, Tuple, Optional, Union

import torch
from typeguard import typechecked

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list
from espnet2.asr.llm.abs_llm import AbsLLM

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


class Llama(AbsLLM):
    @typechecked
    def __init__(
        self,
        model_name_or_path: str,
        template_prompt: Optional[str] = None,
        dtype: str = "bfloat16",
        cache_dir: str = None,
        pad_token: str = "<unk>",
        # ↓ 只保留這兩個量化必要參數 ↓
        # load_in_8bit: bool = False,
        # device_map: Union[str, dict] = "auto",
    ):
        super().__init__()

        assert model_name_or_path in [
            "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"
        ]
        self.is_llama2 = "Llama-2" in model_name_or_path

        logging.info(f"model_name_or_path: {model_name_or_path}")
        logging.info(f"dtype: {dtype}")
        logging.info(f"cache_dir: {cache_dir}")

        # 1) 無論如何先準備這個變數（預設 None）
        # bnb_config = None
        # if load_in_8bit:
        #     # 2) 只有開了才真正用 BitsAndBytesConfig
        #     bnb_config = BitsAndBytesConfig(
        #         load_in_8bit=True,
        #     )

        # 3) 呼叫 from_pretrained 的時候，把 quantization_config=bnb_config
        # self.lm = AutoModelForCausalLM.from_pretrained(
        #     model_name_or_path,
        #     cache_dir=cache_dir,
        #     torch_dtype=dtype,
        #     quantization_config=bnb_config,
        #     device_map=device_map,
        # )

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, torch_dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.template_prompt = template_prompt
        if template_prompt:
            assert "\"((HYP))\"" in template_prompt
            # 把 tokens[i:i+len_hyp_indicator] 串起來，對比字串 "((HYP))"
            template_prompt_tokens = self.tokenizer.tokenize(template_prompt)
            len_hyp_indicator = 5 if self.is_llama2 else 4
            for i in range(len(template_prompt_tokens)):
                if "".join(template_prompt_tokens[i: i + len_hyp_indicator]) == "((HYP))":
                    self.template_prefix_tokens = template_prompt_tokens[:i]
                    self.template_suffix_tokens = template_prompt_tokens[i + len_hyp_indicator:]
                    break
            
            # 轉成 token id 序列
            self.template_prefix_ids = (
                [self.lm.config.bos_token_id]   # Llama3: <|begin_of_text|>
                + self.tokenizer.convert_tokens_to_ids(self.template_prefix_tokens)
            )
            self.template_suffix_ids = self.tokenizer.convert_tokens_to_ids(self.template_suffix_tokens)

            if self.is_llama2:
                self.start_of_response_token_id = 29908 # "
                self.end_of_response_token_id = 29908 # "
            else:
                # llama3
                self.start_of_response_token_id = 1 # "
                self.end_of_response_token_id = 1 # "

            logging.info(f"template_prompt: \n---\n{self.template_prompt}((RESPONSE))\n---")
            logging.info(f"template_prefix_ids: {self.template_prefix_ids}")
            logging.info(f"template_suffix_ids: {self.template_suffix_ids}")
        else:
            self.start_of_response_token_id = self.lm.config.bos_token_id

            if self.is_llama2:
                self.end_of_response_token_id = self.lm.config.eos_token_id
            else:
                # llama3 has several eos tokens
                self.end_of_response_token_id = self.lm.config.eos_token_id[0]

        self.pad_token_id = self.tokenizer.vocab[pad_token] # Llama3: <|finetune_right_pad_id|>

        logging.info(f"start_of_response_token_id: {self.start_of_response_token_id}")
        logging.info(f"start_of_response_token: {self.tokenizer.convert_ids_to_tokens(self.start_of_response_token_id)}")
        logging.info(f"end_of_response_token_id: {self.end_of_response_token_id}")
        logging.info(f"end_of_response_token: {self.tokenizer.convert_ids_to_tokens(self.end_of_response_token_id)}")
        logging.info(f"pad_token_id: {self.pad_token_id}")
        logging.info(f"pad_token: {self.tokenizer.convert_ids_to_tokens(self.pad_token_id)}")

    def prepare_prompt(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.template_prompt is None:
            lm_in_pad, lm_in_lengths = res_in_pad, res_in_lengths
            # 把 -1（未填值）都換成 pad_token_id
            lm_in_pad[lm_in_pad == -1] = self.pad_token_id
        else:
            prefix_ids = res_in_pad[0].new(self.template_prefix_ids)
            suffix_ids = res_in_pad[0].new(self.template_suffix_ids)
    
            # List[str]
            if isinstance(hyp_in[0], str):
                lm_in = []
                lm_in_lengths = []
                for i, hyp in enumerate(hyp_in):
                    hyp_ids = self.tokenizer(
                        hyp, return_tensors="pt"
                    ).input_ids[0][1:].to(res_in_pad.device) # remove sos
                    lm_in.append(
                        torch.cat(
                            [
                                prefix_ids,
                                hyp_ids,
                                suffix_ids,
                                res_in_pad[i][res_in_pad[i] != -1]
                            ],
                            dim=0,
                        )
                    )
                    lm_in_lengths.append(
                        prefix_ids.size(0)
                        + hyp_ids.size(0)
                        + suffix_ids.size(0)
                        + res_in_lengths[i]
                    )
                lm_in_pad = pad_list(lm_in, self.pad_token_id)
                lm_in_lengths = torch.stack(lm_in_lengths)
            # List[torch.Tensor]
            else: 
                lm_in = [
                    torch.cat(
                        [
                            prefix_ids,                                       # 模板前綴的 token id 序列
                            hyp,                                              # 這條 sample 的 hypothesis Tensor
                            suffix_ids,                                       # 模板後綴的 token id 序列
                            res_in_pad[i][res_in_pad[i] != -1]                # 真正要預測的部分（去掉 pad=-1 的位置）
                        ],
                        dim=0,
                    ) for i, hyp in enumerate(hyp_in)
                ]
                lm_in_pad = pad_list(lm_in, self.pad_token_id)
                lm_in_lengths = (
                    prefix_ids.size(0)        # 前綴長度
                    + hyp_in_lengths          # 每筆 hypothesis 的長度（Tensor）
                    + suffix_ids.size(0)      # 後綴長度
                    + res_in_lengths          # 每筆實際要預測部分的長度（Tensor）
                )

        return lm_in_pad, lm_in_lengths

    def forward(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],   # ctc output
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,   # decoder input
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lm_in, lm_in_lengths = self.prepare_prompt(
            hyp_in, hyp_in_lengths, res_in_pad, res_in_lengths
        )
        mask = (~make_pad_mask(lm_in_lengths)).to(lm_in.device).int()

        args = {
            "input_ids": lm_in,
            "attention_mask": mask,
            "use_cache": False,
            "output_hidden_states": True,
            "return_dict": True,
        }

        output = self.lm(**args).hidden_states[-1]

        if self.template_prompt is None:
            return output, res_in_lengths
        else:
            ret = []
            for i, o in enumerate(output):  # o 的 shape: [T_i, H]
                # lm_in_lengths[i]       ← 整段輸入 (prefix+hyp+suffix+res) 的長度
                # res_in_lengths[i]     ← 真正回應 (res) 的長度
                # 所以起始位置 = lm_in_lengths[i] - res_in_lengths[i]
                # 只保留最後 res_in_lengths[i] 個時間步的 hidden state
                ret.append(o[lm_in_lengths[i] - res_in_lengths[i]: lm_in_lengths[i]])   # shape: [res_in_lengths[i], H]
            # 把各 sample 長度可能不同的 ret（list of [L_i, H]）pad 到同一最大長度 R_max
            # pad_list(..., 0.0) 會用 0.0 填充缺失的時間步
            return pad_list(ret, 0.0), res_in_lengths

    def prepare_prompt_for_inference(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.template_prompt is None:
            lm_in_pad, lm_in_lengths = res_in_pad, res_in_lengths
        else:
            prefix_ids = res_in_pad[0].new(
                self.template_prefix_ids
            ).repeat(len(hyp_in), 1)
            suffix_ids = res_in_pad[0].new(
                self.template_suffix_ids
            ).repeat(len(hyp_in), 1)

            if isinstance(hyp_in[0], str):
                hyp_id = self.tokenizer(
                    hyp_in[0], return_tensors="pt"
                ).input_ids[0][1:].to(res_in_pad.device) # remove sos
                hyp_ids = hyp_id.repeat(len(hyp_in), 1)
            else:
                hyp_ids = torch.stack(hyp_in)

            lm_in = torch.cat(
                (prefix_ids, hyp_ids, suffix_ids, res_in_pad),
                dim=-1,
            )
            lm_in_lengths = (
                prefix_ids.size(0)
                + hyp_in_lengths
                + suffix_ids.size(0)
                + res_in_lengths
            )

        return lm_in, lm_in_lengths

    def forward_inference(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
        log_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(res_in_lengths == res_in_lengths[0])

        lm_in, lm_in_lengths = self.prepare_prompt_for_inference(
            hyp_in, hyp_in_lengths, res_in_pad, res_in_lengths
        )
        mask = (~make_pad_mask(lm_in_lengths)).to(lm_in.device).int()

        args = {
            "input_ids": lm_in,
            "attention_mask": mask,
            "use_cache": False,
            "output_hidden_states": not log_softmax,
            "return_dict": True,
        }

        output = self.lm(**args)

        if log_softmax:
            output = torch.log_softmax(output.logits, dim=-1)
        else:
            output = output.hidden_states[-1]

        if self.template_prompt is None:
            return output, res_in_lengths
        else:
            return output[:, -res_in_lengths[0]:], res_in_lengths

    def forward_inference_cached(
        self,
        hyp_in: Union[List[torch.Tensor], List[str], None],
        hyp_in_lengths: Union[torch.Tensor, None],
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
        log_softmax: bool = False,
        cache: List[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        assert torch.all(res_in_lengths == res_in_lengths[0])

        if cache is None:
            lm_in, lm_in_lengths = self.prepare_prompt_for_inference(
                hyp_in, hyp_in_lengths, res_in_pad, res_in_lengths
            )
            cache_position = torch.arange(
                lm_in.shape[1]
            ).long().to(lm_in.device)
        else:
            lm_in = res_in_pad[:, -1].unsqueeze(-1)

            new_cache = []
            for layer, x in enumerate(cache):
                new_cache.append(
                    (
                        torch.cat(
                            [
                                self.prefix_cache[layer][0].repeat(x[0].shape[0], 1, 1, 1),
                                x[0]
                            ],
                            dim=2
                        ),
                        torch.cat(
                            [
                                self.prefix_cache[layer][1].repeat(x[1].shape[0], 1, 1, 1),
                                x[1]
                            ],
                            dim=2
                        )
                    )
                )
            cache = new_cache

            cache_position = torch.Tensor(
                [cache[0][0].shape[-2]]
            ).long().to(lm_in.device)

        args = {
            "input_ids": lm_in,
            # "attention_mask": mask,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": not log_softmax,
            "return_dict": True,
            "cache_position": cache_position,
        }

        output = self.lm(**args)

        # (32 ,2, (batch, head, len, dim))
        past_key_values = output.past_key_values
        if cache is None:
            self.prefix_cache = past_key_values
            new_past_key_values = []
            for x in past_key_values:
                batch_size, head_size, _, hdim = x[0].shape
                new_past_key_values.append(
                    (
                        torch.empty((batch_size, head_size, 0, hdim)).to(lm_in.device),
                        torch.empty((batch_size, head_size, 0, hdim)).to(lm_in.device)
                    )
                )
            past_key_values = new_past_key_values
        else:
            prefix_len = self.prefix_cache[0][0].shape[2]
            new_past_key_values = []
            for x in past_key_values:
                new_past_key_values.append(
                    (x[0][:, :, prefix_len:], x[1][:, :, prefix_len:])
                )
            past_key_values = new_past_key_values

        if log_softmax:
            output = torch.log_softmax(output.logits, dim=-1)
        else:
            output = output.hidden_states[-1]

        return output[:, -1].unsqueeze(1), past_key_values

    def output_size(self) -> int:
        """Get the output size."""
        return self.lm.config.hidden_size
