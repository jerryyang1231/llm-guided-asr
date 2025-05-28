#!/usr/bin/env python3
from espnet2.tasks.asr import ASRTask
import logging, sys
logger = logging.getLogger()                    # 取得 root logger
handler = logging.StreamHandler(sys.stdout)     # 建立一個把訊息輸出到 stdout 的 handler
handler.setLevel(logging.INFO)                  # 設定這個 handler 接受 INFO 以上等級
logger.addHandler(handler)                      # 把它掛到 root logger
logger.setLevel(logging.INFO)                   # 把 root logger 的等級也調到 INFO


def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
