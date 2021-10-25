import argparse
import json
from pathlib import Path
import numpy as np

import torch
from tqdm import tqdm

from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import hw_asr.model as module_model
import hw_asr.loss as module_loss
import hw_asr.text_encoder as module_text_enc
import hw_asr.metric as module_metric
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # text_encoder
    text_encoder_class = getattr(module_text_enc, config['text_encoder']['type'])
    text_encoder = text_encoder_class.get_simple_alphabet(config['text_encoder']['args'])

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    # get function handles of loss and metrics
    print(config["loss"])
    loss_fn = getattr(module_loss, config["loss"]['type'])
    metric_fns = [getattr(module_metric, met['type']) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []
    cer = 0
    wer = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloaders["val"])):
            batch = Trainer.move_batch_to_device(batch, device)
            batch["log_probs"] = model(**batch)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["log_probs"] = batch["log_probs"].detach().cpu().numpy()
            batch["probs"] = np.exp(batch["log_probs"])
            batch["argmax"] = batch["probs"].argmax(-1)
            for i in range(len(batch["text"])):
                gt = batch["text"][i]
                pred = text_encoder.ctc_decode(batch["argmax"][i])
                beam_search_pred = text_encoder.ctc_beam_search(batch["log_probs"][i], beam_width=300)
                beam_search_pred = beam_search_pred.replace(' ', '▁')
                cer_cur = calc_cer(gt, beam_search_pred)
                wer_cur = calc_wer(gt, beam_search_pred)
                results.append({
                    "ground_trurh": gt,
                    "pred_text_argmax": pred,
                    "cer" : cer_cur,
                    "wer" : wer_cur,
                    "pred_text_beam_search": beam_search_pred

                })
                cer += cer_cur
                wer += wer_cur
    print("CER: ", cer / len(dataloaders["val"]))
    print("WER: ", wer / len(dataloaders["val"]))
    
    with Path(out_file).open('w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default='output.json',
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader"
    )
    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args.output)
