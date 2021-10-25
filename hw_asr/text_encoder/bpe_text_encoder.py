import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor
import youtokentome
import kenlm
from pyctcdecode import build_ctcdecoder

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class BpeTextEncoder(CTCCharTextEncoder):
    EMPTY_TOK = '<PAD>'

    def __init__(self, alphabet, bpe):
        super().__init__(alphabet)
        self.bpe = bpe
        self.ind2char = {}
        self.char2ind = {}
        for a in alphabet:
            self.ind2char[bpe.subword_to_id(a)] = a
            self.char2ind[a] = bpe.subword_to_id(a)
        self.vocab_for_beam_search = [self.ind2char[i] for i in range(len(self.ind2char))]
        self.beam_search = build_ctcdecoder(self.vocab_for_beam_search, 'lm.binary')


    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode(text)).unsqueeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}' with BPE'")

    @classmethod
    def get_simple_alphabet(cls, args):
        model_path = args['model_path']
        if args['train']:
            bpe = youtokentome.BPE.train(data=args['train_data'],
                                         vocab_size=args['vocab_size'],
                                         model=model_path)
        else:
            bpe = youtokentome.BPE(model=model_path)
        return cls(alphabet=bpe.vocab(), bpe=bpe)
