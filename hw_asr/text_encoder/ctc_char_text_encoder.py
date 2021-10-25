from typing import List, Tuple

import torch
import kenlm
from pyctcdecode import build_ctcdecoder
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
#         self.vocab_for_beam_search = [self.ind2char[i] for i in range(len(self.ind2char))]
#         self.beam_search = build_ctcdecoder(self.vocab_for_beam_search)

    def ctc_decode(self, inds: List[int]) -> str:
        res = ''
        prev_token = self.EMPTY_TOK
        for i in range(len(inds)):
            c = self.ind2char[inds[i]]
            if c == self.EMPTY_TOK and prev_token == c:
                continue
            if c != prev_token and c != self.EMPTY_TOK:
                res += c
            prev_token = c
        return res


    def ctc_beam_search(self, probs, beam_width=1):
        return self.beam_search.decode_beams(probs, beam_width=beam_width)[0][0]
        
        
