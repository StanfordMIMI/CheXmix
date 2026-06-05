import json
from functools import lru_cache
from typing import TYPE_CHECKING

import regex as re
from transformers.tokenization_utils_base import TextInput
from transformers.utils import is_torch_available, to_py_obj

if TYPE_CHECKING:
    if is_torch_available():
        import torch

import os
from typing import Dict, List, Tuple, Union, Any, Callable, Optional


import numpy as np
import torch
from transformers import PreTrainedTokenizer, AddedToken
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json",
    },
    "merges_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/codegen-350M-mono": 2048,
}

IMG_TOKEN_SPAN = 1024

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def _list_find(
    input_list: List[Any],
    candidates: Tuple[Any],
    start: int = 0,
):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1


def _replace_closed_tag(
    input_tokens: List[Any],
    start_tags: Union[Any, Tuple[Any]],
    end_tags: Union[Any, Tuple[Any]],
    inclusive_replace_func: Callable,
    exclusive_replace_func: Callable = lambda x: x,
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)

    output_tokens = []
    end = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end:start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed image token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start : end + 1]))
        end += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))
    return output_tokens


class CheXagentTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        image_start_tag="<|img|>",
        image_end_tag="<|/img|>",
        image_pad_tag="<|imgpad|>",
        ref_start_tag="<|ref|>",
        ref_end_tag="<|/ref|>",
        box_start_tag="<|box|>",
        box_end_tag="<|/box|>",
        quad_start_tag="<|quad|>",
        quad_end_tag="<|/quad|>",
        **kwargs,
    ):
        bos_token = (
            AddedToken(bos_token, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, special=True)
            if isinstance(pad_token, str)
            else pad_token
        )
        self.add_bos_token = add_bos_token

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.image_pad_tag = image_pad_tag
        self.ref_start_tag = ref_start_tag
        self.ref_end_tag = ref_end_tag
        self.box_start_tag = box_start_tag
        self.box_end_tag = box_end_tag
        self.quad_start_tag = quad_start_tag
        self.quad_end_tag = quad_end_tag
        self.IMAGE_ST = (
            image_start_tag,
            image_end_tag,
            image_pad_tag,
            ref_start_tag,
            ref_end_tag,
            box_start_tag,
            box_end_tag,
            quad_start_tag,
            quad_end_tag,
        )
        for special_token in self.IMAGE_ST:
            if special_token not in self.get_vocab():
                self.add_special_tokens({"additional_special_tokens": [special_token]})
        for coordinate in range(10):
            if f"<{coordinate}>" not in self.get_vocab():
                self.add_special_tokens(
                    {"additional_special_tokens": [f"<|coord_{coordinate}|>"]}
                )
        if len(self) % 64 != 0:
            for extra in range(((len(self) // 64) + 1) * 64 - len(self)):
                if f"<extra_{extra}>" not in self.get_vocab():
                    self.add_special_tokens(
                        {"additional_special_tokens": [f"<|extra_{extra}|>"]}
                    )
        self.img_start_id = self.convert_tokens_to_ids(self.image_start_tag)
        self.img_end_id = self.convert_tokens_to_ids(self.image_end_tag)
        self.img_pad_id = self.convert_tokens_to_ids(self.image_pad_tag)
        self.ref_start_id = self.convert_tokens_to_ids(self.ref_start_tag)
        self.ref_end_id = self.convert_tokens_to_ids(self.ref_end_tag)
        self.box_start_id = self.convert_tokens_to_ids(self.box_start_tag)
        self.box_end_id = self.convert_tokens_to_ids(self.box_end_tag)
        self.quad_start_id = self.convert_tokens_to_ids(self.quad_start_tag)
        self.quad_end_id = self.convert_tokens_to_ids(self.quad_end_tag)
        self.chat_template = DEFAULT_CHAT_TEMPLATE

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        def _encode_imgurl(img_tokens):
            assert (
                img_tokens[0] == self.image_start_tag
                and img_tokens[-1] == self.image_end_tag
            )
            img_tokens = img_tokens[1:-1]
            img_url = "".join(img_tokens)
            out_img_tokens = list(img_url)
            if len(out_img_tokens) > IMG_TOKEN_SPAN:
                raise ValueError(
                    "The content in {}..{} is too long".format(
                        self.image_start_tag, self.image_end_tag
                    )
                )
            out_img_tokens.extend(
                [self.image_pad_tag] * (IMG_TOKEN_SPAN - len(out_img_tokens))
            )
            out_img_tokens = (
                [self.image_start_tag] + out_img_tokens + [self.image_end_tag]
            )
            return out_img_tokens

        tokens = super().tokenize(text, **kwargs)
        tokens = _replace_closed_tag(
            tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl
        )
        return tokens

    def _tokenize(self, text):
        """Tokenize a string."""

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )
        merge_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["merges_file"],
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(
                self.bpe_ranks.items(), key=lambda kv: kv[1]
            ):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        truncate_before_pattern: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):
                A list of regular expression strings that will be used to truncate the returned string. This can be
                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning
                of a new line). An example pattern could be `["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """

        token_ids = to_py_obj(token_ids)

        decoded_text = self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        return decoded_text

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        def _decode_imgurl(img_token_ids):
            assert (
                img_token_ids[0] == self.img_start_id
                and img_token_ids[-1] == self.img_end_id
            )
            img_token_ids = img_token_ids[1:-1]
            img_token_ids = img_token_ids[: img_token_ids.index(self.img_pad_id)]
            return [self.img_start_id] + img_token_ids + [self.img_end_id]

        token_ids = _replace_closed_tag(
            token_ids, self.img_start_id, self.img_end_id, _decode_imgurl
        )

        return super()._decode(
            token_ids,
            skip_special_tokens,
            clean_up_tokenization_spaces,
            spaces_between_special_tokens,
            **kwargs,
        )

    def truncate(self, completion, truncate_before_pattern):
        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        terminals = [
            re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern
        ]

        prints = list(re.finditer("^print", completion, re.MULTILINE))

        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        defs = list(re.finditer("^def", completion, re.MULTILINE))

        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        terminals_pos = [
            pos
            for pos in [
                find_re(completion, terminal, start_pos) for terminal in terminals
            ]
            if pos != -1
        ]

        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion

    def from_list_format(self, list_format: List[Dict]):
        text = ""
        num_images = 0
        for ele in list_format:
            if "image" in ele:
                num_images += 1
                text += f"Picture {num_images}:"
                text += self.image_start_tag + ele["image"] + self.image_end_tag
                text += "\n"
            elif "text" in ele:
                text += ele["text"]
            elif "box" in ele:
                if "ref" in ele:
                    text += self.ref_start_tag + ele["ref"] + self.ref_end_tag
                for box in ele["box"]:
                    text += (
                        self.box_start_tag
                        + "(%d,%d),(%d,%d)" % (box[0], box[1], box[2], box[3])
                        + self.box_end_tag
                    )
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text

    def _fetch_latest_picture(self, response, history):
        if history is None:
            history = []
        _history = history + [(response, None)]
        for q, r in _history[::-1]:
            for ele in self.to_list_format(q)[::-1]:
                if "image" in ele:
                    return ele["image"]
        return None
