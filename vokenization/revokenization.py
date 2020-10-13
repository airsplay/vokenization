# Copyleft 2020 project COL.

from transformers import AutoTokenizer


class ReVokenizer:
    """
    Convert a
    """
    def __init__(self, forward_tokenizer_name, backward_tokenizer_name, vokenizer):
        """
        :args forward_tokenizer:
        :args backward_tokenizer:
        :args vokenizer:
        """
        self.forward_tokenizer = AutoTokenizer.from_pretrained(forward_tokenizer_name, use_fast=True)
        self.backward_tokenizer = AutoTokenizer.from_pretrained(backward_tokenizer_name, use_fast=True)
        self.slow_backward_tokenizer = AutoTokenizer.from_pretrained(backward_tokenizer_name)
        self.vokenizer = vokenizer

        self.prepare_for_unicode()

    def vokenize_sent(self, sents, topk=None):
        pass

    def vokenize_ids(self, input_ids, topk=None, verbose=False):
        """
           backward_input
        <-- Backward Tokenizer
        <--    Sentence   -->
        Forward Tokenizer -->
            forward_input --> Vokenizer --> forward_results
        """
        sents, forward_input, backward_input = self.process(input_ids)
        alignments = self.batch_calculate_alignment(
            forward_input['offset_mapping'],
            backward_input['offset_mapping'],
        )
        forward_results = self.vokenizer.vokenize_ids(
            forward_input['input_ids'], topk
        )
        backward_results = self.batch_map_back(forward_results, alignments)
        if verbose:
        # if True:
            self.show_alignments(
                sents, forward_input, backward_input, alignments,
                input_ids, backward_results)
        return backward_results

    def show_alignments(self, sents, forward_inputs, backward_inputs, alignments, input_ids,
                        backward_results):
        forward_ids = forward_inputs['input_ids']
        forward_offsets = forward_inputs['offset_mapping']
        backward_ids = backward_inputs['input_ids']
        backward_offsets = backward_inputs['offset_mapping']
        _, _, backward_result_tokens, _ = backward_results
        for sent, forward_id, backward_id, forward_offset, backward_offset, alignment, input_id, backward_result_token in zip(
            sents, forward_ids, backward_ids, forward_offsets, backward_offsets, alignments, input_ids, backward_result_tokens
        ):
            print(sent)
            for backward_idx, forward_idx in enumerate(alignment):
                def get_str(l, r):
                    return sent[l: r]
                print("%2d %2d %7s %7s %7s  |  %7s %7s %7s" % (
                    backward_idx, forward_idx,
                    self.backward_tokenizer._convert_id_to_token(input_id[backward_idx]),
                    self.backward_tokenizer._convert_id_to_token(backward_id[backward_idx]),
                    get_str(*backward_offset[backward_idx]),
                    self.forward_tokenizer._convert_id_to_token(forward_id[forward_idx]),
                    backward_result_token[backward_idx + 1],
                    get_str(*forward_offset[forward_idx]),
                ))
            print()

    def show_input(self, sents, forward_inputs, backward_inputs, input_ids):
        forward_ids = forward_inputs['input_ids']
        forward_offsets = forward_inputs['offset_mapping']
        backward_ids = backward_inputs['input_ids']
        backward_offsets = backward_inputs['offset_mapping']

        for sent, forward_id, backward_id, forward_offset, backward_offset, input_id in zip(
                sents, forward_ids, backward_ids, forward_offsets, backward_offsets, input_ids
        ):
            print(sent)
            for i, (backward_i, bo, input_i) in enumerate(zip(backward_id, backward_offset, input_id)):
                print("%7s %7s" % (
                    self.backward_tokenizer._convert_id_to_token(backward_i),
                    self.backward_tokenizer._convert_id_to_token(input_i),
                    # self.forward_tokenizer._convert_id_to_token(forward_i),
                ), bo, sent[bo[0]: bo[1]] if bo is not None else '')
            print()


    def backward_decode(self, input_id):
        # return u''.join(self.backward_tokenizer.convert_ids_to_tokens(input_id)).replace('Ġ', ' ')
        # return self.backward_tokenizer.decode(input_id)
        tokens = self.slow_backward_tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=True)
        # print(tokens)
        return self.slow_backward_tokenizer.convert_tokens_to_string(
            tokens
        )

    def process(self, input_ids):
        """
        :return: two dicts (forward_input, backward_input)
            with keys "input_ids" "offset_mapping"
        """
        sents = [self.backward_decode(input_id) for input_id in input_ids]
        tokenizer_kwargs = {
            'return_token_type_ids': False,
            'return_attention_mask': False,
            'return_offsets_mapping': True,
        }
        # 'add_special_tokens': False,
        forward_input = self.forward_tokenizer.batch_encode_plus(
            sents,
            **tokenizer_kwargs
        )
        backward_input = self.backward_tokenizer.batch_encode_plus(
            sents,
            **tokenizer_kwargs
        )

        # Avoid batch-1
        self._safe_guard(forward_input)
        self._safe_guard(backward_input)

        # Remove <cls> and <sep>
        self._remove_special_tokens(forward_input)
        self._remove_special_tokens(backward_input)

        # postprocessing of the backwards
        self._calibrate_backward_offset(backward_input)
        # self._fix_nouns(backward_input)
        self._fix_length(backward_input, input_ids)

        assert list(map(len, backward_input['input_ids'])) == \
               list(map(len, input_ids)), (list(map(len, backward_input['input_ids'])),
               list(map(len, input_ids)))
        return sents, forward_input, backward_input

    @staticmethod
    def _safe_guard(inputs):
        ids = inputs['input_ids']
        if type(ids[0]) is int:
            for key, value in inputs.items():
                inputs[key] = [value]

    @staticmethod
    def _remove_special_tokens(inputs):
        if type(inputs) is dict:
            for key in inputs:
                inputs[key] = ReVokenizer._remove_special_tokens(inputs[key])
            return inputs
        return [input[1:-1] for input in inputs]

    @staticmethod
    def _fix_nouns(backward_input):
        backward_offsets = backward_input['offset_mapping']
        for backward_offset in backward_offsets:
            last_not_noun_idx = -1
            while backward_offset[last_not_noun_idx] is None:
                last_not_noun_idx -= 1
            for noun_idx in range(last_not_noun_idx + 1, 0):
                backward_offset[noun_idx] = backward_offset[last_not_noun_idx]

    @staticmethod
    def _fix_length(backward_input, input_ids):
        backward_ids = backward_input['input_ids']
        backward_offsets = backward_input['offset_mapping']
        for i in range(len(backward_ids)):
            desired_length = len(input_ids[i])
            if len(backward_ids[i]) > desired_length:
                backward_ids[i] = backward_ids[i][:desired_length]
                backward_offsets[i] = backward_offsets[i][:desired_length]

            while len(backward_ids[i]) < desired_length:
                backward_ids[i].append(backward_ids[i][-1])
                backward_offsets[i].append(backward_offsets[i][-1])

            # print(desired_length)
            # print(len(backward_ids[i]))
            assert desired_length == len(backward_ids[i]) == len(backward_offsets[i])

    def _calibrate_backward_offset(self, backward_input):
        batch_input_ids = backward_input['input_ids']
        batch_new_offset = []
        for input_ids in batch_input_ids:
            now = 0
            byte_list = []
            new_offset = []
            for input_id in input_ids:
                token = self.backward_tokenizer._convert_id_to_token(input_id)
                start = now
                unicode_complete_flag = True
                for char in token:
                    byte = self.c2b[char]
                    byte_list.append(byte)
                    try:
                        unicode_char = bytes(byte_list).decode('utf-8')
                        byte_list = []
                        now += 1
                        unicode_complete_flag = True
                    except UnicodeDecodeError as e:
                        unicode_complete_flag = False
                if unicode_complete_flag:
                    left, right = start, now
                else:
                    left, right = start, now + 1
                new_offset.append((left, right))
            # print(token, sent[left: right].replace(' ', 'Ġ'))
            batch_new_offset.append(new_offset)
        backward_input['offset_mapping'] = batch_new_offset

    def prepare_for_unicode(self):
        def bytes_to_unicode():
            """
            Returns list of utf-8 byte and a mapping to unicode strings.
            We specifically avoids mapping to whitespace/control characters the bpe code barfs on.
            The reversible bpe codes work on unicode strings.
            This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
            When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
            This is a signficant percentage of your normal, say, 32K bpe vocab.
            To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
            """
            bs = (
                    list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
                range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]
            n = 0
            for b in range(2 ** 8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2 ** 8 + n)
                    n += 1
            cs = [chr(n) for n in cs]
            return dict(zip(bs, cs))
        self.b2c = bytes_to_unicode()
        self.c2b = {c: b for b, c in self.b2c.items()}

    def show(self, ids_list):
        print(
            [self.backward_tokenizer.convert_ids_to_tokens(ids) for ids in ids_list]
        )

    @staticmethod
    def batch_map_back(results, alignments):
        if type(results) is tuple:
            # Handle multiple output by the vokenizer
            #   i.e., input_ids, input_scores, ...
            return [ReVokenizer.batch_map_back(one_results, alignments) for one_results in results]
        new_results = []
        for result, alignment in zip(results, alignments):
            # print(result)
            # print(max(alignment), len(result))
            new_results.append(
                [result[0]] + [result[idx + 1] for idx in alignment] + [result[-1]])
            assert max(alignment) < (len(result) - 2)
        return new_results

    @staticmethod
    def batch_calculate_alignment(batch_forward_offsets, batch_backward_offsets):
        """
        for each backward_token indicated by backward offset, align a forward token to it.
        """
        alignments = []
        for forward_offsets, backward_offsets in zip(batch_forward_offsets, batch_backward_offsets):
            alignment = []
            # Backward: I  ha ve a lov ely  c at.
            # Sent:     I  have  a lovely   cat
            # Forward:  I  hav e a lo ve ly cat.
            now_idx = 0
            for backward_offset in backward_offsets:
                best_idx = now_idx
                best_iou = IoU(forward_offsets[best_idx], backward_offset)
                while (now_idx + 1 < len(forward_offsets)) and \
                      (forward_offsets[now_idx][1] < backward_offset[1]):
                    now_idx += 1
                    now_iou = IoU(forward_offsets[now_idx], backward_offset)
                    if now_iou > best_iou:
                        best_idx = now_idx
                        best_iou = now_iou
                alignment.append(best_idx)
            alignments.append(alignment)
        return alignments


def IoU(a, b):
    x1, y1 = a
    x2, y2 = b
    len1 = y1 - x1
    len2 = y2 - x2
    I = max(min(y1, y2) - max(x1, x2), 0)
    U = len1 + len2 - I
    return I / max(U, 1)


if __name__ == "__main__":
    revokenizer = ReVokenizer('bert-base-uncased', 'roberta-base', None)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    sents = ['Do not panic. ',
             ' iso have a dream .',
             ' This is a test???',
             'Congratulations to the LiLT Founder and CEO, @stanfordnlp grad, Spence Green!',
             'Ay congrats Ethan! An awesome crew, well deserved',
             ' By the fourth season, fewer than three million viewers tuned in each week despite what some fans and critics considered an increase in episode quality.',
             'Filming of the final episode began on Friday, February 25, after the first half of the day was spent completing "Terra Prime". Principal photography took eight days to complete, one day longer than usual. ',
             'sda asdo weij sdjf oweif bqosdj weorasd.?SdfasXX...',
             ]

    ids = [tokenizer.encode(sent, add_special_tokens=False) for sent in sents]
    print(sents)
    sents = [tokenizer.decode(idx) for idx in ids]
    print(sents)
    revokenizer.vokenize_ids(ids)

