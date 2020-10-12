import torch


def batchwise_accuracy(lang_output, visn_output, lang_mask):
    """
    Calculate the accuracy of contextual word retrieval, average by batch.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :return:
    """
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)                  # [b, 1, dim]

    # The score of negative pairs. Note that the diagonal is actually the positive score,
    # but it would be zero-graded in calculating the loss below.
    negative_scores = (lang_output.reshape(batch_size, 1, lang_len, dim) *
                       visn_output.reshape(1, batch_size, 1, dim)).sum(-1)    # [b(lang), b(visn), max_len]
    # negative_scores = torch.einsum('ikd,jd->ijk', lang_output, visn_output)

    max_neg_score, max_neg_idx = negative_scores.max(1)        # [batch, max_len], the batch_idx of max-aligned img
    pos_idx = torch.arange(0, batch_size, dtype=torch.int64).to(lang_output.device)

    correct = (pos_idx.unsqueeze(1) == max_neg_idx)
    bool_lang_mask = lang_mask.type(correct.dtype)
    correct = correct * bool_lang_mask
    correct_num = correct.sum()

    accuracy = correct_num * 1. / bool_lang_mask.sum()

    return accuracy


def batchwise_recall(lang_output, visn_output, lang_mask, recalls=(1,)):
    """
    Calculate the accuracy of contextual word retrieval, average by batch.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param recall: a list, which are the number of recalls to be evaluated.
    :return:
    """
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)                  # [b, 1, dim]

    # The score of positive pairs
    positive_score = (lang_output * visn_output).sum(-1)    # [b, max_len]

    # The score of negative pairs. Note that the diagonal is actually the positive score,
    # but it would be zero-graded in calculating the loss below.
    negative_scores = (lang_output.reshape(batch_size, 1, lang_len, dim) *
                       visn_output.reshape(1, batch_size, 1, dim)).sum(-1)    # [b(lang), b(visn), max_len]
    # negative_scores = torch.einsum('ikd,jd->ijk', lang_output, visn_output)

    result = {}
    for recall in recalls:
        kthscore, kthidx = torch.kthvalue(negative_scores, batch_size - recall, dim=1)     # [b, max_len]
        # print(kthscore.shape) print(positive_score.shape)
        correct = (positive_score >= kthscore)                                # [b, max_len]
        bool_lang_mask = lang_mask.type(correct.dtype)
        correct = correct * bool_lang_mask
        correct_num = correct.sum()
        # print(correct_num)
        # print(bool_lang_mask.sum())
        result[recall] = (correct_num * 1. / bool_lang_mask.sum()).item()

    return result
