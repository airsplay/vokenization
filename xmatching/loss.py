import torch


def hinge(x):
    return torch.clamp(x, min=0.)


def paired_hinge_rank_loss(
        lang_output: torch.Tensor,
        visn_output: torch.Tensor,
        lang_mask: torch.Tensor,
        margin: float,
):
    """
    Consider the first half as positive and the second half as negative.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param margin: margin in the ranking loss
    :return: a scalar loss
    """
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]
    assert margin > 0.

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)      # [b, 1, hid_dim]

    # Split to positive and negative sets.
    half_batch_size = batch_size // 2
    pos_lang, neg_lang = torch.split(lang_output, half_batch_size, dim=0)
    pos_visn, neg_visn = torch.split(visn_output, half_batch_size, dim=0)

    # Calculate positive and negative scores.
    true_pos_score = (pos_lang * pos_visn).sum(-1)           # [batch_size / 2, max_len]
    true_neg_score = (neg_lang * neg_visn).sum(-1)           # [batch_size / 2, max_len]
    false_pos_score = (pos_lang * neg_visn).sum(-1)          # [batch_size / 2, max_len]
    false_neg_score = (neg_lang * pos_visn).sum(-1)          # [batch_size / 2, max_len]

    # Hinge Loss
    float_lang_mask = lang_mask.type(lang_output.dtype)      # Either fp16 or fp32
    pos_lang_mask, neg_lang_mask = torch.split(float_lang_mask, half_batch_size, dim=0)
    pos_loss = hinge(margin - true_pos_score + false_pos_score) * pos_lang_mask
    neg_loss = hinge(margin - true_neg_score + false_neg_score) * neg_lang_mask

    # Averaging
    cnt = float_lang_mask.sum()    # Number of words.
    loss = (pos_loss.sum() + neg_loss.sum()) / cnt

    return loss


def batchwise_hinge_rank_loss(
        lang_output: torch.Tensor,
        visn_output: torch.Tensor,
        lang_mask: torch.Tensor,
        margin: float,
):
    """
    Consider all un-matched pairs in the batch as negative samples.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param margin: margin in the ranking loss
    :return: a scalar loss
    """
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]
    assert margin > 0.

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)                  # [b, 1, dim]

    # The score of positive pairs
    positive_score = (lang_output * visn_output.unsqueeze(1)).sum(-1)    # [b, max_len]

    # The score of negative pairs. Note that the diagonal is actually the positive score,
    # but it would be zero-graded in calculating the loss below.
    negative_scores = (lang_output.reshape(batch_size, 1, lang_len, dim) *
                       visn_output.reshape(1, batch_size, 1, dim)).sum(-1)    # [b(lang), b(visn), max_len]
    # negative_scores = torch.einsum('ikd,jd->ijk', lang_output, visn_output)

    # Calculate of the hinge rank loss, let me explain why it works:
    # For the diagonal, the scores are for positive, we thus create a positive_mask to neglect these scores.
    #   max(0., margin - x^T x + (x^T x - 2 margin) )
    # = max(0., -margin)
    # = 0.      , since we have made sure that margin > 0
    # During backwards, the operator max(0., -margin) would raise a grad of 0 to the operand "-margin",
    #   thus it is just what we want.
    float_lang_mask = lang_mask.type(lang_output.dtype)       # Either fp16 or fp32
    positive_mask = torch.eye(batch_size)
    negative_scores = negative_scores - positive_mask.unsqueeze(-1) * margin * 2
    lang_loss = hinge(margin - positive_score.unsqueeze(1) + negative_scores) * float_lang_mask.unsqueeze(1)
    visn_loss = hinge(margin - positive_score.unsqueeze(0) + negative_scores) * float_lang_mask.unsqueeze(1)

    # Averaging
    # Each sentence is duplicated by batch_size thus the total length is also multiplied by this term.
    cnt = max(float_lang_mask.sum() * batch_size, 1.)    # Number of words.
    lang_loss = lang_loss.sum() / cnt
    visn_loss = visn_loss.sum() / cnt

    return lang_loss + visn_loss


