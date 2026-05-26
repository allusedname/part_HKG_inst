import torch
from partcat_hkg.kg.relations import relation_attributes_from_masks, relation_channel_strengths


def test_relation_features_shape():
    a = torch.zeros(16, 16)
    b = torch.zeros(16, 16)
    a[4:10, 4:10] = 1
    b[8:14, 8:14] = 1
    gamma = relation_attributes_from_masks(a, b)
    ch = relation_channel_strengths(gamma)
    assert gamma.numel() == 14
    assert ch.numel() == 8
