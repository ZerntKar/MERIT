import torch

from merit.config import MeritConfig
from merit.losses import merit_loss, pseudo_evidence_loss, stability_loss
from merit.model import MERITReranker, inject_pseudo_evidence
from merit.retriever import SASRec, force_training_target_into_pool


def test_score_is_exact_sum_of_evidence_contributions():
    config = MeritConfig()
    config.data.candidate_pool_size = 8
    config.mceb.semantic_dim = 6
    config.mceb.hidden_dim = 16
    config.mceb.num_heads = 4
    config.mceb.ffn_dim = 32
    config.mceb.evidence_budget = 3
    support = torch.rand(2, 8, 7)
    preference = torch.rand(2, 7)
    semantic = torch.rand(2, 7, 6)
    matching = torch.rand(2, 8, 3)
    candidate_mask = torch.ones(2, 8, dtype=torch.bool)
    evidence_mask = torch.ones(2, 7, dtype=torch.bool)
    model = MERITReranker(config)
    output = model(
        support,
        preference,
        semantic,
        matching,
        candidate_mask,
        evidence_mask,
    )
    assert torch.allclose(output.scores, output.contributions.sum(-1), atol=1e-6)
    losses = merit_loss(output, torch.tensor([1, 2]), 1.0, 0.01, 0.01)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_sasrec_retrieval_and_training_pool_replacement():
    config = MeritConfig()
    config.data.max_sequence_length = 5
    retriever = SASRec(12, config)
    sequence = torch.tensor([[1, 2, 3, 0, 0]])
    ids, scores = retriever.retrieve(sequence, candidate_count=5)
    assert not set(ids[0].tolist()) & {0, 1, 2, 3}
    forced_ids, forced_scores, target_index = force_training_target_into_pool(
        ids, scores, torch.tensor([3]), torch.tensor([-4.25])
    )
    assert forced_ids[0, target_index[0]].item() == 3
    assert forced_scores[0, target_index[0]].item() == -4.25
    assert torch.isfinite(forced_scores).all()


def test_fixed_budget_and_strict_pseudo_reassignment():
    config = MeritConfig()
    config.data.candidate_pool_size = 4
    config.mceb.semantic_dim = 6
    config.mceb.hidden_dim = 16
    config.mceb.num_heads = 4
    config.mceb.ffn_dim = 32
    config.mceb.evidence_budget = 3
    model = MERITReranker(config)
    with torch.no_grad():
        try:
            model(
                torch.rand(1, 4, 2),
                torch.rand(1, 2),
                torch.rand(1, 2, 6),
                torch.rand(1, 4, 3),
                torch.ones(1, 4, dtype=torch.bool),
                torch.ones(1, 2, dtype=torch.bool),
                augment_pseudo=False,
                compute_stability=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("fixed budget violation was not rejected")
    torch.manual_seed(7)
    support = torch.arange(6).float().view(1, 1, 6).expand(1, 4, 6)
    preference = torch.arange(6).float().view(1, 6)
    semantic = torch.eye(6).view(1, 6, 6)
    mask = torch.ones(1, 6, dtype=torch.bool)
    augmented_support, augmented_preference, augmented_semantic, _, fake_mask = (
        inject_pseudo_evidence(support, preference, semantic, mask, 0.5)
    )
    labels = augmented_semantic[0, 6:].argmax(dim=-1)
    support_sources = augmented_support[0, 0, 6:].long()
    preference_sources = augmented_preference[0, 6:].long()
    valid = fake_mask[0, 6:]
    assert torch.all(labels[valid] != support_sources[valid])
    assert torch.all(labels[valid] != preference_sources[valid])
    assert torch.all(support_sources[valid] != preference_sources[valid])


def test_pseudo_evidence_does_not_change_real_evidence_ranking():
    torch.manual_seed(13)
    config = MeritConfig()
    config.data.candidate_pool_size = 4
    config.mceb.semantic_dim = 6
    config.mceb.hidden_dim = 16
    config.mceb.num_heads = 4
    config.mceb.ffn_dim = 32
    config.mceb.evidence_budget = 3
    model = MERITReranker(config).eval()
    support = torch.rand(1, 4, 6)
    preference = torch.rand(1, 6)
    semantic = torch.rand(1, 6, 6)
    matching = torch.rand(1, 4, 3)
    candidate_mask = torch.ones(1, 4, dtype=torch.bool)
    evidence_mask = torch.ones(1, 6, dtype=torch.bool)
    clean = model(
        support,
        preference,
        semantic,
        matching,
        candidate_mask,
        evidence_mask,
        augment_pseudo=False,
        compute_stability=False,
    )
    augmented = model(
        support,
        preference,
        semantic,
        matching,
        candidate_mask,
        evidence_mask,
        augment_pseudo=True,
        compute_stability=False,
    )
    assert torch.allclose(clean.scores, augmented.scores)
    assert torch.allclose(clean.contributions, augmented.contributions)
    assert torch.allclose(clean.activation_probabilities, augmented.activation_probabilities)
    assert augmented.augmented_probabilities is not None
    assert augmented.augmented_probabilities.shape == augmented.fake_mask.shape
    assert augmented.augmented_gate.shape == augmented.fake_mask.shape
    assert augmented.fake_mask.any()
    assert torch.isfinite(pseudo_evidence_loss(augmented))


def test_dropout_does_not_create_stability_penalty_without_neighbor_change():
    torch.manual_seed(17)
    config = MeritConfig()
    config.data.candidate_pool_size = 2
    config.matching.neighbors = 1
    config.mceb.semantic_dim = 6
    config.mceb.hidden_dim = 16
    config.mceb.num_heads = 4
    config.mceb.ffn_dim = 32
    config.mceb.evidence_budget = 1
    config.mceb.dropout = 0.5
    model = MERITReranker(config).train()
    output = model(
        torch.rand(1, 2, 3),
        torch.rand(1, 3),
        torch.rand(1, 3, 6),
        torch.rand(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
        augment_pseudo=False,
        compute_stability=True,
    )
    assert model.mceb.encoder.training
    assert torch.equal(
        output.stability_reference_probabilities, output.activation_probabilities
    )
    assert torch.allclose(stability_loss(output), torch.tensor(0.0), atol=1e-7)
