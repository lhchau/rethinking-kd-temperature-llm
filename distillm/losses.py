"""Token-level distillation objectives compared in the paper.

Every loss takes student logits and teacher logits of shape [B, L, V] and the
labels [B, L] (-100 marks positions to ignore), and averages over the valid
positions. `tau` is the KD temperature: both distributions are softened as
softmax(z / tau) and the loss is scaled by tau^2 (Hinton et al., 2015).
Implementations follow DistiLLM (Ko et al., 2024) and the original papers.
"""
import torch
import torch.nn.functional as F


def _valid(logits, teacher_logits, labels):
    """Flatten to [N, V], keeping only the N positions that have a label."""
    valid = labels.reshape(-1) != -100
    return logits.reshape(-1, logits.size(-1))[valid], teacher_logits.reshape(-1, teacher_logits.size(-1))[valid]


def forward_kl(logits, teacher_logits, labels, tau=1.0):
    """FKL: KL(p || q), written as cross-entropy (the teacher entropy is a constant)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits / tau, dim=-1, dtype=torch.float32)
    x = torch.masked_fill(teacher_probs * student_logprobs, torch.isinf(logits), 0).sum(-1)
    return -x.mean() * tau ** 2


def reverse_kl(logits, teacher_logits, labels, tau=1.0):
    """RKL: KL(q || p)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    student_probs = F.softmax(logits / tau, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits / tau, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    x = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    x -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    return -x.sum(-1).mean() * tau ** 2


def symmetric_kl(logits, teacher_logits, labels, tau=1.0, lam=0.5):
    """Sym-KL: (1 - lam) * FKL + lam * RKL."""
    return (1 - lam) * forward_kl(logits, teacher_logits, labels, tau) + lam * reverse_kl(logits, teacher_logits, labels, tau)


def js_divergence(logits, teacher_logits, labels, tau=1.0, lam=0.9):
    """Generalized JSD with mixture m = (1 - lam) * p + lam * q (Agarwal et al., 2024)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits / tau, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits / tau, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    x = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    x -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    loss = lam * -x.sum(-1).mean()
    x = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    loss += (1 - lam) * -x.sum(-1).mean()
    return loss * tau ** 2


def skewed_forward_kl(logits, teacher_logits, labels, tau=1.0, lam=0.1):
    """SFKL: KL(p || lam * p + (1 - lam) * q) (Ko et al., 2024)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits / tau, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(lam * teacher_probs + (1 - lam) * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    x = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0).sum(-1)
    return -x.mean() * tau ** 2


def skewed_reverse_kl(logits, teacher_logits, labels, tau=1.0, lam=0.1):
    """SRKL: KL(q || (1 - lam) * p + lam * q) (Ko et al., 2024)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits / tau, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits / tau, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    x = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    x -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    return -x.sum(-1).mean() * tau ** 2


def _akl_ratio(teacher_logits, logits, mu=0.5):
    """Head/tail error split of AKL: the head is the smallest set of top teacher tokens holding mass mu."""
    teacher_probs = F.softmax(torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0), dim=-1, dtype=torch.float32)
    student_probs = F.softmax(torch.masked_fill(logits, torch.isinf(logits), 0), dim=-1, dtype=torch.float32).detach()
    sorted_teacher, idx = teacher_probs.sort(dim=-1, descending=True)
    errors = torch.abs(sorted_teacher - student_probs.gather(dim=-1, index=idx))
    tail = torch.cumsum(sorted_teacher, dim=-1) > mu
    tail[:, 0] = False  # the top token always belongs to the head
    head_err = torch.masked_fill(errors, tail, 0.0).sum(-1)
    tail_err = torch.masked_fill(errors, ~tail, 0.0).sum(-1)
    return head_err / (head_err + tail_err), tail_err / (head_err + tail_err)


def _weighted_kl(target_logits, logits, inf_mask, weight, tau):
    """sum_k p_k (log p_k - log q_k) with p from target_logits, weighted per position."""
    target_probs = F.softmax(target_logits / tau, dim=-1, dtype=torch.float32)
    target_logprobs = F.log_softmax(target_logits / tau, dim=-1, dtype=torch.float32)
    logprobs = F.log_softmax(logits / tau, dim=-1, dtype=torch.float32)
    target_x = torch.masked_fill(target_probs * target_logprobs, inf_mask, 0).sum(-1)
    x = torch.masked_fill(target_probs * logprobs, inf_mask, 0).sum(-1)
    return ((target_x - x) * weight).mean()


def adaptive_kl(logits, teacher_logits, labels, tau=1.0):
    """AKL: head-weighted FKL + tail-weighted RKL (Wu et al., 2025)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    inf_mask = torch.isinf(logits)
    head, tail = _akl_ratio(teacher_logits, logits)
    loss = _weighted_kl(teacher_logits, logits, inf_mask, head, tau) + _weighted_kl(logits, teacher_logits, inf_mask, tail, tau)
    return loss * tau ** 2


def ab_divergence(logits, teacher_logits, labels, alpha=0.2, beta=0.7):
    """Alpha-beta divergence (Wang et al., 2025), general case alpha, beta, alpha + beta != 0."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    apb = alpha + beta
    term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
    term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
    term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
    divergence = -(term1 - term2 - term3) / (alpha * beta)
    return torch.where(torch.isfinite(divergence), divergence, 0.0).mean()


def diversity_aware_reverse_kl(logits, teacher_logits, labels, tau=1.0, alpha=1.0, beta=2.0):
    """DRKL (Luong et al., 2026a): alpha * binary target-class RKL + beta * tau^2 * non-target RKL."""
    target = labels.reshape(-1)
    valid = target != -100
    target = target[valid].unsqueeze(1)
    logits, teacher_logits = _valid(logits, teacher_logits, labels)

    # target vs. rest, as a two-class distribution
    q_t = F.softmax(logits, dim=1, dtype=torch.float32).gather(1, target)
    p_t = F.softmax(teacher_logits, dim=1, dtype=torch.float32).gather(1, target)
    q2 = torch.cat([q_t, 1.0 - q_t], dim=1)
    p2 = torch.cat([p_t, 1.0 - p_t], dim=1)
    target_rkl = (q2 * (torch.log(q2.clamp(min=1e-12)) - torch.log(p2.clamp(min=1e-12)))).sum(1).mean()

    # distributions over the non-target tokens (target logit pushed to -1000)
    logits_nt = logits.scatter(1, target, -1000.0)
    teacher_nt = teacher_logits.scatter(1, target, -1000.0)
    inf_mask = torch.isinf(logits_nt) | torch.isinf(teacher_nt)
    q_nt = F.softmax(logits_nt / tau, dim=1, dtype=torch.float32)
    log_q_nt = F.log_softmax(logits_nt / tau, dim=1, dtype=torch.float32)
    log_p_nt = F.log_softmax(teacher_nt / tau, dim=1, dtype=torch.float32)
    non_target_rkl = torch.masked_fill(q_nt * (log_q_nt - log_p_nt), inf_mask, 0).sum(1).mean() * tau ** 2

    return alpha * target_rkl + beta * non_target_rkl


def distillation_loss(args, logits, teacher_logits, labels):
    tau = args.kd_temperature
    if args.loss == "fkl":
        return forward_kl(logits, teacher_logits, labels, tau)
    if args.loss == "rkl":
        return reverse_kl(logits, teacher_logits, labels, tau)
    if args.loss == "symkl":
        return symmetric_kl(logits, teacher_logits, labels, tau)
    if args.loss == "js":
        return js_divergence(logits, teacher_logits, labels, tau)
    if args.loss == "sfkl":
        return skewed_forward_kl(logits, teacher_logits, labels, tau, lam=args.skew_alpha)
    if args.loss == "srkl":
        return skewed_reverse_kl(logits, teacher_logits, labels, tau, lam=args.skew_alpha)
    if args.loss == "akl":
        return adaptive_kl(logits, teacher_logits, labels, tau)
    if args.loss == "ab":
        return ab_divergence(logits, teacher_logits, labels)
    if args.loss == "drkl":
        return diversity_aware_reverse_kl(logits, teacher_logits, labels, tau, args.drkl_alpha, args.drkl_beta)
    raise ValueError(f"unknown loss: {args.loss}")
