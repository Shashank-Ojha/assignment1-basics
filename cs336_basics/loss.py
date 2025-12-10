import torch
from collections.abc import Callable, Iterable
import math


def log_softmax(tensor: torch.Tensor, dim):
    """
    The formula for softmax is e^(x_i) / sum_j e^(x_j)

    The log of this is x_i - log(sum_j e^(x_j))

    We just subtract the max (shift) for numerical stability
    """
    max_values, max_indices = torch.max(tensor, dim=dim, keepdims=True)
    shifted = tensor - max_values

    exp_shifted = torch.exp(shifted)
    sum_exp_shifted = torch.sum(exp_shifted, dim=dim, keepdims=True)
    log_sum_exp_shifted = torch.log(sum_exp_shifted)

    return shifted - log_sum_exp_shifted


def cross_entropy_loss(pred_logits, targets):
    """
    Args:
        pred_logits: (..., classes)
        targets: (..., )
    """
    # We need to take the log of the softmax eventually anyways to compute the loss,
    # so we optimize this by directly computing log(a) - log(b) instead of first
    # computing a/b and then doing the log of that.
    #
    # This is more numerically stable because before we did
    # torch.exp(x_i - shift) where x_i - shift can be a very large negative number.
    # This results in 0. Then when we do torch.log(0) we get -inf. Note that this
    # doesn't happen in the denominator because in the sum there there exists an xj = shift
    # so e^0 = 1 and that means the denominator is at least 1 and the log is stable.

    # Now exp and log cancel out and so we directly return x_i - shift as part of the answer.
    normalized_logits = log_softmax(pred_logits, -1)
    prob_of_target = normalized_logits.gather(dim=-1, index=targets.unsqueeze(-1))
    return -torch.mean(prob_of_target)


# Provided for illustrative purposes.
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


BETAS_FOR_LLMS = (0.9, 0.95)


# This is what we implemented and what we will actually use to train our LLM.
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.99,
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        # Use a global step counter for all parameters in this optimizer
        if not hasattr(self, "global_step"):
            self.global_step = 1
        else:
            self.global_step += 1

        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            (b1, b2) = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Retrieve state.
                state = self.state[p]
                first_moment = state.get("first_moment", 0.0)
                second_moment = state.get("second_moment", 0.0)

                # Get gradient.
                grad = p.grad.data

                # Compute new moments.
                first_moment = b1 * first_moment + (1 - b1) * grad
                second_moment = b2 * second_moment + (1 - b2) * (grad**2)

                # Compute step size using the global step
                t = self.global_step
                lr_for_time_step = lr * math.sqrt(1 - b2**t) / (1 - b1**t)

                # Update parameters.
                p.data -= lr_for_time_step * first_moment / (torch.sqrt(second_moment) + eps)
                p.data = (1 - lr * weight_decay) * p.data

                # Update state, no need to store t per parameter anymore
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment

        return loss


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        # This slowly increases over time until it reaches the max_learning rate.
        return (it / warmup_iters) * max_learning_rate
    if warmup_iters <= it <= cosine_cycle_iters:
        delta_lr = max_learning_rate - min_learning_rate
        # Larges values means higher frequency which means the wave repeats more frequently. Period is 2 * pi / |freq|.
        # Based on this formula, we should see a period in 2 * (cosine_cycle_iters - warmup_iters) steps, but we only
        # run it for (cosine_cycle_iters - warmup_iters) steps, so we see have a period starting from the peak to the trough.
        # The 1 + math.cos() makes it so we range from [2.0, 0.0] instead of [1.0, -1.0]. The 0.5 factor then updates the range
        # to [1.0. 0.0] which gives us a smooth descent from the max_learning_rate to the min_learning_rate.
        angular_freq = math.pi / (
            cosine_cycle_iters - warmup_iters
        )  # This basically means we have 1/2 a period between
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) * angular_freq)) * delta_lr

    return min_learning_rate


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    EPS = 1e-6
    grads = [p for p in parameters if p.grad is not None]

    # Compute l2 norm summing across all gradients.
    total_l2_norm = 0
    for p in grads:
        total_l2_norm += torch.sum(p.grad * p.grad)

    total_l2_norm = torch.sqrt(total_l2_norm)

    # Clip if gradient exceeds max.
    if total_l2_norm > max_l2_norm:
        scale = max_l2_norm / (total_l2_norm + EPS)
        for p in grads:
            p.grad.data *= scale


if __name__ == "__main__":
    LRs = [1e1, 1e2, 1e3]
    for lr in LRs:
        print("Learning rate is", lr)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.

        print("----")
