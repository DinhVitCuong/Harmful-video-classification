
import torch
import wandb


def summarize_and_log_weights(model,epoch):
    weight_means, weight_maxs, weight_mins = [], [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Summarize weights
            weight = param.data.detach()
            weight_means.append(weight.mean().item())
            weight_maxs.append(weight.max().item())
            weight_mins.append(weight.min().item())
            
            # Summarize gradients
            wandb.log({
                f"Weights/{name}": wandb.Histogram(param.data.cpu().numpy()),
                "Epoch": epoch + 1
            })

    # Calculate overall summaries for weights
    weight_summary = {
        "Weight Mean": torch.tensor(weight_means).mean().item(),
        "Weight Max": torch.tensor(weight_maxs).max().item(),
        "Weight Min": torch.tensor(weight_mins).min().item()
    }

    # Log to W&B
    wandb.log({
        **weight_summary,
        "Epoch": epoch + 1
    })

    return weight_summary

def summarize_and_log_gradients(model, epoch):
    grad_means, grad_maxs, grad_mins = [], [], []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Summarize gradients
            if param.grad is not None:
                grad = param.grad.detach()
                grad_means.append(grad.mean().item())
                grad_maxs.append(grad.max().item())
                grad_mins.append(grad.min().item())
            wandb.log({
                f"Gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()) if param.grad is not None else None,
                "Epoch": epoch + 1
            })

    # Calculate overall summaries for gradients
    grad_summary = {
        "Gradient Mean": torch.tensor(grad_means).mean().item() if grad_means else 0.0,
        "Gradient Max": torch.tensor(grad_maxs).max().item() if grad_maxs else 0.0,
        "Gradient Min": torch.tensor(grad_mins).min().item() if grad_mins else 0.0
    }

    # Log to W&B
    wandb.log({
        **grad_summary,
        "Epoch": epoch + 1
    })

    return grad_summary
