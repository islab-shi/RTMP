from __future__ import annotations

import os
import pickle
import time
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# The release pipeline saves plots from scripts, so it must not require a GUI backend.
matplotlib.use("Agg")


def load_model(model: torch.nn.Module, model_save_path: str, device: torch.device | str) -> torch.nn.Module:
    state = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def get_target_point_num(file_path: str, atk_layer: str, target_catego: int = 0) -> int:
    with open(file_path, "rb") as file:
        data = pickle.load(file)[atk_layer]
    return len(data.get(target_catego, []))


def extract_max_attack_intensity(file_path: str, atk_layer: str) -> Dict[int, list]:
    with open(file_path, "rb") as file:
        data = pickle.load(file)[atk_layer]

    result: Dict[int, list] = {}
    for key, values in data.items():
        if not values:
            continue
        result[key] = sorted(values, key=lambda x: x[2], reverse=True)[0]
    return result


def plot_attack_points(
    file_path: Optional[str] = None,
    atk_layer: Optional[str] = None,
    save_path: Optional[str] = None,
    data: Optional[dict] = None,
    only_focus: Optional[Sequence[int]] = None,
    title: str = "",
) -> None:
    data_draw = data or {}
    if atk_layer is not None:
        if file_path is None:
            raise ValueError("file_path is required when atk_layer is provided")
        with open(file_path, "rb") as file:
            data_draw = pickle.load(file)[atk_layer]

    # Keep colors stable across categories so repeated plots are comparable.
    only_focus = list(only_focus or [])
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    plt.figure(figsize=(10, 7))
    added_labels = set()

    for key, values in data_draw.items():
        if only_focus and key not in only_focus:
            continue
        color = colors[key % len(colors)]
        for value in values:
            filter_idx, kernel_idx, attack_intensity = value
            label = f"Class {key}" if key not in added_labels else ""
            plt.scatter(filter_idx, kernel_idx, color=color, label=label, s=max(8.0, attack_intensity * 1.2), alpha=0.75)
            added_labels.add(key)

    x_min, x_max = plt.xlim()
    margin = 0.15 * (x_max - x_min if x_max > x_min else 1.0)
    plt.xlim(x_min, x_max + margin)

    plt.xlabel("Filter Index")
    plt.ylabel("Kernel Index")
    plt.title(title)
    if added_labels:
        plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.2)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=240)
    else:
        plt.show()
    plt.close()


def remove_all_files_in_dir(dir_path: str) -> None:
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def test_acc(
    model: torch.nn.Module,
    device: torch.device | str,
    test_loader,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    verbose: bool = False,
    tag: str = "Eval",
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(test_loader)
    start_ts = time.time()
    log_every = max(1, total_batches // 100)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if verbose and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - start_ts
                rate = batch_idx / max(elapsed, 1e-6)
                eta = (total_batches - batch_idx) / max(rate, 1e-6)
                pct = 100.0 * batch_idx / total_batches
                print(
                    f"\r[{tag}] {batch_idx}/{total_batches} ({pct:6.2f}%) "
                    f"| elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
                    end="",
                    flush=True,
                )

    if verbose:
        print()

    if total == 0:
        return 0.0, 0.0

    return running_loss / total, 100.0 * correct / total


def _top1_accuracy(
    model: torch.nn.Module,
    test_loader,
    device: torch.device | str,
    verbose: bool = False,
    tag: str = "Top1",
) -> float:
    model.eval()
    total = 0
    correct = 0
    total_batches = len(test_loader)
    start_ts = time.time()
    log_every = max(1, total_batches // 100)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

            if verbose and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - start_ts
                rate = batch_idx / max(elapsed, 1e-6)
                eta = (total_batches - batch_idx) / max(rate, 1e-6)
                pct = 100.0 * batch_idx / total_batches
                print(
                    f"\r[{tag}] {batch_idx}/{total_batches} ({pct:6.2f}%) "
                    f"| elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
                    end="",
                    flush=True,
                )

    if verbose:
        print()

    if total == 0:
        return 0.0
    return 100.0 * correct / total


def get_atk_info(
    attacker,
    file_path: str,
    attack_layer_idx_list: Sequence[int],
    member_num: str = "member_1",
    verbose: bool = False,
    show_progress: bool = True,
) -> dict:
    attacker.jsonOpt.readJson(file_path)

    # Rebuild the selected attack exactly as stored in the generated JSON.
    atk_weight = attacker.jsonOpt.jsonData[member_num]["bestAttackedWeight"]
    atk_config_list = attacker.jsonOpt.jsonData[member_num]["kernel_idx"]

    clean_acc = _top1_accuracy(
        attacker.model,
        attacker.testloader,
        attacker.device,
        verbose=show_progress,
        tag="AttackEval Clean",
    )
    attacker.recover()

    attacker.attack_config_list = atk_config_list
    attacker.dnaSize = len(atk_weight)

    layer_names = [name for name, _ in attacker.model.named_modules()]
    attack_layer_list = [layer_names[layer_idx] for layer_idx in attack_layer_idx_list]

    attacker.set_atk_layer(attack_layer_list)
    attacker.kernel_swap(atk_weight, recover=True, advance_attack=True)

    malicious_acc = _top1_accuracy(
        attacker.model,
        attacker.testloader,
        attacker.device,
        verbose=show_progress,
        tag="AttackEval Malicious",
    )

    if verbose:
        print(f"[AttackKernel] {atk_config_list}")

    return {
        "clean_acc": clean_acc,
        "malicious_acc": malicious_acc,
        "attack_kernel": atk_config_list,
        "dna_size": len(atk_weight),
    }


def fineturning_last_fc(
    model: torch.nn.Module,
    train_loader,
    device: torch.device | str,
    num_epochs: int = 5,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    fc_name: Optional[Iterable[str]] = None,
    test_loader=None,
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    verbose: bool = True,
    round_tag: str = "",
) -> torch.nn.Module:
    trainable_set = set(fc_name or [])

    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_set

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )

    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_ts = time.time()
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 100)

        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if verbose and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - epoch_start_ts
                rate = batch_idx / max(elapsed, 1e-6)
                eta = (total_batches - batch_idx) / max(rate, 1e-6)
                pct = 100.0 * batch_idx / total_batches
                running_train_loss = running_loss / max(1, total)
                running_train_acc = 100.0 * correct / max(1, total)
                prefix = f"[FineTune {round_tag}]" if round_tag else "[FineTune]"
                print(
                    f"\r{prefix} epoch={epoch:02d}/{num_epochs} batch={batch_idx}/{total_batches} "
                    f"({pct:6.2f}%) | loss={running_train_loss:.4f} | acc={running_train_acc:6.2f}% "
                    f"| elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
                    end="",
                    flush=True,
                )

        if verbose:
            print()

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = 100.0 * correct / max(1, total)

        val_loss, val_acc = (0.0, 0.0)
        if test_loader is not None:
            val_loss, val_acc = test_acc(
                model,
                device,
                test_loader,
                verbose=verbose,
                tag=f"FineTune {round_tag} Val e{epoch:02d}".strip(),
            )

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if verbose:
            prefix = f"[FineTune {round_tag}]" if round_tag else "[FineTune]"
            print(
                f"{prefix} epoch={epoch:02d}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:6.2f}% | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:6.2f}% | lr={lr:.2e}"
            )

    model.eval()
    return model


def plot_list(data: Sequence[float], file_path: str) -> None:
    if not data:
        return

    plt.figure(figsize=(8, 5))
    x_values = list(range(len(data)))

    plt.plot(x_values, data, marker="o", linewidth=1.8)
    plt.xticks(x_values)
    plt.title("Accuracy Curve")
    plt.xlabel("Round")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(alpha=0.25)

    for x, y in zip(x_values, data):
        plt.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
