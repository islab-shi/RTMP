from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torchvision.models import ResNet18_Weights, resnet18

from combination_attack import combAttacker
from exploration import get_attack_config_list, multi_layer_exploration
from imagenet_data import get_imagenet_data
from paper_utils import (
    extract_max_attack_intensity,
    fineturning_last_fc,
    get_atk_info,
    get_target_point_num,
    load_model,
    plot_attack_points,
    plot_list,
    remove_all_files_in_dir,
    test_acc,
)


@dataclass
class BaseConfig:
    defence_batch_size: int = 40
    train_batch_size: int = 128
    attack_filter_num: int = 1
    attack_kernel_num: int = 1
    attack_elem_num: int = 9
    sensitive_threshold: int = 20
    start_filter_idx: int = 0
    start_kernel_idx: int = 0
    defence_catego: int = 811
    use_hamming_dist: bool = False
    release_num: int = 4
    max_rounds: int = 10
    saved_round: int = 99


@dataclass
class FineTuneConfig:
    num_epochs: int = 6
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler_step_size: int = 2
    scheduler_gamma: float = 0.1


def log_header(title: str) -> None:
    line = "=" * 96
    print(f"\n{line}\n{title}\n{line}")


def log_kv(title: str, data: dict) -> None:
    print(f"[{title}]")
    for key, value in data.items():
        print(f"  - {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-defence pipeline for ImageNet ResNet18.")
    parser.add_argument("--dataset-dir", required=True, help="ImageNet directory that contains train/ and val/.")
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional local checkpoint path. If empty, use torchvision ResNet18 IMAGENET1K_V1 weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="./auto_defence_imagenet",
        help="Directory to store intermediate and final outputs.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-point", type=int, default=1, help="Start index in fc_name_list.")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--verbose-attack", action="store_true", help="Show raw attacker logs.")
    return parser.parse_args()


def setup_output_dirs(output_dir: Path) -> dict:
    paths = {
        "root": output_dir,
        "input_model": output_dir / "input_model",
        "atk_result_pkl": output_dir / "atk_result_pkl",
        "point": output_dir / "point",
        "atk_result_json": output_dir / "atk_result_json",
        "malicious_model": output_dir / "malicious_model",
        "fineturning_model": output_dir / "fineturning_model",
        "acc_drop": output_dir / "acc_drop",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def get_imagenet_labels() -> List[str]:
    # Use the same torchvision weight metadata as the initial ResNet18 checkpoint.
    labels = list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
    if len(labels) != 1000:
        raise ValueError(f"Expected 1000 ImageNet labels, got {len(labels)}.")
    return labels


def get_params_between_layers(model: torch.nn.Module, start_layer: str, end_layer: str) -> List[str]:
    # Fine-tuning updates the contiguous parameter span selected by fc_name_list.
    all_param_names = [name for name, _ in model.named_parameters()]

    start_idx = next((i for i, p in enumerate(all_param_names) if p.startswith(f"{start_layer}.")), None)
    end_idx = next((i for i, p in enumerate(all_param_names) if p == end_layer), None)

    if start_idx is None or end_idx is None:
        raise ValueError(f"Cannot find layers in model parameters: start={start_layer}, end={end_layer}")

    if start_idx <= end_idx:
        return all_param_names[start_idx : end_idx + 1]
    return all_param_names[end_idx : start_idx + 1]


def prepare_initial_checkpoint(model_path: str, model_dir: Path, device: torch.device) -> str:
    if model_path and Path(model_path).exists():
        return model_path

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    checkpoint_path = model_dir / "resnet18_imagenet1k_v1.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return str(checkpoint_path)


def resolve_saved_pkl_path(atk_result_dir: Path, saved_round: int, prefix: str, atk_layer: str) -> str:
    preferred = atk_result_dir / f"_{saved_round}_{prefix} {atk_layer}.pkl"
    if preferred.exists():
        return str(preferred)

    pattern = str(atk_result_dir / f"_*_{prefix} {atk_layer}.pkl")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No exploration PKL found. Pattern: {pattern}")
    return candidates[-1]


def first_defence_batch(test_loader, device: torch.device, defence_batch_size: int):
    # The attack search uses a fixed validation mini-batch as its optimization target.
    for x_data, y_data in test_loader:
        return x_data[:defence_batch_size].to(device), y_data[:defence_batch_size].to(device)
    raise RuntimeError("Empty test_loader.")


def main() -> None:
    args = parse_args()
    base_cfg = BaseConfig(max_rounds=args.max_rounds)
    finetune_cfg = FineTuneConfig()

    device = torch.device(args.device)
    paths = setup_output_dirs(Path(args.output_dir))
    labels = get_imagenet_labels()

    log_header("Auto Defence - ImageNet")
    log_kv(
        "Runtime Config",
        {
            "device": device,
            "dataset_dir": args.dataset_dir,
            "label_source": "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
            "output_dir": str(paths["root"]),
            "defence_batch_size": base_cfg.defence_batch_size,
            "train_batch_size": base_cfg.train_batch_size,
            "max_rounds": base_cfg.max_rounds,
            "defence_catego": base_cfg.defence_catego,
        },
    )

    train_dataset, train_loader, test_dataset, test_loader = get_imagenet_data(
        args.dataset_dir,
        batch_size=base_cfg.train_batch_size,
        num_workers=args.num_workers,
    )
    log_kv(
        "Dataset",
        {
            "train_size": len(train_dataset),
            "val_size": len(test_dataset),
            "num_workers": args.num_workers,
        },
    )

    x_data, y_data = first_defence_batch(test_loader, device, base_cfg.defence_batch_size)

    attack_layer_idx_list = [20, 16, 13]
    attack_final_layer_name = "layer2.0.conv1"
    fc_name_list = [
        ["fc", "fc.bias"],
        ["layer4.1.conv1", "fc.bias"],
        ["layer4.0.conv1", "fc.bias"],
        ["layer3.1.conv1", "fc.bias"],
    ]

    initial_checkpoint = prepare_initial_checkpoint(args.model_path, paths["input_model"], device)

    for fc_name_idx in range(args.start_point, len(fc_name_list)):
        start_layer_name, end_layer_name = fc_name_list[fc_name_idx]
        last_model_path = initial_checkpoint
        acc_list: List[float] = []

        log_header(f"Defence Target #{fc_name_idx}: {start_layer_name} -> {end_layer_name}")

        for round_idx in range(base_cfg.max_rounds):
            model = resnet18(weights=None)
            load_model(model, last_model_path, device)

            fc_name = get_params_between_layers(model, start_layer_name, end_layer_name)
            print(f"[Round {round_idx:02d}] Stage 0/5: evaluating baseline accuracy on full val set...")
            baseline_start_ts = time.time()
            _, base_accuracy = test_acc(
                model,
                device,
                test_loader,
                verbose=True,
                tag=f"Round {round_idx:02d} BaselineEval",
            )
            acc_list.append(base_accuracy)
            baseline_elapsed = time.time() - baseline_start_ts

            print(
                f"[Round {round_idx:02d}] baseline_acc={base_accuracy:.2f}% "
                f"| checkpoint={os.path.basename(last_model_path)} | elapsed={baseline_elapsed/60:.1f} min"
            )

            attacker = combAttacker(
                model=model,
                networkName="ResNet18",
                dnaSize=0,
                popBound=[],
                numGenerations=1,
                popSize=6,
                filterPosition=0,
                layerName=[],
                expectedAcc=0.3,
                x_test=x_data,
                y_test=y_data,
                batch_size=base_cfg.defence_batch_size,
                className=labels,
                device=device,
                attack_config_list=[],
                onnxModel=None,
                testloader=test_loader,
                json_file_path=str(paths["atk_result_json"]),
            )

            saved_file_prefix = f"fcIdx_{fc_name_idx}_round_{round_idx}"
            print(f"[Round {round_idx:02d}] Stage 1/5: exploring sensitive kernels (this is usually the longest step)...")
            explore_start_ts = time.time()
            multi_layer_exploration(
                attacker=attacker,
                attack_layer_idx_list=attack_layer_idx_list,
                attack_filter_num=base_cfg.attack_filter_num,
                attack_kernel_num=base_cfg.attack_kernel_num,
                sensitive_threshold=base_cfg.sensitive_threshold,
                saved_file_prefix=saved_file_prefix,
                through_kernel=True,
                start_filter_idx=base_cfg.start_filter_idx,
                start_kernel_idx=base_cfg.start_kernel_idx,
                attack_catego=-1,
                attack_elem_num=base_cfg.attack_elem_num,
                use_hamming_dist=base_cfg.use_hamming_dist,
                atk_single_kernel=False,
                saved_round=base_cfg.saved_round,
                saved_dir=str(paths["atk_result_pkl"]),
                quiet=not args.verbose_attack,
                show_progress=True,
                progress_label=f"Round {round_idx:02d} Explore",
            )
            attacker.recover()
            explore_elapsed = time.time() - explore_start_ts
            print(f"[Round {round_idx:02d}] Stage 1/5 done | elapsed={explore_elapsed/60:.1f} min")

            pkl_file_path = resolve_saved_pkl_path(
                paths["atk_result_pkl"],
                base_cfg.saved_round,
                saved_file_prefix,
                attack_final_layer_name,
            )
            point_save_path = paths["point"] / f"fcIdx_{fc_name_idx}_round_{round_idx}.png"

            print(f"[Round {round_idx:02d}] Stage 2/5: summarizing sensitive points and plotting distribution...")
            target_catgo_num = get_target_point_num(pkl_file_path, attack_final_layer_name, base_cfg.defence_catego)
            print(
                f"[Round {round_idx:02d}] sensitive_points={target_catgo_num} "
                f"| pkl={os.path.basename(pkl_file_path)}"
            )

            plot_attack_points(
                file_path=pkl_file_path,
                atk_layer=attack_final_layer_name,
                save_path=str(point_save_path),
                only_focus=[base_cfg.defence_catego],
                title=f"target={base_cfg.defence_catego}, points={target_catgo_num}",
            )

            max_sensitive_kernel_info = extract_max_attack_intensity(pkl_file_path, attack_final_layer_name)
            target_catgo = max_sensitive_kernel_info.get(base_cfg.defence_catego)
            print(f"[Round {round_idx:02d}] Stage 3/5: selecting target kernel to neutralize...")

            if target_catgo is None or target_catgo_num <= base_cfg.release_num:
                print(
                    f"[Round {round_idx:02d}] stop: target category is no longer sensitive "
                    f"(points={target_catgo_num}, threshold={base_cfg.release_num})"
                )
                break

            target_f_idx, target_k_idx = target_catgo[0], target_catgo[1]
            print(
                f"[Round {round_idx:02d}] target_kernel=(filter={target_f_idx}, kernel={target_k_idx}, "
                f"score={target_catgo[2]:.2f})"
            )

            config_list = get_attack_config_list(
                model=model,
                attack_layer_idx_list=attack_layer_idx_list,
                attack_filter_num=1,
                attack_kernel_num=1,
                start_filter_idx=target_f_idx,
                start_kernel_idx=target_k_idx,
                attack_elem_num=base_cfg.attack_elem_num,
            )

            remove_all_files_in_dir(str(paths["atk_result_json"]))
            attacker.numGenerations = 50
            print(f"[Round {round_idx:02d}] Stage 4/5: generating attack JSON for the selected kernel...")
            attack_json_start_ts = time.time()
            multi_layer_exploration(
                attacker=attacker,
                attack_layer_idx_list=attack_layer_idx_list,
                attack_filter_num=1,
                attack_kernel_num=1,
                sensitive_threshold=base_cfg.sensitive_threshold,
                through_kernel=True,
                start_filter_idx=config_list[0][2],
                start_kernel_idx=config_list[0][3],
                attack_catego=base_cfg.defence_catego,
                attack_elem_num=base_cfg.attack_elem_num,
                use_hamming_dist=False,
                atk_single_kernel=True,
                quiet=not args.verbose_attack,
                show_progress=True,
                progress_label=f"Round {round_idx:02d} AttackJSON",
            )
            attack_json_elapsed = time.time() - attack_json_start_ts
            print(f"[Round {round_idx:02d}] Stage 4/5 done | elapsed={attack_json_elapsed:.1f} s")

            json_file_list = sorted(
                f for f in os.listdir(paths["atk_result_json"]) if (paths["atk_result_json"] / f).is_file()
            )
            if not json_file_list:
                raise RuntimeError("No JSON attack result generated.")
            json_file_path = paths["atk_result_json"] / json_file_list[0]

            metric = get_atk_info(
                attacker,
                str(json_file_path),
                attack_layer_idx_list,
                verbose=args.verbose_attack,
                show_progress=True,
            )
            print(
                f"[Round {round_idx:02d}] clean_acc={metric['clean_acc']:.2f}% "
                f"-> malicious_acc={metric['malicious_acc']:.2f}%"
            )

            malicious_model_path = paths["malicious_model"] / f"fcIdx_{fc_name_idx}_malicious_loop_{round_idx}.pth"
            torch.save(attacker.model.state_dict(), malicious_model_path)

            finetune_model = resnet18(weights=None)
            load_model(finetune_model, str(malicious_model_path), device)

            print(f"[Round {round_idx:02d}] fine-tuning {len(fc_name)} params ({start_layer_name} -> {end_layer_name})")
            finetune_start_ts = time.time()
            fineturning_last_fc(
                finetune_model,
                train_loader,
                device,
                num_epochs=finetune_cfg.num_epochs,
                learning_rate=finetune_cfg.learning_rate,
                momentum=finetune_cfg.momentum,
                weight_decay=finetune_cfg.weight_decay,
                fc_name=fc_name,
                test_loader=test_loader,
                scheduler_step_size=finetune_cfg.scheduler_step_size,
                scheduler_gamma=finetune_cfg.scheduler_gamma,
                verbose=True,
                round_tag=f"fc{fc_name_idx}-r{round_idx}",
            )
            finetune_elapsed = time.time() - finetune_start_ts
            print(f"[Round {round_idx:02d}] Stage 5/5 done | fine-tuning elapsed={finetune_elapsed/60:.1f} min")

            finetuned_model_path = paths["fineturning_model"] / f"fcIdx_{fc_name_idx}_finetuned_loop_{round_idx}.pth"
            torch.save(finetune_model.state_dict(), finetuned_model_path)
            print(f"[Round {round_idx:02d}] saved_finetuned={finetuned_model_path}")

            last_model_path = str(finetuned_model_path)

        if acc_list:
            acc_drop_path = paths["acc_drop"] / f"fcIdx_{fc_name_idx}_acc_drop.png"
            plot_list(acc_list, str(acc_drop_path))
            print(f"[Summary fcIdx={fc_name_idx}] accuracy_curve={acc_drop_path}")
            print(f"[Summary fcIdx={fc_name_idx}] acc_list={', '.join(f'{x:.2f}' for x in acc_list)}")


if __name__ == "__main__":
    main()
