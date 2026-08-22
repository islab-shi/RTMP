from __future__ import annotations

import contextlib
import io
import os
import pickle
import time
from typing import List, Sequence

import torch


def _resolve_attack_layers(model: torch.nn.Module, attack_layer_idx_list: Sequence[int]) -> List[str]:
    # Attack configuration files use module indices; runtime code needs module names.
    layer_names = [name for name, _ in model.named_modules()]
    return [layer_names[idx] for idx in attack_layer_idx_list]


def _layer_weight(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
    params = dict(model.named_parameters())
    key = f"{layer_name}.weight"
    if key not in params:
        raise KeyError(f"Parameter not found: {key}")
    return params[key]


def _build_attack_config_for_kernel(
    model: torch.nn.Module,
    attack_layer_list: Sequence[str],
    kernel_size_list: Sequence[int],
    start_filter_idx: int,
    start_kernel_idx: int,
    attack_elem_num: int,
) -> list:
    filter_idx = start_filter_idx
    kernel_idx = start_kernel_idx
    attack_config_list_tmp = []

    for idx, layer_name in enumerate(attack_layer_list):
        weight = _layer_weight(model, layer_name)
        attack_config = []

        # Follow the selected kernel through subsequent layers using the weakest weights.
        if idx == 0:
            kernel = weight[filter_idx][kernel_idx]
            kernel_size = kernel.shape[0]
            sorted_indices = torch.sort(torch.flatten(kernel)).indices.tolist()
            if kernel_size == 1:
                attack_config.append([1, 1, filter_idx, kernel_idx, layer_name, 1, int(sorted_indices[0])])
            else:
                for pos_idx in sorted_indices[:attack_elem_num]:
                    attack_config.append([1, 1, filter_idx, kernel_idx, layer_name, 1, int(pos_idx)])
            filter_idx, kernel_idx = start_kernel_idx, start_filter_idx

        elif idx != len(attack_layer_list) - 1:
            kernel_size = weight[0][0].shape[0]
            if kernel_size == 1:
                filter_flatten = torch.flatten(weight[filter_idx])
                sorted_indices = torch.sort(filter_flatten).indices
                lsb_idx = int(sorted_indices[0].item())
                kernel_idx_sc = lsb_idx // (kernel_size_list[idx] ** 2)
                attack_config.append([1, 1, filter_idx, kernel_idx_sc, layer_name, 1, 0])
            else:
                filter_flatten = torch.flatten(weight[filter_idx])
                sorted_indices = torch.sort(filter_flatten).indices
                lsb_idx = int(sorted_indices[0].item())
                kernel_idx = lsb_idx // (kernel_size_list[idx] ** 2)

                kernel = weight[filter_idx][kernel_idx]
                sorted_indices = torch.sort(torch.flatten(kernel)).indices.tolist()
                for pos_idx in sorted_indices[:attack_elem_num]:
                    attack_config.append([1, 1, filter_idx, kernel_idx, layer_name, 1, int(pos_idx)])
            filter_idx = kernel_idx

        if idx == len(attack_layer_list) - 1:
            attack_config = []
            filter_flatten = torch.flatten(weight[filter_idx])
            sorted_indices = torch.sort(filter_flatten).indices
            lsb_idx = int(sorted_indices[0].item())

            kernel_size = weight[filter_idx][0].shape[0]
            if kernel_size == 1:
                kernel_idx_sc = lsb_idx // (kernel_size_list[idx] ** 2)
                attack_config.append([1, 1, filter_idx, kernel_idx_sc, layer_name, 1, 0])
            else:
                kernel_idx = lsb_idx // (kernel_size_list[idx] ** 2)
                kernel = weight[filter_idx][kernel_idx]
                sorted_indices = torch.sort(torch.flatten(kernel)).indices.tolist()
                for pos_idx in sorted_indices[:attack_elem_num]:
                    attack_config.append([1, 1, filter_idx, kernel_idx, layer_name, 1, int(pos_idx)])

        attack_config_list_tmp.extend(attack_config)

    return attack_config_list_tmp


def get_attack_config_list(
    model=None,
    attack_layer_idx_list: Sequence[int] = (),
    attack_filter_num: int = 1,
    attack_kernel_num: int = 1,
    start_filter_idx: int = 0,
    start_kernel_idx: int = 0,
    attack_elem_num: int = 9,
):
    attack_layer_list = _resolve_attack_layers(model, attack_layer_idx_list)
    kernel_size_list = [_layer_weight(model, layer).shape[2] for layer in attack_layer_list]

    first_weight = _layer_weight(model, attack_layer_list[0])
    filter_num = first_weight.shape[0]
    kernel_num = first_weight.shape[1]

    for i in range(start_filter_idx, filter_num, attack_filter_num):
        for k in range(start_kernel_idx, kernel_num, attack_kernel_num):
            return _build_attack_config_for_kernel(
                model=model,
                attack_layer_list=attack_layer_list,
                kernel_size_list=kernel_size_list,
                start_filter_idx=i,
                start_kernel_idx=k,
                attack_elem_num=attack_elem_num,
            )

    return []


def _run_attack(attacker, attack_catego: int, quiet: bool) -> None:
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            attacker.attack("advanceAttack_2", round=1, attack_catego=attack_catego)
    else:
        attacker.attack("advanceAttack_2", round=1, attack_catego=attack_catego)


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _format_score(score: float, score_scale: int) -> str:
    if score_scale <= 0:
        return f"{score:.2f}"
    ratio = score / score_scale * 100.0
    return f"{score:.2f}/{score_scale} ({ratio:5.1f}%)"


def multi_layer_exploration(
    attacker=None,
    attack_layer_idx_list: Sequence[int] = (),
    attack_filter_num: int = 1,
    attack_kernel_num: int = 1,
    sensitive_threshold: float = 1,
    saved_file_prefix: str = "",
    through_kernel: bool = False,
    start_filter_idx: int = 0,
    start_kernel_idx: int = 0,
    attack_catego: int = -1,
    attack_elem_num: int = 9,
    use_hamming_dist: bool = False,
    atk_weight: Sequence[float] = (),
    only_get_config: bool = False,
    atk_single_kernel: bool = False,
    saved_round: int = 31,
    saved_dir: str = "./atk_result",
    use_caffe: bool = False,
    only_find_one: bool = False,
    quiet: bool = False,
    show_progress: bool = True,
    progress_label: str = "Explore",
):
    del atk_weight, use_caffe

    attack_layer_list = _resolve_attack_layers(attacker.model, attack_layer_idx_list)
    kernel_size_list = [_layer_weight(attacker.model, layer).shape[2] for layer in attack_layer_list]

    pop_bound_list = []
    for layer in attack_layer_list:
        layer_weight = _layer_weight(attacker.model, layer)
        layer_max_val = torch.max(layer_weight).cpu().item()
        layer_min_val = torch.min(layer_weight).cpu().item()
        pop_bound_list.append([layer_min_val, layer_max_val])

    # Evolution bounds mirror the current checkpoint weights for each attacked layer.
    attacker.popBound = pop_bound_list
    attacker.attack_maxVal = [bound[1] for bound in pop_bound_list]

    first_layer = attack_layer_list[0]
    layer_param = _layer_weight(attacker.model, first_layer)
    filter_num = layer_param.shape[0]
    kernel_num = layer_param.shape[1]

    sensitive_kernels_collection = {first_layer: {}}
    max_val = 0.0
    os.makedirs(saved_dir, exist_ok=True)
    filter_steps = len(range(start_filter_idx, filter_num, attack_filter_num))
    kernel_steps = len(range(start_kernel_idx, kernel_num, attack_kernel_num))
    if atk_single_kernel:
        total_iterations = 1
    elif through_kernel:
        total_iterations = filter_steps * kernel_steps
    else:
        total_iterations = filter_steps
    processed_iterations = 0
    progress_start_ts = time.time()
    progress_log_every = max(1, total_iterations // 200)
    score_scale = int(attacker.x_test.shape[0]) if hasattr(attacker.x_test, "shape") else 0

    if show_progress:
        print(
            f"[{progress_label}] metric: sensitivity_score = class confidence-sum increase "
            f"on defence batch (higher means more sensitive). threshold={sensitive_threshold:.2f}"
        )

    for i in range(start_filter_idx, filter_num, attack_filter_num):
        exit_current_filter = False
        current_filter_max_val = 0.0

        for k in range(start_kernel_idx, kernel_num, attack_kernel_num):
            attack_config_list_tmp = _build_attack_config_for_kernel(
                model=attacker.model,
                attack_layer_list=attack_layer_list,
                kernel_size_list=kernel_size_list,
                start_filter_idx=i,
                start_kernel_idx=k,
                attack_elem_num=attack_elem_num,
            )

            if only_get_config:
                return attack_config_list_tmp

            attack_weight_num = 0
            for atk_info in attack_config_list_tmp:
                attack_weight_num += atk_info[5] * atk_info[0] * atk_info[1]

            attacker.attack_config_list = attack_config_list_tmp
            attacker.attack_layer_dna_size = attack_config_list_tmp
            attacker.dnaSize = attack_weight_num
            attacker.set_atk_layer(attack_layer_list)

            if use_hamming_dist:
                attacker.set_attack_layer_idx_list(attack_layer_idx_list)
                attacker.use_hamming_dist = True

            _run_attack(attacker, attack_catego=attack_catego, quiet=quiet)
            attacker.recover()

            for j, catego_index in enumerate(attacker.sensitive_info_index):
                score = attacker.sensitive_info_val[j]
                current_filter_max_val = max(current_filter_max_val, score)
                threshold = sensitive_threshold

                if score > threshold:
                    sensitive_kernels_collection[first_layer].setdefault(catego_index, [])
                    sensitive_kernels_collection[first_layer][catego_index].append([i, k, score])
                    sensitive_kernels_collection[first_layer][catego_index].sort(key=lambda x: x[2], reverse=True)
                    max_val = max(max_val, score)

                    points = sensitive_kernels_collection[first_layer][catego_index]
                    if (
                        len(points) > 4
                        and points[0][0] == i
                        and points[1][0] == i
                        and points[2][0] == i
                        and points[3][0] == i
                        and points[4][0] == i
                    ):
                        exit_current_filter = True

            attacker.sensitive_info_recover()
            processed_iterations += 1

            if show_progress and (
                processed_iterations == 1
                or processed_iterations % progress_log_every == 0
                or processed_iterations == total_iterations
            ):
                elapsed = time.time() - progress_start_ts
                rate = processed_iterations / max(elapsed, 1e-6)
                eta = (total_iterations - processed_iterations) / max(rate, 1e-6)
                pct = 100.0 * processed_iterations / max(1, total_iterations)
                cur_score = _format_score(current_filter_max_val, score_scale)
                best_score = _format_score(max_val, score_scale)
                print(
                    f"\r[{progress_label}] {processed_iterations}/{total_iterations} ({pct:6.2f}%) "
                    f"| f={i}, k={k} | current_peak_sensitivity={cur_score} "
                    f"| best_peak_sensitivity={best_score} | elapsed={_format_seconds(elapsed)} "
                    f"| eta={_format_seconds(eta)}",
                    end="",
                    flush=True,
                )

            if not quiet:
                print(
                    f"[Explore] filter={i}, kernel={k}, current_max={current_filter_max_val:.2f}, global_max={max_val:.2f}"
                )

            if only_find_one and sensitive_kernels_collection[first_layer].get(attack_catego):
                if show_progress:
                    print()
                return

            if not through_kernel:
                break

            if atk_single_kernel:
                if not quiet:
                    print(f"[SingleKernelConfig] {attack_config_list_tmp}")
                if show_progress:
                    print()
                return

            if exit_current_filter:
                break

        if i % saved_round == 0 and i != 0:
            saved_path = os.path.join(saved_dir, f"_{i}_{saved_file_prefix} {first_layer}.pkl")
            with open(saved_path, "wb") as tf:
                pickle.dump(sensitive_kernels_collection, tf)
            if not quiet:
                print(f"[ExploreSaved] {saved_path}")

    if show_progress:
        print()

    final_path = f"{saved_file_prefix} {first_layer}.pkl"
    with open(final_path, "wb") as tf:
        pickle.dump(sensitive_kernels_collection, tf)
