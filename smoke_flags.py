#!/usr/bin/env python3
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SmokeConfig:
    name: str
    args: List[str]
    use_state_mod: bool
    use_adv: bool
    use_counterfactual: bool
    adv_apply_to: Optional[str]

    def flag_line(self) -> str:
        adv_apply_to = self.adv_apply_to if self.adv_apply_to is not None else "auto"
        return (
            "TRAIN_FLAGS: "
            f"use_state_mod={self.use_state_mod}, "
            f"use_adv={self.use_adv}, "
            f"use_counterfactual={self.use_counterfactual}, "
            f"adv_apply_to={adv_apply_to}"
        )


def run_config(config: SmokeConfig, base_args: List[str]) -> bool:
    cmd = [sys.executable, "-u", "train_min.py", *base_args, *config.args]
    flag_line = config.flag_line()
    print(f"Running smoke config: {config.name}", flush=True)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    combined_log = result.stdout or ""
    if combined_log:
        print(result.stdout)

    has_flag_log = flag_line in combined_log
    passed = result.returncode == 0 and has_flag_log

    if not passed:
        print("SMOKE_FAIL: exit_code=", result.returncode, flush=True)
        if not has_flag_log:
            print("SMOKE_FAIL: missing flag log line", flush=True)

    return passed


def render_summary(results: List[SmokeConfig], statuses: List[bool]) -> str:
    name_width = max(len(cfg.name) for cfg in results)
    header = f"{'组合'.ljust(name_width)} | 结果"
    divider = f"{'-' * name_width}-|-----"
    rows = []
    for cfg, status in zip(results, statuses):
        outcome = "PASS" if status else "FAIL"
        rows.append(f"{cfg.name.ljust(name_width)} | {outcome}")
    return "\n".join([header, divider, *rows])


def main() -> int:
    base_args = ["--train_num", "2", "--val_num", "1"]
    configs = [
        SmokeConfig(
            name="baseline",
            args=[],
            use_state_mod=False,
            use_adv=False,
            use_counterfactual=False,
            adv_apply_to=None,
        ),
        SmokeConfig(
            name="--use_state_mod",
            args=["--use_state_mod"],
            use_state_mod=True,
            use_adv=False,
            use_counterfactual=False,
            adv_apply_to=None,
        ),
        SmokeConfig(
            name="--use_state_mod --use_adv",
            args=["--use_state_mod", "--use_adv"],
            use_state_mod=True,
            use_adv=True,
            use_counterfactual=False,
            adv_apply_to=None,
        ),
        SmokeConfig(
            name="--use_state_mod --use_adv --use_counterfactual --adv_apply_to both",
            args=[
                "--use_state_mod",
                "--use_adv",
                "--use_counterfactual",
                "--adv_apply_to",
                "both",
            ],
            use_state_mod=True,
            use_adv=True,
            use_counterfactual=True,
            adv_apply_to="both",
        ),
    ]

    statuses = [run_config(cfg, base_args) for cfg in configs]
    print("\nSmoke Test Summary")
    print(render_summary(configs, statuses))

    return 0 if all(statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
