"""Checks for the best-model rule of sdc_eval/evaluator.py: by default the best model is the lowest co2_work_adjusted
whatever it drops; drop_tolerance restores the old guard; every row of evals.csv has the header's columns.
Run from the repo root: python tests/verify_best_model.py"""
import csv, os, sys, tempfile
sys.path.insert(0, os.getcwd())
import yaml

from sdc_eval.evaluator import FixedSeedEvaluator
from sdc_eval.policies import NoOpPolicy

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


class AlwaysDefer(NoOpPolicy):
    """Defers every shiftable task: the queue fills up within the day, so it drops tasks (they still run)."""
    def act(self, sdc):
        return {'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2}


def rows(run_dir):
    with open(os.path.join(run_dir, 'eval', 'evals.csv')) as f:
        lines = list(csv.reader(f))
    return lines[0], lines[1:]


def main():
    with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
        scenario = {**yaml.safe_load(f), 'location': 'tx', 'days_per_episode': 1, 'month': 6}
    for tol, expect_best in ((None, True), (1.0, False)):
        run_dir = tempfile.mkdtemp(prefix='verify_best_')
        ev = FixedSeedEvaluator(scenario, run_dir, n_episodes=1, drop_tolerance=tol, save_decisions=False)
        saved = []
        out1 = ev.evaluate(AlwaysDefer(), 100, save_fn=saved.append)
        after_first = list(saved)
        out2 = ev.evaluate(NoOpPolicy(), 200, save_fn=saved.append)
        check(f"drop_tolerance={tol}: the dropping policy drops tasks", out1['tasks_dropped'] > 1,
              f"{out1['tasks_dropped']:.0f} dropped")
        check(f"drop_tolerance={tol}: dropping policy {'selected' if expect_best else 'not selectable'}",
              (after_first == ['best']) == expect_best and bool(out1['eligible']) == expect_best,
              f"saved after it: {after_first}")
        header, data = rows(run_dir)
        check(f"drop_tolerance={tol}: every evals.csv row has the header's columns",
              all(len(r) == len(header) for r in data) and 'best_co2_work_adjusted' in header,
              f"{len(header)} columns, rows {[len(r) for r in data]}")
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
