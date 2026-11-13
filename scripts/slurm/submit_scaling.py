"""Submit a geo_scaling.py grid to Slurm: one array job per scenario (network size), one task per
(month, utilization, sigma24, seed), with time and memory set from the network size and episode length.

Run from the repo root, e.g. (go/no-go check 2: |B| / N against N, screening only):
    python scripts/slurm/submit_scaling.py --name bn_v1 --scenarios n20,n50,n200 --sigmas 0.1,0.2 --seeds 1,2 \
        --days 2 --methods origin,screen
Full comparison (check 3):
    python scripts/slurm/submit_scaling.py --name cmp_v1 --scenarios dccluster5,n20 --months 0,6 --utils 0.5,0.7 \
        --sigmas 0,0.1,0.2 --seeds 1,2,3,4,5 --days 7 --methods origin,merit,index,screen,index_full,mpc --bound
Extra geo_scaling.py arguments go after --, e.g.  -- --time_limit 900
Results: <root>/<name>/case_*.json (a task resumes the methods already in its file); then
    python geo_scaling.py --combine <root>/<name>
--dry_run prints the sbatch commands and resources without submitting.
"""
import argparse
import itertools
import os
import re
import subprocess
import sys

ROOT = '/network/scratch/f/florence.cloutier/sustaindc-rl_results/geo_scaling'
# hours of wall time per simulated day for the LP methods, by network size (single CPU, measured order of magnitude:
# ~30 ms per site LP incl. building it; screening solves ~39 site LPs per site per hour + one per site per step)
HOURS_PER_DAY = [(5, 0.25), (20, 0.6), (50, 1.5), (100, 3.0), (200, 6.0)]
MEM_GB = [(20, 8), (50, 12), (100, 16), (200, 24)]


def n_sites(scenario):
    from geo_scenarios import FIXED_SETS
    if scenario in FIXED_SETS:
        return len(FIXED_SETS[scenario])
    if scenario == 'dcc15':
        return 15
    m = re.fullmatch(r'n(\d+).*', scenario)
    if not m:
        raise ValueError(f'Unknown scenario {scenario!r}')
    return int(m.group(1))


def lookup(table, n):
    for limit, v in table:
        if n <= limit:
            return v
    return table[-1][1] * n / table[-1][0]


def main():
    sys.path.insert(0, os.getcwd())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--scenarios', required=True)
    p.add_argument('--months', default='6')
    p.add_argument('--utils', default='0.5')
    p.add_argument('--sigmas', default='0.1')
    p.add_argument('--seeds', default='1')
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--methods', default='origin,merit,index,screen')
    p.add_argument('--bound', action='store_true')
    p.add_argument('--root', default=ROOT)
    p.add_argument('--time_scale', type=float, default=1.0, help='multiply the estimated wall time')
    p.add_argument('--max_hours', type=float, default=72.0)
    p.add_argument('--dry_run', action='store_true')
    args, extra = p.parse_known_args()
    extra = [e for e in extra if e != '--']
    out_dir = os.path.join(args.root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.root), 'slurm'), exist_ok=True)
    n_methods = len(args.methods.split(','))
    for scen in args.scenarios.split(','):
        N = n_sites(scen)
        grid = list(itertools.product(args.months.split(','), args.utils.split(','), args.sigmas.split(','),
                                      args.seeds.split(',')))
        lines = [f'--scenario {scen} --month {m} --util {u} --sigma24 {s} --seed {sd} --days {args.days} '
                 f'--methods {args.methods} --out_dir {out_dir}' + (' --bound' if args.bound else '')
                 + (' ' + ' '.join(extra) if extra else '') for m, u, s, sd in grid]
        taskfile = os.path.join(out_dir, f'tasks_{scen}.txt')
        with open(taskfile, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        hours = lookup(HOURS_PER_DAY, N) * args.days * max(n_methods - 1, 1) * 0.6 * args.time_scale + 0.5
        hours = min(max(hours, 1.0), args.max_hours)
        mem = lookup(MEM_GB, N) + (8 if args.bound else 0)
        cmd = ['sbatch', f'--job-name=scal-{args.name}-{scen}', f'--array=0-{len(lines) - 1}',
               f'--time={int(hours)}:{int(60 * (hours % 1)):02d}:00', f'--mem={int(mem)}G',
               'scripts/slurm/geo_scaling_array.sbatch', taskfile]
        print(f'{scen}: {N} sites, {len(lines)} tasks, {hours:.1f} h, {mem:.0f} GB\n  ' + ' '.join(cmd))
        if not args.dry_run:
            print('  ' + subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
    print(f'Combine when done: python geo_scaling.py --combine {out_dir}')


if __name__ == '__main__':
    main()
