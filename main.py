"""Sequential Learning for Cost-Sensitive Credit Card Fraud Detection:
Supervised Classifiers vs. Contextual Bandit Policies.

Project layout
  Data/      Sparkov dataset (fraudTrain.csv, fraudTest.csv); generated scenarios cached in Data/cache/
  Graphs/    Figures 1-3
  Results/   run logs (Results/runs/), tuned settings, tables (Results/tables/)
  fcmab/     the experiment code (all parameters in fcmab/config.py)
  main.py    this file

Commands
  python main.py tune     [--jobs 4]                    tune bandit settings on 2019 only
  python main.py run      [--scenarios S1] [--seeds 0] [--policies all] [--jobs 4]
  python main.py check                                  consistency checks on finished runs
  python main.py analyze  [--figure-scenario S1 S5]     tables + Figures 1-3
  python main.py all      [--seeds all] [--jobs 4]      everything above, in order

Examples
  python main.py run --scenarios S1 --seeds 0 --jobs 4            # pilot (a few minutes)
  python main.py analyze --figure-scenario S1
  python main.py all --jobs 4                                     # full experiment
"""
import os
import sys


def _opt(args, name, default):
    """Read the values after --name from an argument list."""
    if name not in args:
        return default
    i = args.index(name) + 1
    vals = []
    while i < len(args) and not args[i].startswith("--"):
        vals.append(args[i])
        i += 1
    return vals


def main():
    for stream in (sys.stdout, sys.stderr):        # e.g. Windows consoles using cp1252
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    from fcmab import analysis, checks, config, run, tune

    if cmd == "tune":
        tune.main(rest)
    elif cmd == "run":
        run.main(rest)
    elif cmd == "check":
        raise SystemExit(checks.main())
    elif cmd in ("analyze", "analyse"):
        analysis.main(rest)
    elif cmd == "all":
        jobs = _opt(rest, "--jobs", [str(os.cpu_count() or 2)])
        seeds = _opt(rest, "--seeds", ["all"])
        tuned = os.path.join(config.RESULTS_DIR, "tuned_params.json")
        if not os.path.exists(tuned) or "--retune" in rest:
            tune.main(["--jobs", *jobs])
        run.main(["--scenarios", "all", "--seeds", *seeds, "--jobs", *jobs])
        checks.main()
        analysis.main(["--figure-scenario", "S1", "S5"])
    else:
        print(f"unknown command '{cmd}'\n")
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":  
    main()