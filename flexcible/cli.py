"""Command line of FlexCible: ``flexcible <command> ...``.

Each command is the ``main()`` of the module of the same name (predict,
precompute, plot, loops, evaluate, diagnose, train); ``flexcible predict
--help`` shows that module's options.
"""
import importlib
import sys

from flexcible import __version__

COMMANDS = ("predict", "precompute", "plot", "loops", "evaluate", "diagnose", "train", "train_multires")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: flexcible <command> [options]\n       flexcible --version\ncommands: " + ", ".join(COMMANDS))
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"flexcible {__version__}")
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"flexcible: unknown command {cmd!r}; commands: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    mod = importlib.import_module(f"flexcible.{cmd}")
    sys.argv = [f"flexcible {cmd}"] + argv[1:]
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
