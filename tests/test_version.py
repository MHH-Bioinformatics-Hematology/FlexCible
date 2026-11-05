"""The version number is kept in one place and every reader agrees."""
from importlib.metadata import version

import flexcible
from flexcible import _version, cli


def test_version_single_source(capsys):
    assert flexcible.__version__ == _version.__version__
    assert version("flexcible") == flexcible.__version__
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"flexcible {flexcible.__version__}"


def test_help_mentions_version(capsys):
    assert cli.main(["--help"]) == 0
    assert "--version" in capsys.readouterr().out
