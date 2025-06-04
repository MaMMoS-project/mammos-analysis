import mammos_analysis


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(mammos_analysis.__version__, str)
