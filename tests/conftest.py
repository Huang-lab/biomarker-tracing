def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a method script end to end (deselect with -m 'not slow')")
