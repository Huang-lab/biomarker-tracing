def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a method script end to end (deselect with -m 'not slow')")


def pytest_terminal_summary(terminalreporter, config):
    """After the method tests, print how many cell types each method called significant."""
    rows = getattr(config, "significance_rows", [])
    if not rows:
        return
    terminalreporter.section("significant cell types per method (tutorial disease)")
    for method, n_selected, n_tested, selected in rows:
        terminalreporter.write_line(f"{method:<20} {n_selected:>4} of {n_tested} cell types")
    for method, _, _, selected in rows:
        terminalreporter.write_line(f"\n{method}: " + ", ".join(selected[:8]) + (" ..." if len(selected) > 8 else ""))
