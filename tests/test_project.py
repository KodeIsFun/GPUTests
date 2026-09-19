from lab.project import WeekProject, scaffold


def test_scaffold_creates_kernel_and_manifest(tmp_path) -> None:
    dest = scaffold(
        WeekProject(
            year_week="2026-W38",
            slug="hello-t4",
            title="Hello T4",
            budget_hours=1,
            estimated_usd=0,
        ),
        projects_dir=tmp_path,
    )
    assert (dest / "job.py").exists()
    assert (dest / "task.py").exists()
    assert (dest / "run_manifest.json").exists()
    assert (dest / "kernel-metadata.json").exists()
    assert (dest / "results").is_dir()
