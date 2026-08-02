import subprocess
from pathlib import Path

from pixiecad.executors import (
    Capabilities,
    GPUInfo,
    Job,
    JobResult,
    LocalExecutor,
    SSHExecutor,
    parse_nvidia_smi,
)


def test_parse_nvidia_smi_good_line():
    info = parse_nvidia_smi("NVIDIA GeForce GTX 1060, 6144, 6.1")
    assert info == GPUInfo(name="NVIDIA GeForce GTX 1060", vram_mb=6144, compute_cap=6.1)


def test_parse_nvidia_smi_mib_variant():
    info = parse_nvidia_smi("NVIDIA RTX 3080, 10240 MiB, 8.6")
    assert info == GPUInfo(name="NVIDIA RTX 3080", vram_mb=10240, compute_cap=8.6)


def test_parse_nvidia_smi_garbage():
    assert parse_nvidia_smi("invalid output line") is None
    assert parse_nvidia_smi("NVIDIA Card, invalid_vram, 6.1") is None


def test_parse_nvidia_smi_empty():
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("   \n  ") is None


def test_capabilities_cuda_ok():
    gpu = GPUInfo(name="GTX 1060", vram_mb=6144, compute_cap=6.1)
    cap_ok = Capabilities(hostname="host", gpu=gpu, reachable=True)
    assert cap_ok.cuda_ok() is True
    assert cap_ok.cuda_ok(min_vram_mb=4000) is True
    assert cap_ok.cuda_ok(min_vram_mb=8000) is False

    cap_unreachable = Capabilities(hostname="host", gpu=gpu, reachable=False)
    assert cap_unreachable.cuda_ok() is False

    cap_no_gpu = Capabilities(hostname="host", gpu=None, reachable=True)
    assert cap_no_gpu.cuda_ok() is False


def test_local_executor_probe():
    executor = LocalExecutor()
    caps = executor.probe()
    assert caps.reachable is True
    assert isinstance(caps.hostname, str)


def test_local_executor_run(tmp_path: Path):
    input_file = tmp_path / "in.txt"
    input_file.write_text("hello input")
    output_dir = tmp_path / "out"

    job = Job(
        command=["sh", "-c", "cat in.txt && echo 'hello output' > result.txt"],
        inputs=[input_file],
        output_dir=output_dir,
        remote_subdir="test-job",
    )

    executor = LocalExecutor()
    res = executor.run(job)

    assert res.ok is True
    assert res.returncode == 0
    assert "hello input" in res.stdout_tail
    assert (output_dir / "result.txt").exists()
    assert (output_dir / "result.txt").read_text().strip() == "hello output"


def test_ssh_executor_unreachable(monkeypatch):
    executor = SSHExecutor(host="remote.test")

    def fake_exec(cmd, timeout_s=None):
        return subprocess.CompletedProcess(
            args=cmd, returncode=255, stdout="", stderr="Connection timed out"
        )

    monkeypatch.setattr(executor, "_exec", fake_exec)
    caps = executor.probe()
    assert caps.reachable is False
    assert caps.gpu is None
    assert caps.hostname == "remote.test"


def test_ssh_executor_reachable_with_gpu(monkeypatch):
    executor = SSHExecutor(host="remote.test")
    calls = []

    def fake_exec(cmd, timeout_s=None):
        calls.append(cmd)
        if "echo ok" in cmd[-1]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")
        elif "nvidia-smi" in cmd[-1]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="NVIDIA RTX 3080, 10240, 8.6\n", stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(executor, "_exec", fake_exec)
    caps = executor.probe()
    assert caps.reachable is True
    assert caps.gpu == GPUInfo(name="NVIDIA RTX 3080", vram_mb=10240, compute_cap=8.6)
    assert len(calls) == 2


def test_ssh_executor_run_happy_path(monkeypatch, tmp_path: Path):
    executor = SSHExecutor(host="remote.test", base_dir="/tmp/jobs")
    calls = []

    def fake_exec(cmd, timeout_s=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="step success", stderr=""
        )

    monkeypatch.setattr(executor, "_exec", fake_exec)

    input1 = tmp_path / "in1.txt"
    input1.write_text("input 1")
    output_dir = tmp_path / "out"

    job = Job(
        command=["python", "process.py", "--flag"],
        inputs=[input1],
        output_dir=output_dir,
        remote_subdir="my_job_1",
    )

    res = executor.run(job)

    assert res.ok is True
    assert res.returncode == 0
    assert res.stdout_tail == "step success"

    assert len(calls) == 4
    assert calls[0] == ["ssh", "remote.test", "mkdir -p /tmp/jobs/my_job_1"]
    assert calls[1] == ["rsync", "-az", str(input1), "remote.test:/tmp/jobs/my_job_1/"]
    assert calls[2] == [
        "ssh",
        "remote.test",
        "cd /tmp/jobs/my_job_1 && python process.py --flag",
    ]
    assert calls[3] == [
        "rsync",
        "-az",
        "remote.test:/tmp/jobs/my_job_1/out/",
        f"{output_dir}/",
    ]


def test_ssh_executor_run_rsync_input_failure(monkeypatch, tmp_path: Path):
    executor = SSHExecutor(host="remote.test", base_dir="/tmp/jobs")
    calls = []

    def fake_exec(cmd, timeout_s=None):
        calls.append(cmd)
        if cmd[0] == "rsync":
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="rsync error: network down"
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(executor, "_exec", fake_exec)

    input1 = tmp_path / "in1.txt"
    input1.write_text("input 1")
    output_dir = tmp_path / "out"

    job = Job(
        command=["python", "process.py"],
        inputs=[input1],
        output_dir=output_dir,
        remote_subdir="my_job_2",
    )

    res = executor.run(job)

    assert res.ok is False
    assert res.returncode == 1
    assert "rsync error" in res.stderr_tail
    assert len(calls) == 2
