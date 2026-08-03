from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from .base import Capabilities, GPUInfo, Job, JobResult, parse_nvidia_smi


def _ssh_hint(stderr: str) -> str:
    """Condense ssh's failure into one line, with the fix where there is one.

    ssh emits twenty lines of banner for a changed host key and buries the
    actionable sentence in the middle. Verbatim, that is unreadable in a job
    log; discarded, it costs a real debugging session. So the known cases get
    named and everything else falls back to the last non-empty line.
    """
    text = stderr.strip()
    if not text:
        return ""
    low = text.lower()

    if "host key verification failed" in low or "remote host identification has changed" in low:
        return (
            "SSH host key mismatch. This is normal after a VM is recreated with "
            "the same name -- the new machine has a new key. Refresh the alias "
            "with: gcloud compute config-ssh"
        )
    if "permission denied" in low:
        return "SSH permission denied -- check the key and the login user"
    if "could not resolve hostname" in low or "name or service not known" in low:
        return "hostname does not resolve -- is the alias from gcloud compute config-ssh?"
    if "connection refused" in low:
        return "connection refused -- the host may still be booting"
    if "connection timed out" in low or "operation timed out" in low:
        return "connection timed out -- check the VM is running and the firewall allows SSH"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


class SSHExecutor:
    def __init__(
        self,
        host: str,
        # Relative to the remote login dir ($HOME): a quoted "~" would NOT
        # expand on the remote shell, so the default must not contain one.
        base_dir: str = ".pixiecad-jobs",
        ssh_opts: list[str] | None = None,
    ):
        self.host = host
        self.base_dir = base_dir
        # Generative jobs sit silent for many minutes while the remote worker
        # loads a model, and an idle SSH channel is exactly what NAT and
        # firewalls drop. Keepalives make the connection survive that silence,
        # and the count bounds how long a genuinely dead host wedges us.
        self.ssh_opts = [
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10",
        ]
        if ssh_opts:
            self.ssh_opts.extend(ssh_opts)

    def _exec(
        self, cmd: list[str], timeout_s: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            # Keep the "remote failure -> failed JobResult, not exception"
            # contract for every caller through this single seam.
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return subprocess.CompletedProcess(
                args=cmd, returncode=124, stdout=stdout,
                stderr=stderr + f"\ntimed out after {timeout_s}s",
            )

    def probe(self) -> Capabilities:
        echo_cmd = ["ssh"] + self.ssh_opts + [self.host, "echo ok"]
        try:
            res = self._exec(echo_cmd, timeout_s=5)
            if res.returncode != 0:
                return Capabilities(
                    hostname=self.host,
                    gpu=None,
                    reachable=False,
                    detail=_ssh_hint(res.stderr or ""),
                )
        except Exception as exc:
            return Capabilities(
                hostname=self.host, gpu=None, reachable=False, detail=str(exc)
            )

        smi_cmd = [
            "ssh",
            *self.ssh_opts,
            self.host,
            "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits",
        ]
        gpu: GPUInfo | None = None
        detail = ""
        try:
            smi_res = self._exec(smi_cmd, timeout_s=10)
            if smi_res.returncode == 0:
                gpu = parse_nvidia_smi(smi_res.stdout)
                if gpu is None:
                    detail = f"nvidia-smi output not understood: {smi_res.stdout[:200]!r}"
            else:
                detail = _ssh_hint(smi_res.stderr or "") or "nvidia-smi failed"
        except Exception as exc:
            detail = str(exc)

        return Capabilities(hostname=self.host, gpu=gpu, reachable=True, detail=detail)

    def run(self, job: Job) -> JobResult:
        start_time = time.monotonic()
        remote_job_dir = f"{self.base_dir}/{job.remote_subdir}"

        # 1. Create remote workspace directory
        mkdir_cmd = ["ssh"] + self.ssh_opts + [self.host, f"mkdir -p {shlex.quote(remote_job_dir)}"]
        res = self._exec(mkdir_cmd, timeout_s=30)
        if res.returncode != 0:
            duration = time.monotonic() - start_time
            return JobResult(
                returncode=res.returncode,
                stdout_tail=(res.stdout or "")[-4000:],
                stderr_tail=(res.stderr or "")[-4000:],
                duration_s=duration,
            )

        # 2. rsync inputs
        for inp in job.inputs:
            rsync_cmd = ["rsync", "-az"]
            if self.ssh_opts:
                ssh_opts_str = " ".join(shlex.quote(opt) for opt in self.ssh_opts)
                rsync_cmd.extend(["-e", f"ssh {ssh_opts_str}"])
            rsync_cmd.extend([str(inp), f"{self.host}:{remote_job_dir}/"])

            res = self._exec(rsync_cmd, timeout_s=job.timeout_s)
            if res.returncode != 0:
                duration = time.monotonic() - start_time
                return JobResult(
                    returncode=res.returncode,
                    stdout_tail=(res.stdout or "")[-4000:],
                    stderr_tail=(res.stderr or "")[-4000:],
                    duration_s=duration,
                )

        # 3. Remote execution
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in job.command)
        remote_exec_str = f"cd {shlex.quote(remote_job_dir)} && {cmd_str}"
        ssh_run_cmd = ["ssh"] + self.ssh_opts + [self.host, remote_exec_str]

        res = self._exec(ssh_run_cmd, timeout_s=job.timeout_s)
        if res.returncode != 0:
            duration = time.monotonic() - start_time
            return JobResult(
                returncode=res.returncode,
                stdout_tail=(res.stdout or "")[-4000:],
                stderr_tail=(res.stderr or "")[-4000:],
                duration_s=duration,
            )

        cmd_stdout = res.stdout
        cmd_stderr = res.stderr

        # 4. rsync output back
        job.output_dir.mkdir(parents=True, exist_ok=True)
        rsync_out_cmd = ["rsync", "-az"]
        if self.ssh_opts:
            ssh_opts_str = " ".join(shlex.quote(opt) for opt in self.ssh_opts)
            rsync_out_cmd.extend(["-e", f"ssh {ssh_opts_str}"])
        rsync_out_cmd.extend([f"{self.host}:{remote_job_dir}/out/", f"{job.output_dir}/"])

        res = self._exec(rsync_out_cmd, timeout_s=job.timeout_s)
        duration = time.monotonic() - start_time

        if res.returncode != 0:
            return JobResult(
                returncode=res.returncode,
                stdout_tail=(res.stdout or "")[-4000:],
                stderr_tail=(res.stderr or "")[-4000:],
                duration_s=duration,
            )

        return JobResult(
            returncode=0,
            stdout_tail=(cmd_stdout or "")[-4000:],
            stderr_tail=(cmd_stderr or "")[-4000:],
            duration_s=duration,
        )
