from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from .base import Capabilities, GPUInfo, Job, JobResult, parse_nvidia_smi


class LocalExecutor:
    def probe(self) -> Capabilities:
        hostname = socket.gethostname()
        gpu: GPUInfo | None = None
        smi_path = shutil.which("nvidia-smi")
        if smi_path:
            try:
                res = subprocess.run(
                    [
                        smi_path,
                        "--query-gpu=name,memory.total,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    gpu = parse_nvidia_smi(res.stdout)
            except Exception:
                gpu = None
        return Capabilities(hostname=hostname, gpu=gpu, reachable=True)

    def run(self, job: Job) -> JobResult:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            for inp in job.inputs:
                inp_path = Path(inp)
                target = temp_dir / inp_path.name
                if not target.exists():
                    target.symlink_to(inp_path.resolve())

            try:
                res = subprocess.run(
                    job.command,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=job.timeout_s,
                )
                duration = time.monotonic() - start_time
                stdout_tail = (res.stdout or "")[-4000:]
                stderr_tail = (res.stderr or "")[-4000:]
                returncode = res.returncode

                # Inputs are symlinks; anything the job *wrote* is a real file,
                # so an output that reuses an input's name is still collected.
                for item in temp_dir.iterdir():
                    if not item.is_symlink():
                        dest = job.output_dir / item.name
                        if dest.exists():
                            if dest.is_dir():
                                shutil.rmtree(dest)
                            else:
                                dest.unlink()
                        shutil.move(str(item), str(dest))

                return JobResult(
                    returncode=returncode,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                    duration_s=duration,
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - start_time
                stdout_tail = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
                stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
                return JobResult(
                    returncode=-1,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail + "\nJob timed out",
                    duration_s=duration,
                )
