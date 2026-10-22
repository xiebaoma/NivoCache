#!/usr/bin/env python3
"""Archive and restore a prebuilt CacheLib SDK on compatible Linux hosts."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile


REVISION = "f4012d692d849d1ae9927dd17ea186fb45f6ef00"
PACKAGES = (
    "build-essential", "gcc-13", "g++-13", "cmake", "git", "python3",
    "pkg-config", "libssl-dev", "libgflags-dev", "libevent-dev", "zlib1g-dev",
    "liblz4-dev", "libzstd-dev", "libsnappy-dev", "libdwarf-dev",
    "libiberty-dev", "libaio-dev", "libsodium-dev", "libxxhash-dev",
    "libgtest-dev", "libgmock-dev",
)


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def host_info():
    release = platform.freedesktop_os_release()
    return {"os": release["ID"], "version": release["VERSION_ID"],
            "arch": platform.machine()}


def check_source(root):
    source = str(root / "source")
    if output("git", "-C", source, "rev-parse", "HEAD") != REVISION:
        raise RuntimeError("CacheLib is not at the pinned revision")
    if output("git", "-C", source, "status", "--porcelain"):
        raise RuntimeError("CacheLib source has local changes")


def pack(args):
    root = args.work_root.resolve(strict=True)
    archive = args.archive.resolve()
    checksum = Path(str(archive) + ".sha256")
    if archive.exists() or checksum.exists():
        raise RuntimeError("Refusing to overwrite an existing bundle or checksum")
    check_source(root)
    for relative in ("install/cachelib/lib/libcachelib_allocator.a", "environment.sh"):
        if not (root / relative).is_file():
            raise RuntimeError(f"Missing {root / relative}")
    package_versions = {}
    for package in PACKAGES:
        status = output("dpkg-query", "-W", "-f=${db:Status-Status}", package)
        if status != "installed":
            raise RuntimeError(f"Required system package is not installed: {package}")
        package_versions[package] = output("dpkg-query", "-W", "-f=${Version}", package)
    manifest = {"format": 1, "host": host_info(), "root": str(root),
                "cachelib_revision": REVISION,
                "compiler": output("g++-13", "-dumpfullversion"),
                "system_packages": package_versions,
                "cpu_model": next((line.split(":", 1)[1].strip() for line in
                                   Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")), "unknown"),
                "note": "Prebuilt SDK; no scratch builds, virtualenv, credentials, or NivoCache sources."}
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packing installed SDK and pinned source from {root}", flush=True)
    # Build privately, then publish the archive only after tar has finished.
    with tempfile.TemporaryDirectory(prefix="cachelib-pack-", dir=archive.parent) as temp:
        temporary = Path(temp) / archive.name
        with tarfile.open(temporary, "w:gz", compresslevel=3) as bundle:
            data = json.dumps(manifest, indent=2).encode() + b"\n"
            header = tarfile.TarInfo("manifest.json")
            header.size = len(data)
            header.mode = 0o644
            bundle.addfile(header, io.BytesIO(data))

            def keep(member):
                if Path(member.name).name.startswith("._"):
                    return None
                return member

            for relative in ("install", "source", "environment.sh"):
                bundle.add(root / relative, arcname="sdk/" + relative, filter=keep)
        temporary.rename(archive)
    checksum.write_text(f"{digest(archive)}  {archive.name}\n")
    print(f"Bundle: {archive}\nSize: {archive.stat().st_size / 1024**2:.1f} MiB")
    print(f"Checksum: {checksum}\nRestore path: {root}")


def restore(args):
    archive = args.archive.resolve(strict=True)
    checksum = Path(str(archive) + ".sha256")
    expected = checksum.read_text().split()[0]
    if digest(archive) != expected:
        raise RuntimeError("Bundle SHA-256 does not match its checksum file")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        if manifest["format"] != 1 or manifest["cachelib_revision"] != REVISION:
            raise RuntimeError("Unsupported bundle format or CacheLib revision")
        if manifest["host"] != host_info():
            raise RuntimeError(f"Host mismatch: required {manifest['host']}, got {host_info()}")
        root = Path(manifest["root"])
        if not root.is_absolute() or len(root.parts) < 3 or root.name != "cachelib-dependency":
            raise RuntimeError("Unsupported SDK restore path")
        if root.exists() or root.is_symlink():
            raise RuntimeError(f"Refusing to overwrite existing SDK: {root}")
        if args.install_system_packages:
            if os.geteuid() != 0:
                raise RuntimeError("Installing system packages requires root")
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", *PACKAGES],
                           check=True, env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
        differences = []
        for package in PACKAGES:
            try:
                status = output("dpkg-query", "-W", "-f=${db:Status-Status}", package)
                version = output("dpkg-query", "-W", "-f=${Version}", package)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Install system dependencies with --install-system-packages: {package}") from error
            if status != "installed":
                raise RuntimeError(f"Required package not installed: {package}")
            if version != manifest["system_packages"][package]:
                differences.append(f"{package}: {manifest['system_packages'][package]} -> {version}")
        compiler = output("g++-13", "-dumpfullversion")
        if compiler.split(".")[0] != manifest["compiler"].split(".")[0]:
            raise RuntimeError("Compiler major version mismatch")
        root.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root.parent).free < 3 * 1024**3:
            raise RuntimeError("At least 3 GiB of free space is required for SDK extraction")
        # Extraction filters reject absolute paths and escaping symlinks.
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError("Python with tarfile.data_filter is required (Ubuntu 24.04)")
        with tempfile.TemporaryDirectory(prefix="cachelib-restore-", dir=root.parent) as temp:
            def safe_member(member, destination):
                if member.name != "manifest.json" and not member.name.startswith("sdk/"):
                    raise RuntimeError(f"Unexpected archive entry: {member.name}")
                return tarfile.data_filter(member, destination)

            bundle.extractall(temp, filter=safe_member)
            staged = Path(temp) / "sdk"
            check_source(staged)
            (staged / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            staged.rename(root)
    print(f"Restored SDK to {root}. No dependency compilation was needed.")
    if differences:
        print("System package versions differ from the export host:")
        print("\n".join(differences))
        print("Run a smoke benchmark and record these versions before comparing results.")
    print("Next, from the NivoCache checkout:")
    print(f"CC=gcc-13 CXX=g++-13 bash benchmarks/scripts/run_comparison.sh {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("pack", help="Export installed dependencies; excludes build scratch")
    exporter.add_argument("archive", type=Path)
    exporter.add_argument("--work-root", type=Path, default=Path("/root/cachelib-dependency"))
    importer = commands.add_parser("restore", help="Restore at the original absolute SDK path")
    importer.add_argument("archive", type=Path)
    importer.add_argument("--install-system-packages", action="store_true",
                          help="Install Ubuntu build/runtime packages using apt (network required)")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run pack/restore on Linux. The archive can be stored on macOS.")
    try:
        if args.command == "pack":
            pack(args)
        else:
            restore(args)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
