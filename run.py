"""
run.py
Comprehensive local launcher and environment validator for FacePulse AI Attendance System.

Automated Workflow:
1. Environment & Python version check (>= 3.9)
2. Hardware & GPU detection (NVIDIA CUDA, AMD/Intel DirectML, or CPU fallback)
3. Dependency verification & automated installation
4. Model asset verification & automated downloading
5. ONNX Runtime provider testing (GPU accelerated with automatic CPU fallback)
6. Database & .env configuration checks
7. Local application launch with live status banner
"""

import sys
import os
import subprocess
import platform
import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE_FILE = BASE_DIR / ".env.example"
MODELS_DIR = BASE_DIR / "models"


def print_step(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_ok(msg):
    print(f"  [OK] {msg}")


def print_info(msg):
    print(f"  [INFO] {msg}")


def print_warn(msg):
    print(f"  [WARN] {msg}")


def print_err(msg):
    print(f"  [ERROR] {msg}")


# ------------------------------------------------------------------------------
# 1. Environment & Hardware Detection
# ------------------------------------------------------------------------------

def check_python_version():
    print_step("Step 1: Python Environment Check")
    major, minor = sys.version_info[:2]
    py_version = f"{major}.{minor}.{sys.version_info.micro}"
    print_info(f"Python executable: {sys.executable}")
    print_info(f"Python version   : {py_version} ({platform.architecture()[0]})")
    print_info(f"Platform         : {platform.system()} {platform.release()}")

    if (major, minor) < (3, 9):
        print_err(f"Python 3.9+ is required. Found Python {py_version}.")
        sys.exit(1)
    print_ok("Python version meets requirement (>= 3.9)")


def detect_gpus():
    """
    Detects available GPU hardware on Windows/Linux:
    - NVIDIA GPUs (via nvidia-smi / nvml)
    - AMD / Intel / Other GPUs (via Windows WMI / CIM or lspci)
    Returns: dict(has_nvidia=bool, has_amd=bool, has_intel=bool, gpu_names=list[str])
    """
    gpus = {
        "has_nvidia": False,
        "has_amd": False,
        "has_intel": False,
        "gpu_names": [],
    }

    # 1. Check NVIDIA via nvidia-smi
    smi_paths = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for smi in smi_paths:
        try:
            res = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, check=True)
            output = res.stdout.strip()
            if output:
                for line in output.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    name = parts[0]
                    gpus["has_nvidia"] = True
                    gpus["gpu_names"].append(name)
                break
        except Exception:
            continue

    # 2. On Windows, query Win32_VideoController for AMD, Intel, or other GPUs
    if platform.system() == "Windows":
        try:
            ps_cmd = 'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                                 capture_output=True, text=True)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    name = line.strip()
                    if not name or "Virtual" in name or "RDP" in name or "Basic Render" in name:
                        continue
                    if name not in gpus["gpu_names"]:
                        gpus["gpu_names"].append(name)
                    name_lower = name.lower()
                    if "nvidia" in name_lower or "geforce" in name_lower or "quadro" in name_lower or "rtx" in name_lower:
                        gpus["has_nvidia"] = True
                    elif "amd" in name_lower or "radeon" in name_lower:
                        gpus["has_amd"] = True
                    elif "intel" in name_lower or "iris" in name_lower or "arc" in name_lower:
                        gpus["has_intel"] = True
        except Exception as e:
            print_warn(f"Could not query Windows VideoController: {e}")

    return gpus


def check_and_configure_gpu(force_cpu=False):
    """
    Determines hardware acceleration strategy:
    - If force_cpu is True -> CPU only
    - If NVIDIA GPU -> recommends/installs onnxruntime-gpu
    - If AMD/Intel GPU on Windows -> recommends/installs onnxruntime-directml
    - If no GPU or GPU initialization fails -> CPU fallback
    """
    print_step("Step 2: Hardware Acceleration & GPU Detection")

    if force_cpu:
        print_info("User requested CPU-only mode (--cpu or FORCE_CPU=1).")
        os.environ["FORCE_CPU"] = "1"
        return "CPU"

    gpu_info = detect_gpus()
    if gpu_info["gpu_names"]:
        print_ok(f"Detected Display Adapter(s): {', '.join(gpu_info['gpu_names'])}")
    else:
        print_info("No dedicated GPU hardware detected. Using CPU execution mode.")
        return "CPU"

    # Evaluate target onnxruntime package
    target_package = None
    if gpu_info["has_nvidia"]:
        print_info("NVIDIA GPU detected. Optimal provider: CUDAExecutionProvider")
        target_package = "onnxruntime-gpu"
    elif platform.system() == "Windows" and (gpu_info["has_amd"] or gpu_info["has_intel"]):
        gpu_type = "AMD Radeon" if gpu_info["has_amd"] else "Intel Arc/Iris"
        print_info(f"{gpu_type} GPU detected on Windows. DirectX 12 hardware acceleration available.")
        target_package = "onnxruntime-directml"
    else:
        print_info("Standard GPU architecture detected. ONNX Runtime will use available providers or CPU.")

    return target_package or "CPU"


# ------------------------------------------------------------------------------
# 2. Dependency Management
# ------------------------------------------------------------------------------

REQUIRED_MODULES = {
    "fastapi": "fastapi>=0.100.0",
    "uvicorn": "uvicorn[standard]>=0.22.0",
    "pydantic": "pydantic>=2.0.0",
    "cv2": "opencv-python>=4.8.0.76",
    "numpy": "numpy>=1.24.0",
    "PIL": "Pillow>=10.0.0",
    "psycopg2": "psycopg2-binary>=2.9.9",
    "dotenv": "python-dotenv>=1.0.0",
    "openpyxl": "openpyxl>=3.1.0",
    "reportlab": "reportlab>=4.0.0",
    "multipart": "python-multipart>=0.0.6",
}


def verify_and_install_dependencies(target_runtime="CPU", no_install=False):
    print_step("Step 3: Dependencies Verification")

    missing = []
    for mod_name, pkg_req in REQUIRED_MODULES.items():
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg_req)

    # Check onnxruntime installation
    ort_installed = False
    try:
        import onnxruntime
        ort_installed = True
    except ImportError:
        pass

    if missing or not ort_installed:
        if no_install:
            print_err("Missing required packages and --no-install flag was passed.")
            print_err(f"Missing: {missing + (['onnxruntime'] if not ort_installed else [])}")
            sys.exit(1)

        print_warn("One or more required packages are missing. Installing now...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if missing:
            print_info(f"Installing missing core dependencies: {missing}")
            subprocess.run(cmd + missing, check=True)

        if not ort_installed:
            if target_runtime in ("onnxruntime-gpu", "onnxruntime-directml"):
                print_info(f"Installing hardware-accelerated runtime: {target_runtime}")
                try:
                    subprocess.run(cmd + [target_runtime], check=True)
                except Exception as e:
                    print_warn(f"Failed to install {target_runtime}: {e}. Falling back to standard onnxruntime...")
                    subprocess.run(cmd + ["onnxruntime>=1.16.0"], check=True)
            else:
                print_info("Installing standard CPU onnxruntime>=1.16.0")
                subprocess.run(cmd + ["onnxruntime>=1.16.0"], check=True)

    print_ok("All core application dependencies are satisfied.")


# ------------------------------------------------------------------------------
# 3. Model Assets Check & Download
# ------------------------------------------------------------------------------

def ensure_model_files():
    print_step("Step 4: Verifying AI Model Assets")
    try:
        from download_models import ensure_models
        ensure_models()
        print_ok("All required ONNX models are present in models/ directory.")
    except Exception as e:
        print_err(f"Model verification/download failed: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------------
# 4. Provider Testing & Resilient Fallback
# ------------------------------------------------------------------------------

def test_inference_and_fallback(force_cpu=False):
    """
    Tests ONNX Runtime session creation.
    If GPU provider fails (e.g. missing CUDA libraries), automatically falls back to CPU.
    """
    print_step("Step 5: Inference Engine & Provider Verification")

    if force_cpu:
        os.environ["FORCE_CPU"] = "1"
        print_info("Operating in CPU-only mode by request.")

    import onnxruntime as ort
    providers = ort.get_available_providers()
    print_info(f"Available ONNX Runtime Providers: {providers}")

    model_path = MODELS_DIR / "yolov8n-face.onnx"
    test_model = str(model_path) if model_path.exists() else None

    gpu_providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider"] if p in providers]

    active_provider = "CPUExecutionProvider"
    if gpu_providers and not force_cpu:
        target_gpu_prov = gpu_providers[0]
        print_info(f"Testing hardware acceleration provider: {target_gpu_prov} ...")
        try:
            if test_model:
                _ = ort.InferenceSession(test_model, providers=[target_gpu_prov, "CPUExecutionProvider"])
                print_ok(f"Hardware acceleration verified: using {target_gpu_prov}!")
                active_provider = target_gpu_prov
            else:
                print_ok(f"Provider {target_gpu_prov} registered in runtime.")
                active_provider = target_gpu_prov
        except Exception as e:
            print_warn(f"GPU provider {target_gpu_prov} initialization failed: {e}")
            print_warn("Falling back safely to CPUExecutionProvider. Performance remains reliable.")
            os.environ["FORCE_CPU"] = "1"
            active_provider = "CPUExecutionProvider"
    else:
        print_ok("Inference engine initialized on CPU (CPUExecutionProvider).")
        active_provider = "CPUExecutionProvider"

    return active_provider


# ------------------------------------------------------------------------------
# 5. Configuration & Database Check
# ------------------------------------------------------------------------------

def check_configuration_and_database():
    print_step("Step 6: Configuration & Database Validation")

    # 1. Check .env file
    if not ENV_FILE.exists():
        if ENV_EXAMPLE_FILE.exists():
            print_warn(".env file not found. Copying default configuration from .env.example...")
            shutil.copy(ENV_EXAMPLE_FILE, ENV_FILE)
            print_ok("Created .env from .env.example")
        else:
            print_warn("Neither .env nor .env.example found. Default environment variables will be used.")

    # 2. Check Database Connectivity
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        import app.database as database
        print_info("Testing Postgres / Supabase database connection...")
        database.init_db()
        print_ok("Database connection verified and schema initialized successfully.")
    except Exception as e:
        print_warn(f"Database connection note: {e}")
        print_info("The server will start, but database operations may fail until valid credentials are in .env.")


# ------------------------------------------------------------------------------
# 6. Launch Server
# ------------------------------------------------------------------------------

def launch_server(host="127.0.0.1", port=7860, active_provider="CPUExecutionProvider"):
    print_step("Step 7: Launching FacePulse AI Attendance Server")

    banner_mode = f"GPU ACCELERATED ({active_provider})" if active_provider != "CPUExecutionProvider" else "CPU MODE"
    print(f"""
    ===============================================================
               FacePulse AI - Attendance & Recognition
    ===============================================================
      * Local UI        : http://{host}:{port}/
      * Admin Dashboard : http://{host}:{port}/admin
      * OpenAPI Docs    : http://{host}:{port}/docs
      * Health Check    : http://{host}:{port}/health
      * Hardware Mode   : {banner_mode}
    ===============================================================
    Press Ctrl+C to stop the server at any time.
    """)

    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FacePulse AI Attendance System — Local Launcher with GPU auto-detection and CPU fallback"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    parser.add_argument("--cpu", "--force-cpu", action="store_true", help="Force CPU mode, bypassing any GPU")
    parser.add_argument("--no-install", action="store_true", help="Skip automatic pip package installation")
    parser.add_argument("--check-only", action="store_true", help="Run environment and model checks without starting server")
    args = parser.parse_args()

    # Step 1: Python check
    check_python_version()

    # Step 2: GPU hardware detection
    target_runtime = check_and_configure_gpu(force_cpu=args.cpu)

    # Step 3: Dependencies check and auto-install
    verify_and_install_dependencies(target_runtime=target_runtime, no_install=args.no_install)

    # Step 4: Model assets
    ensure_model_files()

    # Step 5: Provider test & CPU fallback
    active_provider = test_inference_and_fallback(force_cpu=args.cpu)

    # Step 6: Config & Database check
    check_configuration_and_database()

    # Step 7: Launch
    if args.check_only:
        print_step("Pre-flight Checks Complete")
        print_ok("System is ready for local execution. Run without --check-only to start server.")
        return

    launch_server(host=args.host, port=args.port, active_provider=active_provider)


if __name__ == "__main__":
    main()
