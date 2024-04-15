import os
import subprocess
import torch

def check_cuda_available():
    """Check if CUDA is available via PyTorch and return a boolean."""
    return torch.cuda.is_available()

def check_cuda_version():
    """Return the CUDA version used by PyTorch."""
    return torch.version.cuda

def check_nvidia_smi():
    """Run nvidia-smi command to check GPU and CUDA installation health."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"nvidia-smi failed to execute: {e}"

def check_env_path():
    """Check if CUDA path is correctly set in the system PATH environment variable."""
    env_path = os.environ.get('PATH', '')
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    return cuda_path in env_path

def check_cuda_path_env():
    """Check if CUDA_PATH environment variable is set correctly."""
    cuda_path = os.environ.get('CUDA_PATH', '')
    return cuda_path == r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

def main():
    print("Checking CUDA availability in PyTorch...")
    if check_cuda_available():
        print("CUDA is available in PyTorch.")
        print(f"CUDA version: {check_cuda_version()}")
    else:
        print("CUDA is not available in PyTorch. Please check your PyTorch installation and CUDA drivers.")

    print("\nRunning nvidia-smi to check GPU and CUDA status...")
    print(check_nvidia_smi())

    print("\nChecking system PATH for CUDA bin directory...")
    if check_env_path():
        print("CUDA bin directory is correctly set in the PATH.")
    else:
        print("CUDA bin directory is not found in the PATH. Please add it and try again.")

    print("\nChecking CUDA_PATH environment variable...")
    if check_cuda_path_env():
        print("CUDA_PATH environment variable is correctly set.")
    else:
        print("CUDA_PATH environment variable is not set correctly. Please set it to the CUDA installation directory.")

if __name__ == "__main__":
    main()
