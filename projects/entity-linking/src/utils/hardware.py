import logging
import platform
import sys

logger = logging.getLogger(__name__)

# Try to import torch optionally to handle hardware detection without strict PyTorch dependency
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import spacy optionally to configure its backend
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


def get_device_type() -> str:
    """
    Detect the most powerful available hardware accelerator.
    Returns:
        str: 'cuda' for NVIDIA GPUs, 'rocm' for AMD GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if HAS_TORCH:
        if torch.cuda.is_available():
            # Check if this is a ROCm/HIP build of PyTorch
            if hasattr(torch, "version") and getattr(torch.version, "hip", None) is not None:
                return "rocm"
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    else:
        # Fallback heuristic if PyTorch is not yet installed
        system = platform.system()
        machine = platform.machine()
        
        # Check for Apple Silicon
        if system == "Darwin" and (machine == "arm64" or "arm" in platform.processor().lower()):
            return "mps"
        
        # Check if AMD/ROCm GPU is present (heuristic via simple path/command check)
        import shutil
        if shutil.which("rocm-smi") is not None or shutil.which("rocminfo") is not None:
            return "rocm"
        
        # Check if NVIDIA GPU is present (heuristic via simple path/command check)
        if shutil.which("nvidia-smi") is not None:
            return "cuda"
            
    return "cpu"


def get_torch_device():
    """
    Get the PyTorch device object for the preferred hardware accelerator.
    Returns:
        torch.device or None: PyTorch device object if torch is installed, else None.
    """
    if not HAS_TORCH:
        logger.warning("PyTorch (torch) is not installed. Returning None for device.")
        return None
        
    device_type = get_device_type()
    # PyTorch uses "cuda" for both NVIDIA CUDA and AMD ROCm/HIP device types
    torch_device_type = "cuda" if device_type == "rocm" else device_type
    return torch.device(torch_device_type)


def configure_pipeline_hardware() -> bool:
    """
    Configure spaCy/medspaCy to run on the optimal hardware backend.
    Returns:
        bool: True if GPU/accelerator activation succeeded, False if fallback to CPU.
    """
    if not HAS_SPACY:
        logger.warning("spaCy is not installed. Cannot configure pipeline hardware.")
        return False
        
    device = get_device_type()
    
    if device in ("cuda", "rocm"):
        # spaCy requires CuPy or PyTorch for GPU acceleration.
        # prefer_gpu() will try to allocate GPU resources.
        try:
            gpu_activated = spacy.prefer_gpu()
            if gpu_activated:
                logger.info(f"Successfully activated {device.upper()} GPU for spaCy/medspaCy pipelines.")
                return True
            else:
                logger.warning(f"{device.upper()} detected but spaCy prefer_gpu() returned False. Falling back to CPU for spaCy.")
        except Exception as e:
            logger.error(f"Error configuring spaCy GPU: {e}. Falling back to CPU.")
    elif device == "mps":
        logger.info("Apple Silicon (MPS) detected. Neural networks and Hugging Face components in spaCy will be directed to use MPS.")
        # spaCy pipelines utilizing thinc/pytorch components can leverage MPS.
    else:
        logger.info("Using standard CPU for spaCy/medspaCy pipelines.")
        
    return False


def get_hardware_summary() -> dict:
    """
    Gather detailed information about the system and available accelerators.
    Returns:
        dict: Diagnostic details of system hardware.
    """
    device_type = get_device_type()
    
    summary = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_installed": HAS_TORCH,
        "spacy_installed": HAS_SPACY,
        "preferred_accelerator": device_type.upper(),
    }
    
    if HAS_TORCH:
        summary["torch_version"] = torch.__version__
        if device_type == "cuda":
            summary["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            summary["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            summary["cuda_capability"] = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "Unknown"
        elif device_type == "rocm":
            summary["rocm_version"] = getattr(torch.version, "hip", "Unknown")
            summary["rocm_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            summary["rocm_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        elif device_type == "mps":
            summary["mps_built"] = torch.backends.mps.is_built()
            
    if HAS_SPACY:
        summary["spacy_version"] = spacy.__version__
        
    return summary


def log_hardware_status() -> None:
    """
    Print or log a clean, human-readable summary of the active hardware accelerator.
    """
    info = get_hardware_summary()
    
    border = "=" * 55
    logger.info(border)
    logger.info(" CLINICAL ENTITY LINKING PIPELINE - HARDWARE DETECTION")
    logger.info(border)
    logger.info(f" Operating System       : {info['os']} ({info['architecture']})")
    logger.info(f" Python Version         : {info['python_version']}")
    logger.info(f" PyTorch Installed      : {info['torch_installed']}")
    logger.info(f" spaCy Installed        : {info['spacy_installed']}")
    logger.info(f" Preferred Accelerator  : {info['preferred_accelerator']}")
    
    if info.get("cuda_device_name"):
        logger.info(f" GPU Model              : {info['cuda_device_name']}")
        logger.info(f" CUDA Capability        : {info['cuda_capability']}")
    elif info.get("rocm_device_name"):
        logger.info(f" GPU Model              : {info['rocm_device_name']}")
        logger.info(f" ROCm Version           : {info['rocm_version']}")
        logger.info(f" ROCm Device Count      : {info['rocm_device_count']}")
    elif info.get("mps_built"):
        logger.info(" MPS Acceleration       : Enabled (Apple Silicon)")
        
    logger.info(border)
