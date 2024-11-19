import torch
import intel_extension_for_pytorch as ipex
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator

class XPUAccelerator(Accelerator):
    """Use Intel Extension for PyTorch with XPU support."""

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    def get_device_stats(self, device: torch.device) -> Dict[str, Any]:
        # Implement any optional device stats retrieval
        return {}

# Trainer configuration in your main script:
trainer = Trainer(accelerator=XPUAccelerator(), devices="auto")
