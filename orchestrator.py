from pathlib import Path
from brats import AfricaSegmenter
from brats.constants import AfricaAlgorithms

segmenter = AfricaSegmenter(algorithm=AfricaAlgorithms.BraTS25_1, cuda_devices="0")

input_path = Path("/content/BraTS-Africa/95_Glioma")
output_path = Path("Results/BraTS-Africa(Extended)")

segmenter.infer_batch(
    data_folder=input_path,
    output_folder=output_path,
    backend="singularity"
)

print(f"Inferred segmentations: {[path.name for path in output_path.iterdir()]}")