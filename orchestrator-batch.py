import time
from pathlib import Path
from brats import AfricaSegmenter
from brats.constants import AfricaAlgorithms

segmenter = AfricaSegmenter(algorithm=AfricaAlgorithms.BraTS25_1, cuda_devices="0")

input_path = Path("/teamspace/studios/this_studio/BraTS2025-GLI-PRE-Challenge-TrainingData")
output_path = Path("/teamspace/studios/this_studio/Results/BraTS2025-GLI-Batch-Results")
output_path.mkdir(parents=True, exist_ok=True)

subject_folders = sorted([f for f in input_path.iterdir() if f.is_dir()])
total = len(subject_folders)
print(f"Found {total} subjects")

completed, skipped, failed = [], [], []
durations = []

batch_start = time.time()

for i, subject_folder in enumerate(subject_folders, 1):
    subject_id = subject_folder.name
    output_file = output_path / f"{subject_id}.nii.gz"

    # Skip if already done
    if output_file.exists():
        print(f"[SKIP] ({i}/{total}) {subject_id} — output already exists")
        skipped.append(subject_id)
        continue

    # Locate modality files
    t1c = next(subject_folder.glob("*t1c*"), None)
    t1n = next(subject_folder.glob("*t1n*"), None)
    t2f = next(subject_folder.glob("*t2f*"), None)
    t2w = next(subject_folder.glob("*t2w*"), None)

    if not all([t1c, t1n, t2f, t2w]):
        print(f"[SKIP] ({i}/{total}) {subject_id} — missing modalities: "
              f"t1c={bool(t1c)}, t1n={bool(t1n)}, t2f={bool(t2f)}, t2w={bool(t2w)}")
        skipped.append(subject_id)
        continue

    try:
        print(f"[RUN ] ({i}/{total}) {subject_id}")
        t_start = time.time()

        segmenter.infer_single(
            t1c=str(t1c),
            t1n=str(t1n),
            t2f=str(t2f),
            t2w=str(t2w),
            output_file=str(output_file),
        )

        elapsed = time.time() - t_start
        durations.append(elapsed)

        # ETA based on average of completed inferences
        avg = sum(durations) / len(durations)
        remaining = total - i - len(skipped)
        eta_s = avg * remaining
        eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(eta_s)) if remaining > 0 else "—"

        print(f"[DONE] ({i}/{total}) {subject_id} — {elapsed:.1f}s | avg {avg:.1f}s | ETA {eta_str}")
        completed.append(subject_id)
    except Exception as e:
        print(f"[FAIL] ({i}/{total}) {subject_id} — {e}")
        failed.append(subject_id)

total_elapsed = time.time() - batch_start
avg_duration = sum(durations) / len(durations) if durations else 0

print(f"\n=== Summary ===")
print(f"  Completed : {len(completed)}")
print(f"  Skipped   : {len(skipped)}")
print(f"  Failed    : {len(failed)}")
print(f"  Total time: {time.strftime('%Hh %Mm %Ss', time.gmtime(total_elapsed))}")
print(f"  Avg/case  : {avg_duration:.1f}s")
if failed:
    print(f"  Failed IDs: {failed}")