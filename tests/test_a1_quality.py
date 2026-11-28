import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.a1_quality import run_agent
IMAGE_DIR = Path("data/validation_check/val_check_20/images")

image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(IMAGE_DIR.glob(ext))

pass_count = 0
fail_count = 0

print("\n===== A1 IMAGE TESTING =====")

for image_path in sorted(image_files):
    result = run_agent(image_path.read_bytes(), {})

    status = result["quality_status"]

    if status == "PASS":
        pass_count += 1
    else:
        fail_count += 1

    print(
        f"{image_path.name} | "
        f"{status} | "
        f"Score: {result['quality_score']} | "
        f"Issues: {result['issues']}"
    )

print("\n===== SUMMARY =====")
print("Total images:", len(image_files))
print("PASS:", pass_count)
print("FAIL:", fail_count)
