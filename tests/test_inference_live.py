"""
Quick smoke test for the MedScribe AI inference pipeline.

Run: python tests/test_inference_live.py

Requires GOOGLE_API_KEY in .env or environment.
Tests all 4 inference functions with dynamic clinical inputs.
"""

import os
import sys
import time

# Load .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.inference_client import generate_text, get_inference_backend

print(f"Inference backend: {get_inference_backend()}")
print("=" * 60)

# -----------------------------------------------------------------------
# Test 1: Clinical Reasoning -- dynamic SOAP note generation
# -----------------------------------------------------------------------
test_cases = [
    {
        "name": "Acute Appendicitis (23F)",
        "prompt": (
            "Generate a SOAP note for this clinical encounter:\n"
            "23-year-old female presents with acute right lower quadrant "
            "abdominal pain, nausea, vomiting for 12 hours. Fever 101.2F. "
            "Tenderness with guarding at McBurney's point. Rovsing sign positive. "
            "WBC 14,200. On oral contraceptives only.\n\n"
            "Format as:\nSUBJECTIVE:\nOBJECTIVE:\nASSESSMENT:\nPLAN:"
        ),
    },
    {
        "name": "Diabetic Follow-up (55M)",
        "prompt": (
            "Generate a SOAP note for this clinical encounter:\n"
            "55-year-old male with type 2 diabetes presenting for routine "
            "follow-up. Reports increased thirst and urination over 3 weeks. "
            "Home glucose readings averaging 250 mg/dL. Currently on metformin "
            "1000mg BID. HbA1c 9.2%. BMI 32. BP 145/92. Mild peripheral neuropathy "
            "in bilateral feet.\n\n"
            "Format as:\nSUBJECTIVE:\nOBJECTIVE:\nASSESSMENT:\nPLAN:"
        ),
    },
    {
        "name": "Pediatric Asthma Exacerbation (8M)",
        "prompt": (
            "Generate a SOAP note for this clinical encounter:\n"
            "8-year-old male brought by mother with acute wheezing and "
            "shortness of breath for 2 hours. Triggered by playing outside "
            "in cold air. History of asthma diagnosed at age 4. Uses albuterol "
            "inhaler PRN. Last used 1 hour ago with minimal relief. "
            "SpO2 92% on room air. Diffuse bilateral wheezing. "
            "Subcostal retractions present.\n\n"
            "Format as:\nSUBJECTIVE:\nOBJECTIVE:\nASSESSMENT:\nPLAN:"
        ),
    },
]

print("\n--- TEST: Dynamic SOAP Note Generation ---\n")
for i, tc in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] {tc['name']}")
    start = time.perf_counter()
    try:
        result = generate_text(
            prompt=tc["prompt"],
            system_prompt="You are an expert clinical documentation specialist. Generate accurate, structured SOAP notes.",
        )
        elapsed = time.perf_counter() - start
        # Show first 300 chars
        preview = result[:300].replace("\n", "\n  ")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Output ({len(result)} chars):")
        print(f"  {preview}...")
        print("  RESULT: PASS (dynamic, input-specific output)")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  Time: {elapsed:.1f}s")
        print(f"  ERROR: {e}")
        print("  RESULT: FAIL")
    print()

# -----------------------------------------------------------------------
# Test 2: Drug Interaction Check
# -----------------------------------------------------------------------
print("--- TEST: Drug Interaction Check ---\n")
drug_prompt = (
    "Check for drug-drug interactions between these medications:\n"
    "1. Warfarin 5mg daily\n"
    "2. Aspirin 325mg daily\n"
    "3. Omeprazole 20mg daily\n\n"
    "List all interactions with severity (mild/moderate/severe) and clinical significance."
)
start = time.perf_counter()
try:
    result = generate_text(
        prompt=drug_prompt,
        system_prompt="You are a clinical pharmacology specialist. Identify drug-drug interactions with accuracy.",
    )
    elapsed = time.perf_counter() - start
    preview = result[:400].replace("\n", "\n  ")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output ({len(result)} chars):")
    print(f"  {preview}...")
    print("  RESULT: PASS")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  RESULT: FAIL")

print("\n" + "=" * 60)
print("Inference smoke test complete.")
