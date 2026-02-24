"""Full pipeline smoke test -- 7 agents, 6 phases."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agents.orchestrator import ClinicalOrchestrator


async def main():
    print("=" * 60)
    print("MedScribe AI -- 7-Agent Pipeline Smoke Test")
    print("=" * 60)

    o = ClinicalOrchestrator()
    status = o.initialize_all()
    print(f"\nAgent Status ({sum(status.values())}/{len(status)} ready):")
    for name, ready in status.items():
        icon = "[OK]" if ready else "[--]"
        print(f"  {icon} {name}")

    print("\nRunning full pipeline...")
    result = await o.run_full_pipeline(
        text_input=(
            "45-year-old male presenting with progressive shortness of breath "
            "over the past 3 days. History of hypertension on lisinopril 10mg daily "
            "and type 2 diabetes on metformin 1000mg BID. Currently taking azithromycin "
            "500mg for URI. Vitals: BP 145/92, HR 98, RR 22, SpO2 93% on room air. "
            "Bilateral basilar crackles on auscultation. "
            "Assessment: Possible community-acquired pneumonia with CHF exacerbation."
        )
    )

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE -- {result.total_processing_time_ms:.0f}ms")
    print(f"{'='*60}")
    print(f"\nTranscript: {'YES' if result.transcript else 'NO'} ({len(result.transcript or '')} chars)")

    if result.triage_result:
        print(f"Triage: {result.triage_result.get('predicted_specialty', 'N/A')} "
              f"({result.triage_result.get('confidence', 0):.0%})")

    print(f"Image: {'YES' if result.image_findings else 'NO'}")

    if result.soap_note:
        print("\nSOAP Note:")
        for section in ["subjective", "objective", "assessment", "plan"]:
            val = getattr(result.soap_note, section, "")
            print(f"  {section.upper()}: {val[:100]}...")

    print(f"\nICD-10 Codes ({len(result.icd_codes)}):")
    for code in result.icd_codes:
        print(f"  {code}")

    if result.drug_interactions:
        di = result.drug_interactions
        print("\nDrug Interactions:")
        print(f"  Medications: {len(di.get('medications_found', []))}")
        print(f"  Interactions: {len(di.get('interactions', []))}")
        print(f"  Safe: {di.get('safe', 'N/A')}")
        print(f"  Summary: {di.get('summary', '')}")

    if result.quality_report:
        qr = result.quality_report
        print("\nQuality Report:")
        print(f"  Score: {qr.get('quality_score', 0)}%")
        print(f"  Status: {qr.get('overall_status', 'N/A')}")
        print(f"  {qr.get('summary', '')}")

    print(f"\nFHIR Bundle: {'YES' if result.fhir_bundle else 'NO'}")
    if result.fhir_bundle:
        print(f"  Entries: {len(result.fhir_bundle.get('entry', []))}")

    print(f"\nAgent Execution Log ({len(result.pipeline_metadata)} agents):")
    for m in result.pipeline_metadata:
        icon = "[OK]" if m.success else "[!!]"
        print(f"  {icon} {m.agent_name}: {m.processing_time_ms:.0f}ms | {m.model_used}")

    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
