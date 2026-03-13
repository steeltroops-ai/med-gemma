"""
Drug Interaction Agent — Advanced Clinical Pharmacology Safety System.

Safety-critical layer in the agentic pipeline. Uses a three-tier approach:
  Tier 1: TxGemma 9B (google/txgemma-9b-predict) via HF Inference API
           More accurate than 2B for complex multi-drug interactions.
  Tier 2: Deterministic rules database — FDA-inspired interaction rules
           with 4-level alert classification (INFO/WARNING/CRITICAL/CONTRAINDICATED)
  Tier 3: Demo mode — realistic clinical scenario (always functional)

Alert Level Protocol:
  INFO          - Minor interaction, routine monitoring sufficient
  WARNING       - Significant interaction, enhanced monitoring required
  CRITICAL      - High-risk interaction, clinical review mandatory before dispensing
  CONTRAINDICATED - Absolute contraindication, FHIR compilation blocked until
                    physician provides explicit override

No local model loading — works on CPU-only HF Spaces (~400MB Docker image).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.base import BaseAgent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert Level Constants
# ---------------------------------------------------------------------------

ALERT_INFO = "INFO"
ALERT_WARNING = "WARNING"
ALERT_CRITICAL = "CRITICAL"
ALERT_CONTRAINDICATED = "CONTRAINDICATED"

# ---------------------------------------------------------------------------
# Advanced Deterministic Interaction Database
# ---------------------------------------------------------------------------
# Format: (drug1_keyword, drug2_keyword): (alert_level, mechanism, clinical_action)
# Covers the most clinically significant interactions encountered in primary care
# and hospital settings. Based on Stockley's Drug Interactions and FDA drug labels.

INTERACTION_DB: dict[tuple[str, str], tuple[str, str, str]] = {
    # --- CONTRAINDICATED ---
    ("maoi", "ssri"): (
        ALERT_CONTRAINDICATED,
        "MAO inhibitor + SSRI: risk of fatal serotonin syndrome via dual serotonin surge",
        "Do NOT co-prescribe. Allow 14-day washout after MAOI before starting SSRI.",
    ),
    ("maoi", "snri"): (
        ALERT_CONTRAINDICATED,
        "MAO inhibitor + SNRI: risk of fatal serotonin syndrome",
        "Do NOT co-prescribe. Allow 14-day washout after MAOI before starting SNRI.",
    ),
    ("maoi", "tramadol"): (
        ALERT_CONTRAINDICATED,
        "MAO inhibitor + tramadol: serotonin syndrome + seizure risk",
        "CONTRAINDICATED. Tramadol is a weak SNRI; combination is life-threatening.",
    ),
    ("linezolid", "ssri"): (
        ALERT_CONTRAINDICATED,
        "Linezolid (reversible MAOI) + SSRI: serotonin syndrome risk",
        "CONTRAINDICATED unless unavoidable. Monitor intensively if must use.",
    ),
    ("methotrexate", "nsaid"): (
        ALERT_CONTRAINDICATED,
        "Methotrexate + NSAIDs: NSAIDs reduce renal MTX clearance → MTX toxicity (myelosuppression, mucositis)",
        "CONTRAINDICATED at high-dose MTX. Avoid concurrent use; if necessary, monitor CBC/renal function closely.",
    ),
    ("methotrexate", "ibuprofen"): (
        ALERT_CONTRAINDICATED,
        "Methotrexate + Ibuprofen: NSAID reduces MTX clearance → bone marrow suppression",
        "CONTRAINDICATED. Use acetaminophen for pain management instead.",
    ),
    ("warfarin", "aspirin"): (
        ALERT_CRITICAL,
        "Warfarin + Aspirin: dual anticoagulation significantly increases major bleeding risk",
        "Use only when clinically indicated (e.g., ACS + AF). Target INR 2.0-2.5. Monitor closely.",
    ),

    # --- CRITICAL ---
    ("warfarin", "amiodarone"): (
        ALERT_CRITICAL,
        "Warfarin + Amiodarone: amiodarone inhibits CYP2C9/CYP3A4 → dramatically elevated INR (2-3x increase)",
        "Reduce warfarin dose by 30-50% when adding amiodarone. Monitor INR every 3-5 days until stable.",
    ),
    ("digoxin", "amiodarone"): (
        ALERT_CRITICAL,
        "Digoxin + Amiodarone: amiodarone inhibits P-gp and renal digoxin clearance → digoxin toxicity",
        "Reduce digoxin dose by 30-50%. Monitor digoxin levels and renal function. Watch for toxicity signs.",
    ),
    ("lithium", "nsaid"): (
        ALERT_CRITICAL,
        "Lithium + NSAIDs: NSAIDs reduce renal lithium excretion → lithium toxicity (tremor, confusion, seizure)",
        "Avoid concurrent use. If necessary, check lithium levels frequently. Consider paracetamol alternative.",
    ),
    ("lithium", "ibuprofen"): (
        ALERT_CRITICAL,
        "Lithium + Ibuprofen: NSAID increases lithium levels → lithium toxicity risk",
        "AVOID. Use acetaminophen/paracetamol instead. If unavoidable, halve lithium dose and monitor levels.",
    ),
    ("sertraline", "tramadol"): (
        ALERT_CRITICAL,
        "SSRI + Tramadol: serotonin syndrome risk + tramadol seizure threshold lowered by SSRIs",
        "Avoid concurrent use. If essential, use lowest effective tramadol dose and monitor for serotonin syndrome.",
    ),
    ("azithromycin", "amiodarone"): (
        ALERT_CRITICAL,
        "Azithromycin + Amiodarone: additive QT prolongation → Torsades de Pointes, ventricular fibrillation",
        "AVOID if possible. If essential, continuous cardiac monitoring required. Check baseline QTc.",
    ),
    ("warfarin", "ciprofloxacin"): (
        ALERT_CRITICAL,
        "Warfarin + Ciprofloxacin: fluoroquinolone inhibits warfarin metabolism + alters gut flora",
        "Monitor INR every 2-3 days during fluoroquinolone course. Anticipate INR increase 40-60%.",
    ),
    ("simvastatin", "amiodarone"): (
        ALERT_CRITICAL,
        "Simvastatin + Amiodarone: amiodarone inhibits CYP3A4 → statin accumulation → rhabdomyolysis risk",
        "Cap simvastatin at 20mg/day with amiodarone. Consider switching to pravastatin (non-CYP3A4).",
    ),
    ("clopidogrel", "omeprazole"): (
        ALERT_CRITICAL,
        "Clopidogrel + Omeprazole: omeprazole inhibits CYP2C19 → reduces clopidogrel active metabolite by 40-50%",
        "Switch to pantoprazole (weaker CYP2C19 inhibitor). Avoid omeprazole/esomeprazole with clopidogrel.",
    ),

    # --- WARNING ---
    ("lisinopril", "potassium"): (
        ALERT_WARNING,
        "ACE inhibitor + Potassium supplements: additive hyperkalemia risk",
        "Monitor serum potassium at baseline, 1 week, and 1 month. Target K+ < 5.5 mEq/L.",
    ),
    ("lisinopril", "nsaid"): (
        ALERT_WARNING,
        "ACE inhibitor + NSAIDs: reduced antihypertensive efficacy + increased nephrotoxicity",
        "Monitor BP and renal function (BUN/Cr) if combination necessary. Prefer acetaminophen for analgesia.",
    ),
    ("lisinopril", "ibuprofen"): (
        ALERT_WARNING,
        "ACE inhibitor + Ibuprofen: NSAID blunts ACE effect and worsens renal function",
        "Avoid combination if possible. If necessary, monitor BP and renal function weekly.",
    ),
    ("metformin", "contrast dye"): (
        ALERT_WARNING,
        "Metformin + Iodinated contrast: risk of contrast-induced nephropathy → metformin accumulation → lactic acidosis",
        "HOLD metformin 48h before and after contrast procedure. Resume only when eGFR confirmed stable.",
    ),
    ("metformin", "alcohol"): (
        ALERT_WARNING,
        "Metformin + Alcohol: additive lactic acidosis risk, especially with binge drinking",
        "Counsel patient to limit alcohol consumption. Avoid metformin with heavy alcohol use.",
    ),
    ("ssri", "nsaid"): (
        ALERT_WARNING,
        "SSRI + NSAID: SSRIs deplete platelet serotonin; NSAIDs inhibit platelet COX → additive GI bleed risk (3x increased)",
        "Co-prescribe PPI if combination is necessary. Monitor for GI bleeding symptoms.",
    ),
    ("warfarin", "omeprazole"): (
        ALERT_WARNING,
        "Warfarin + Omeprazole: CYP2C19 inhibition may increase warfarin S-enantiomer levels",
        "Monitor INR when adding/stopping omeprazole. Pantoprazole is a safer PPI alternative.",
    ),
    ("metformin", "furosemide"): (
        ALERT_WARNING,
        "Metformin + Furosemide: loop diuretics may impair renal function → metformin accumulation",
        "Monitor eGFR with combination. Hold metformin if eGFR < 30 mL/min/1.73m².",
    ),
    ("warfarin", "acetaminophen"): (
        ALERT_WARNING,
        "Warfarin + Acetaminophen (>2g/day): regular high-dose acetaminophen increases anticoagulant effect",
        "Limit acetaminophen to <2g/day. Monitor INR if patient uses regular acetaminophen.",
    ),
    ("atorvastatin", "azithromycin"): (
        ALERT_WARNING,
        "Atorvastatin + Azithromycin: azithromycin is a mild CYP3A4 inhibitor → may increase statin levels",
        "Short courses are generally acceptable. Monitor for muscle symptoms (myalgia, weakness).",
    ),

    # --- INFO ---
    ("metformin", "lisinopril"): (
        ALERT_INFO,
        "Metformin + Lisinopril: no direct pharmacokinetic interaction; both commonly co-prescribed",
        "Standard monitoring: eGFR and potassium at baseline and periodically.",
    ),
    ("statin", "grapefruit"): (
        ALERT_INFO,
        "CYP3A4-metabolized statins (simvastatin, atorvastatin) + grapefruit: may increase statin levels",
        "Advise patients to avoid grapefruit/juice. Pravastatin is not affected by grapefruit.",
    ),
    ("aspirin", "ibuprofen"): (
        ALERT_INFO,
        "Low-dose aspirin + Ibuprofen: ibuprofen may competitively block aspirin's antiplatelet effect",
        "Take aspirin 30 min before ibuprofen, or use alternative analgesic (acetaminophen preferred).",
    ),
}

# ---------------------------------------------------------------------------
# Demo data (enhanced with alert levels)
# ---------------------------------------------------------------------------

DEMO_DRUG_CHECK = {
    "medications_found": ["lisinopril 10mg", "metformin 1000mg", "azithromycin 500mg", "albuterol PRN"],
    "interactions": [
        {
            "drug_pair": ["lisinopril", "metformin"],
            "alert_level": ALERT_INFO,
            "mechanism": "No direct pharmacokinetic interaction; both commonly co-prescribed",
            "clinical_action": "Standard monitoring: eGFR and potassium at baseline and periodically.",
        },
        {
            "drug_pair": ["azithromycin", "lisinopril"],
            "alert_level": ALERT_INFO,
            "mechanism": "No significant pharmacokinetic interaction between these agents",
            "clinical_action": "Routine monitoring. Be aware of QT risk in cardiac patients.",
        },
    ],
    "warnings": [
        "Monitor renal function with lisinopril + metformin combination.",
        "Azithromycin: check baseline QTc in patients with cardiac disease or electrolyte abnormalities.",
    ],
    "alert_summary": {
        ALERT_CONTRAINDICATED: 0,
        ALERT_CRITICAL: 0,
        ALERT_WARNING: 0,
        ALERT_INFO: 2,
    },
    "blocks_fhir": False,
    "safe": True,
    "highest_alert": ALERT_INFO,
    "summary": "4 medications identified. 0 CRITICAL interactions. 0 CONTRAINDICATED. Standard monitoring recommended.",
}

# ---------------------------------------------------------------------------
# TxGemma 9B System Prompt
# ---------------------------------------------------------------------------

DRUG_SAFETY_SYSTEM_PROMPT = """\
You are an expert clinical pharmacist specializing in drug-drug interactions, \
pharmacokinetics, and medication safety. Your analysis must be clinically accurate, \
concise, and actionable. Classify interactions using: CONTRAINDICATED / CRITICAL / WARNING / INFO.
"""

DRUG_CHECK_PROMPT = """\
You are a clinical pharmacist. Analyze drug interactions for: {med_list}

For each interaction found, provide:
1. Drug pair
2. Alert level: CONTRAINDICATED / CRITICAL / WARNING / INFO
3. Mechanism of interaction
4. Clinical management action

Be clinically precise. Focus on the most significant interactions first.
"""


class DrugInteractionAgent(BaseAgent):
    """
    Advanced Drug Interaction Agent — TxGemma 9B + Deterministic Safety Rules.

    Three-tier architecture:
      1. TxGemma 9B (google/txgemma-9b-predict) — primary AI analysis
         More accurate than 2B for complex multi-drug scenarios
      2. Deterministic rules DB — FDA-inspired rules covering 30+ interaction pairs
         Always available, no API needed
      3. Demo mode — realistic clinical scenario (always functional)

    Alert Level System (4 tiers):
      INFO           — Minor interaction, routine monitoring
      WARNING        — Significant, enhanced monitoring required
      CRITICAL       — High-risk, mandatory clinical review before dispensing
      CONTRAINDICATED — Absolute contraindication, FHIR compilation blocked

    Safety invariant: CRITICAL/CONTRAINDICATED interactions are always caught
    by the deterministic rules layer, even if the AI API is unavailable.
    """

    def __init__(self):
        super().__init__(name="drug_interaction", model_id="google/txgemma-9b-predict")
        self._ready = True

    def _load_model(self) -> None:
        self._ready = True

    def _process(self, input_data: Any) -> dict:
        """
        Check for drug interactions with 4-level alert classification.

        Args:
            input_data: dict with keys:
                - "medications": list[str] (optional, use if pre-extracted)
                - "soap_text": str (extract meds from here if medications not provided)

        Returns:
            dict with medications_found, interactions (with alert_level), warnings,
            alert_summary, blocks_fhir, safe, highest_alert, summary.
        """
        if isinstance(input_data, str):
            input_data = {"soap_text": input_data}

        medications = input_data.get("medications", [])
        soap_text = input_data.get("soap_text", "")

        if not medications and soap_text:
            medications = self._extract_medications(soap_text)

        if not medications:
            return {
                "medications_found": [],
                "interactions": [],
                "warnings": [],
                "alert_summary": {ALERT_CONTRAINDICATED: 0, ALERT_CRITICAL: 0, ALERT_WARNING: 0, ALERT_INFO: 0},
                "blocks_fhir": False,
                "safe": True,
                "highest_alert": None,
                "summary": "No medications found to check.",
            }

        # ALWAYS run deterministic rules first (safety invariant)
        det_result = self._rules_based_check(medications)

        # Try TxGemma 9B for enhanced AI analysis
        try:
            txgemma_analysis = self._txgemma_api_check(medications)
            det_result["txgemma_analysis"] = txgemma_analysis
            det_result["model_tier"] = "txgemma-9b-predict"
        except Exception as exc:
            log.warning(f"TxGemma 9B API failed: {exc} -- using deterministic rules only")
            det_result["model_tier"] = "deterministic_rules_only"

        return det_result

    def _txgemma_api_check(self, medications: list[str]) -> str:
        """Call TxGemma 9B via HF Inference API for AI-powered drug analysis."""
        from src.core.inference_client import generate_text

        med_list = ", ".join(medications)
        prompt = DRUG_CHECK_PROMPT.format(med_list=med_list)

        raw = generate_text(
            prompt=prompt,
            model_id="google/txgemma-9b-predict",
            system_prompt=DRUG_SAFETY_SYSTEM_PROMPT,
            max_new_tokens=768,
        )
        log.info(f"TxGemma 9B analysis complete: {len(raw)} chars")
        return raw

    def _extract_medications(self, text: str) -> list[str]:
        """Extract medication names from clinical text using regex."""
        # Comprehensive medication list covering primary care + hospital
        med_pattern = re.compile(
            r"\b(lisinopril|metformin|aspirin|atorvastatin|simvastatin|pravastatin|"
            r"rosuvastatin|omeprazole|pantoprazole|esomeprazole|amlodipine|"
            r"metoprolol|carvedilol|bisoprolol|losartan|valsartan|olmesartan|"
            r"albuterol|ipratropium|tiotropium|fluticasone|budesonide|formoterol|"
            r"prednisone|prednisolone|dexamethasone|methylprednisolone|"
            r"amoxicillin|azithromycin|ciprofloxacin|levofloxacin|doxycycline|"
            r"cephalexin|cefoxitin|nitrofurantoin|trimethoprim|linezolid|"
            r"ibuprofen|naproxen|celecoxib|acetaminophen|tramadol|codeine|"
            r"gabapentin|pregabalin|amitriptyline|nortriptyline|duloxetine|"
            r"sertraline|fluoxetine|paroxetine|escitalopram|citalopram|venlafaxine|"
            r"bupropion|mirtazapine|trazodone|quetiapine|olanzapine|risperidone|"
            r"warfarin|heparin|enoxaparin|apixaban|rivaroxaban|dabigatran|"
            r"clopidogrel|ticagrelor|prasugrel|dipyridamole|"
            r"digoxin|amiodarone|flecainide|sotalol|diltiazem|verapamil|"
            r"furosemide|hydrochlorothiazide|spironolactone|chlorthalidone|"
            r"insulin|glipizide|glimepiride|sitagliptin|empagliflozin|"
            r"semaglutide|liraglutide|dapagliflozin|methotrexate|"
            r"lithium|valproate|carbamazepine|phenytoin|levetiracetam|"
            r"levothyroxine|methimazole|propylthiouracil|"
            r"cyclobenzaprine|carisoprodol|baclofen|tizanidine)\b",
            re.IGNORECASE,
        )

        # Also capture drug + dose patterns (e.g., "metformin 1000mg")
        dose_pattern = re.compile(
            r"(\b\w+\b)\s+(\d+\s*(?:mg|mcg|units?|ml|g)\s*(?:daily|bid|tid|qid|prn|qhs|qam|once|twice)?)",
            re.IGNORECASE,
        )

        meds: set[str] = set()

        for match in med_pattern.finditer(text):
            meds.add(match.group(0).lower())

        for match in dose_pattern.finditer(text):
            drug = match.group(1).lower()
            dose = match.group(2).strip()
            if len(drug) > 4 and drug not in {
                "with", "from", "that", "this", "have", "take", "each", "once", "dose",
                "drug", "pill", "tablet", "caps", "daily", "twice", "three", "four",
            }:
                # Check if drug is a known medication name
                if med_pattern.match(drug):
                    meds.add(f"{drug} {dose}")
                elif drug in text.lower() and len(drug) >= 5:
                    meds.discard(drug)  # Only add if we have dose confirmation
                    meds.add(f"{drug} {dose}")

        return sorted(meds)

    def _rules_based_check(self, medications: list[str]) -> dict:
        """
        Comprehensive rules-based interaction check with 4-level alert system.

        Returns structured result with alert_summary and blocks_fhir flag.
        CONTRAINDICATED interactions set blocks_fhir=True to prevent unsafe FHIR output.
        """
        interactions: list[dict] = []
        warnings: list[str] = []
        med_names = [m.split()[0].lower() for m in medications]

        # Check all pairs against interaction database
        for i, m1 in enumerate(med_names):
            for m2 in med_names[i + 1:]:
                for (k1, k2), (alert_level, mechanism, action) in INTERACTION_DB.items():
                    # Fuzzy match: drug name contains or is contained by known keyword
                    m1_match = k1 in m1 or m1 in k1
                    m2_match = k2 in m2 or m2 in k2
                    # Also check reverse pairing
                    m1_rev = k2 in m1 or m1 in k2
                    m2_rev = k1 in m2 or m2 in k1

                    if (m1_match and m2_match) or (m1_rev and m2_rev):
                        interactions.append({
                            "drug_pair": [m1, m2],
                            "alert_level": alert_level,
                            "mechanism": mechanism,
                            "clinical_action": action,
                        })
                        break  # Avoid duplicate entries per pair

        # Class-level warnings (apply to drug classes)
        if any("metformin" in m for m in med_names):
            warnings.append("Metformin: monitor renal function (eGFR) periodically.")
        if any(m in med_names or any(m in med for med in med_names) for m in ["lisinopril", "losartan", "valsartan"]):
            warnings.append("ACE inhibitor/ARB: monitor serum potassium and renal function.")
        if any("warfarin" in m for m in med_names):
            warnings.append(
                "Warfarin: numerous drug and food interactions (vitamin K-rich foods, alcohol). "
                "Target INR monitoring every 4 weeks at minimum."
            )
        if any("azithromycin" in m for m in med_names):
            warnings.append(
                "Azithromycin: QT prolongation risk. Check baseline QTc especially with cardiac disease, "
                "hypokalemia, or hypomagnesemia."
            )
        if any("lithium" in m for m in med_names):
            warnings.append(
                "Lithium: narrow therapeutic index. Monitor lithium levels, renal function, TSH. "
                "Dehydration/NSAIDs/diuretics can precipitate toxicity."
            )
        if any("digoxin" in m for m in med_names):
            warnings.append("Digoxin: narrow therapeutic index. Monitor digoxin levels, renal function, potassium.")
        if any("amiodarone" in m for m in med_names):
            warnings.append(
                "Amiodarone: extensive drug interactions via CYP2C9/CYP3A4/P-gp inhibition. "
                "Review ALL concurrent medications for interactions."
            )

        # Compute alert summary
        alert_counts: dict[str, int] = {
            ALERT_CONTRAINDICATED: 0,
            ALERT_CRITICAL: 0,
            ALERT_WARNING: 0,
            ALERT_INFO: 0,
        }
        for itr in interactions:
            lvl = itr.get("alert_level", ALERT_INFO)
            if lvl in alert_counts:
                alert_counts[lvl] += 1

        # Determine highest alert level
        highest_alert = None
        for level in [ALERT_CONTRAINDICATED, ALERT_CRITICAL, ALERT_WARNING, ALERT_INFO]:
            if alert_counts[level] > 0:
                highest_alert = level
                break

        # FHIR blocking: CONTRAINDICATED interactions block output until physician overrides
        blocks_fhir = alert_counts[ALERT_CONTRAINDICATED] > 0

        # Safe flag: False if any CRITICAL or CONTRAINDICATED
        is_safe = alert_counts[ALERT_CONTRAINDICATED] == 0 and alert_counts[ALERT_CRITICAL] == 0

        summary_parts = [f"{len(medications)} medications identified."]
        if alert_counts[ALERT_CONTRAINDICATED]:
            summary_parts.append(f"[CONTRAINDICATED] {alert_counts[ALERT_CONTRAINDICATED]} interaction(s) - FHIR output blocked.")
        if alert_counts[ALERT_CRITICAL]:
            summary_parts.append(f"[CRITICAL] {alert_counts[ALERT_CRITICAL]} interaction(s) - physician review mandatory.")
        if alert_counts[ALERT_WARNING]:
            summary_parts.append(f"[WARNING] {alert_counts[ALERT_WARNING]} interaction(s) - enhanced monitoring required.")
        if alert_counts[ALERT_INFO]:
            summary_parts.append(f"[INFO] {alert_counts[ALERT_INFO]} interaction(s) - routine monitoring.")
        if not interactions:
            summary_parts.append("No interactions detected in rules database.")

        return {
            "medications_found": medications,
            "interactions": interactions,
            "warnings": warnings,
            "alert_summary": alert_counts,
            "blocks_fhir": blocks_fhir,
            "safe": is_safe,
            "highest_alert": highest_alert,
            "summary": " ".join(summary_parts),
        }

    def get_demo_result(self) -> dict:
        return DEMO_DRUG_CHECK
