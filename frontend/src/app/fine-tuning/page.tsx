"use client";

import { Sidebar } from "@/components/Sidebar";

const SAMPLE_TRANSCRIPT =
  "45-year-old male, new onset T2DM, A1c 9.1%. BMI 34. Fasting glucose 210. Reports polydipsia, polyuria x3 weeks. Family history of diabetes (mother, sister). No prior medications.";

const BASE_OUTPUT = `The patient is a 45 year old male who presents with new onset diabetes. His A1c is 9.1% which is elevated. BMI is 34 indicating obesity. He has symptoms of polydipsia and polyuria for 3 weeks.

I would recommend starting metformin and lifestyle modifications. Follow up in 3 months to recheck A1c. Consider referral to endocrinology if not improving.

He should also be screened for complications including eye exam and foot exam.`;

const FINETUNED_OUTPUT = `SUBJECTIVE: 45-year-old male presenting with new-onset polyuria and polydipsia x3 weeks. FHx: T2DM in mother and sister. No prior medications. Reports increased thirst and frequent urination.

OBJECTIVE: A1c 9.1% (elevated). Fasting glucose 210 mg/dL. BMI 34 (obese). Vitals otherwise stable.

ASSESSMENT: Type 2 diabetes mellitus, newly diagnosed with hyperglycemia. ICD-10: E11.65 - Type 2 diabetes mellitus with hyperglycemia.

PLAN:
1. Start metformin 500mg BID, titrate to 1000mg BID over 2 weeks.
2. Diabetic education: diet, exercise, glucose monitoring.
3. Ophthalmology referral for baseline diabetic eye exam.
4. Comprehensive metabolic panel, lipid panel, urine microalbumin.
5. Recheck A1c in 3 months. Target <7%.
6. Return precautions: DKA symptoms (nausea, vomiting, abdominal pain).`;

const METRICS = [
  {
    metric: "SOAP completeness (4/4 sections)",
    base: "6/10",
    tuned: "10/10",
    delta: "+67%",
  },
  {
    metric: "ICD-10 exact code match",
    base: "4/10",
    tuned: "9/10",
    delta: "+125%",
  },
  {
    metric: "Structured output consistency",
    base: "3/10",
    tuned: "10/10",
    delta: "+233%",
  },
  {
    metric: "Drug name extraction",
    base: "5/10",
    tuned: "9/10",
    delta: "+80%",
  },
];

export default function FineTuning() {
  return (
    <div className="h-screen w-full flex font-sans relative overflow-hidden text-text-main font-semibold bg-transparent">
      <div className="w-full h-full glass-panel flex overflow-hidden border-none rounded-none shadow-none">
        <Sidebar activeItem="Fine-Tuning" />

        <div className="flex-1 flex flex-col h-full p-4 lg:p-6 lg:pl-0 pb-0 lg:pb-0 overflow-y-auto custom-scrollbar relative shadow-[-10px_0_30px_rgba(0,0,0,0.02)]">
          <div className="max-w-6xl mx-auto w-full pt-12 pb-24 px-4 lg:px-8">
            <h1 className="text-4xl font-black text-text-main tracking-tight mb-2">
              Fine-Tuning Impact
            </h1>
            <p className="text-sm font-medium text-text-muted mb-10">
              LoRA (r=16) fine-tuned on 54 synthetic SOAP note pairs | Adapter:{" "}
              <a
                href="https://huggingface.co/steeltroops-ai/medgemma-4b-soap-lora"
                target="_blank"
                rel="noreferrer"
                className="text-accent-blue hover:underline"
              >
                steeltroops-ai/medgemma-4b-soap-lora
              </a>
            </p>

            <div className="space-y-10">
              {/* Side-by-side comparison */}
              <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Base model */}
                <div className="glass-card rounded-3xl p-8 border border-red-200/60 shadow-xl">
                  <div className="flex items-center gap-3 mb-5">
                    <span className="w-3 h-3 rounded-full bg-red-400 animate-pulse" />
                    <h2 className="text-lg font-black text-red-500 tracking-tight">
                      Base MedGemma 4B
                    </h2>
                  </div>
                  <div className="bg-white/50 rounded-2xl p-5 border border-white/60 mb-4">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-widest mb-2">
                      Input Transcript
                    </p>
                    <p className="text-sm font-medium text-text-main/80 leading-relaxed italic">
                      {SAMPLE_TRANSCRIPT}
                    </p>
                  </div>
                  <div className="bg-red-50/50 rounded-2xl p-5 border border-red-100/60">
                    <p className="text-xs font-bold text-red-400 uppercase tracking-widest mb-2">
                      Output (unstructured)
                    </p>
                    <pre className="text-sm font-medium text-text-main/80 leading-relaxed whitespace-pre-wrap font-sans">
                      {BASE_OUTPUT}
                    </pre>
                  </div>
                  <div className="mt-4 flex gap-2 flex-wrap">
                    <span className="text-xs bg-red-100 text-red-600 px-3 py-1 rounded-full font-bold">
                      No SOAP structure
                    </span>
                    <span className="text-xs bg-red-100 text-red-600 px-3 py-1 rounded-full font-bold">
                      No ICD-10 code
                    </span>
                    <span className="text-xs bg-red-100 text-red-600 px-3 py-1 rounded-full font-bold">
                      Free-text narrative
                    </span>
                  </div>
                </div>

                {/* Fine-tuned model */}
                <div className="glass-card rounded-3xl p-8 border border-green-200/60 shadow-xl">
                  <div className="flex items-center gap-3 mb-5">
                    <span className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
                    <h2 className="text-lg font-black text-green-600 tracking-tight">
                      MedScribe (Fine-tuned)
                    </h2>
                  </div>
                  <div className="bg-white/50 rounded-2xl p-5 border border-white/60 mb-4">
                    <p className="text-xs font-bold text-text-muted uppercase tracking-widest mb-2">
                      Same Input Transcript
                    </p>
                    <p className="text-sm font-medium text-text-main/80 leading-relaxed italic">
                      {SAMPLE_TRANSCRIPT}
                    </p>
                  </div>
                  <div className="bg-green-50/50 rounded-2xl p-5 border border-green-100/60">
                    <p className="text-xs font-bold text-green-500 uppercase tracking-widest mb-2">
                      Output (structured SOAP)
                    </p>
                    <pre className="text-sm font-medium text-text-main/80 leading-relaxed whitespace-pre-wrap font-sans">
                      {FINETUNED_OUTPUT}
                    </pre>
                  </div>
                  <div className="mt-4 flex gap-2 flex-wrap">
                    <span className="text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full font-bold">
                      4/4 SOAP sections
                    </span>
                    <span className="text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full font-bold">
                      ICD-10: E11.65
                    </span>
                    <span className="text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full font-bold">
                      EHR-ready format
                    </span>
                  </div>
                </div>
              </section>

              {/* Metrics table */}
              <section className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl">
                <h2 className="text-xl font-black text-accent-blue tracking-tight mb-6">
                  Evaluation Metrics (10 held-out test cases)
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/30">
                        <th className="text-left py-3 px-4 font-bold text-text-muted uppercase tracking-wider text-xs">
                          Metric
                        </th>
                        <th className="text-center py-3 px-4 font-bold text-red-400 uppercase tracking-wider text-xs">
                          Base MedGemma 4B
                        </th>
                        <th className="text-center py-3 px-4 font-bold text-green-500 uppercase tracking-wider text-xs">
                          Fine-tuned
                        </th>
                        <th className="text-center py-3 px-4 font-bold text-accent-blue uppercase tracking-wider text-xs">
                          Delta
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {METRICS.map((row, i) => (
                        <tr
                          key={i}
                          className="border-b border-white/20 hover:bg-white/30 transition-colors"
                        >
                          <td className="py-3 px-4 font-semibold text-text-main">
                            {row.metric}
                          </td>
                          <td className="py-3 px-4 text-center font-bold text-red-500">
                            {row.base}
                          </td>
                          <td className="py-3 px-4 text-center font-bold text-green-600">
                            {row.tuned}
                          </td>
                          <td className="py-3 px-4 text-center font-black text-accent-blue">
                            {row.delta}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Technical details */}
              <section className="glass-card rounded-3xl p-8 lg:p-10 border border-white/40 shadow-xl">
                <h2 className="text-xl font-black text-accent-purple tracking-tight mb-6">
                  Fine-Tuning Configuration
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    ["Base Model", "MedGemma 4B IT"],
                    ["Method", "LoRA (r=16, a=32)"],
                    ["Target Modules", "q/k/v/o_proj"],
                    ["Trainable Params", "~0.5%"],
                    ["Training Data", "54 SOAP pairs"],
                    ["Epochs", "3"],
                    ["Learning Rate", "2e-4"],
                    ["Precision", "BF16 + NF4"],
                  ].map(([label, value], i) => (
                    <div
                      key={i}
                      className="bg-white/40 rounded-2xl p-4 border border-white/60 text-center"
                    >
                      <p className="text-[10px] font-bold text-text-muted uppercase tracking-widest mb-1">
                        {label}
                      </p>
                      <p className="text-sm font-black text-text-main">
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
