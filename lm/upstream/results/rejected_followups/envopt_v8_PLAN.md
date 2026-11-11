# EnvOpt v8 – vorregistrierter Plan (local laptop, RTX 4060, inference only)

**Modell:** GoldWormV8 `~/own_coder/v8/full1/sft_best.pt` (d768/L12/H12, 85M, byte256, ctx 2048), geladen strikt wie `coder_harness/v3/gen.py`.
Kein Training. Modell und Harness-Logik unverändert; optimiert wird nur die Umgebung E (envlib.py, identisch zur Box-Version).
Box-Referenz (v6-Code-Experte, REJECT) liegt in box_v6_ref/.

## Umgebung E
- Kontext: Retrieval-Spickzettel aus TRAIN-Beispielen `v8/teacher_v4/train_data.json` (324 Aufgaben), Pool = Train ohne Opt-Aufgaben,
  dekontaminiert gegen sichtbare Test-Prompts (kein gleicher Funktionsname, kein gemeinsames 13-Byte-Gramm im Docstring).
  Budget = min(900, 2048 − len(prompt) − 160) Bytes. Hinweis: teacher_v4-Prompts haben „# Goal:“-Kommentare statt Docstrings,
  char3-Retrieval nutzt daher v. a. Parameternamen; words-Retrieval Funktionsnamen.
- Validatoren nur aus dem sichtbaren Prompt (Syntax, Typ, nicht-None, ≤ Hälfte Exceptions auf 8 Fuzz-Eingaben), optional Konsens (CodeT-lite).
- Harness-Logik fix: Ranking nach niedrigster Completion-BPC, Index als Tie-Break.

## Splits
- Opt: 30 teacher_v4-Train-Aufgaben (sortierte IDs, jede 11.), ohne Tests.
- Val: distill/dev_tasks.json (40, mit Tests) – nur Auswahl.
- Test: eval1 (sha256 3792a731…) + eval2 (sha256 60c20524…), 60 Aufgaben; versteckte Tests nur im Final-Lauf.

## Optimierung
- Stufe A (Opt): 10 Kontexte (E0; shots∈{1,2}×retr∈{char3,words}×helper∈{0,1}; helper-only), Score = Anteil Samples mit bestandenen sichtbaren Validatoren. Top 3 + E0 weiter.
- Stufe B (Val): Kontext × 4 Selektoren; E* = max harness-p1, dann raw p1, Gültigkeit, Einfachheit.

## Final-Protokoll (eingefroren)
k=16, T=0.8, top-k 40, max 160 neue Bytes (v8/full_eval.sample), Seed 20261107+Task-Index (gleich für E0 und E*),
Sandbox goldworm_agent.sandbox (python -I, 5 s, 1 GiB). Metriken: raw p1, p16, harness-p1, Gültigkeit.
Gepaarter Bootstrap über Aufgaben, B=10000, Seed 777, Perzentile 2,5/97,5.

## Gates
- **G0:** Leckage-Audit besteht (0 Namens-/13-Gramm-Überlapp Pool↔Test, 0 Referenzkörper im Pool, alle Test-Präfixblöcke aus dem Pool, Opt/Val ohne Test-Namen).
- **G1:** harness-p1(E*) − harness-p1(E0) > 0 mit CI-Untergrenze > 0.
- **G2:** ΔGültigkeit (Punktschätzer) ≥ −2 pp.
- Urteil: nur G0 ∧ G1 ∧ G2 → „EnvOpt bringt etwas auf v8“, sonst verworfen. Kein Push/Publish.
