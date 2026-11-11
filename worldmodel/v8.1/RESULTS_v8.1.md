# RESULTS v8.1 — GreyBox-MPC (WM) vs. Baselines auf Held-out-Interventionen

**Outcome: NO_ADVANCE** (gültiger Lauf, kein INVALID, kein NOT_LAUNCHED)

Alle Zahlen stammen unverändert aus `run_output/` dieses einen Laufs. Nichts ist veröffentlicht.

## Lauf
- RELEASE_v8.1.json: FINAL, sha256 `e88400e3214375eed62d47bf396472f742755b0181d56930b0340af415211f84`
- Code-Manifest: `be232be6b42d534c7bfd8ee3f0d6e3c75bd1c1c3129f47ab88de8e174ff4d4e8`, CODE_PASS dahill `36aefe31…6928`
- Start: 2026-11-10 09:38 Europe/Berlin, CPU only (`CUDA_VISIBLE_DEVICES=""`, `/workspace/venv_wm/bin/python`).
  Ein erster Startversuch um 09:37 brach beim `import torch` ab (falscher Interpreter), bevor `run_output` existierte; kein Datenpunkt erzeugt, daher kein Rerun.
- Ende (Ledger `final`): 12:22:21 Europe/Berlin. Die Wanduhrzeit enthält eine Pause des Rechners (~09:57 bis ~11:56, aus `ps`-Elapsed geschlossen, nicht gemessen); maßgeblich ist CPU-Zeit.
- Budget: 1558.6 von 3600 CPU-s, `all_caps_respected: true`, keine Shrink-Leiter.

| Phase | CPU-s | Cap |
|---|---|---|
| fits | 305.4 | 800 |
| tuning | 465.9 | 1000 |
| test | 436.7 | 900 |
| in_distribution | 57.9 | 120 |
| pi_oracle | 10.4 | 30 |
| adaptation | 277.7 | 520 |
| bootstrap_stats | 3.5 | 60 |
| data / overhead / audit | 0.18 / 0.98 / 0.03 | 30 / 60 / 80 |

Validität (FINAL_STATUS): all_caps_respected, equal_search_effort, frozen_hashes_match_release, identical_inputs, no_heldout_pre_test, selection_sealed_before_test, stage0_model_imports — alle `true`. Fallback-Gate für alle Arme `true`.

## Comparator-Auswahl (Tuning-Seeds 7000–7019, versiegelt vor Test)
COMPARATOR_SELECTION.json sha256 `1c422d5b…47b5`. Gewählt (alle λ = 0.5):

| Arm | Struktur | mean_J | mean_E |
|---|---|---|---|
| RC_only | 2 | 89.97 | 88.90 |
| LIN | 3 | 90.51 | 89.08 |
| WM | 96 | 91.67 | 90.23 |
| GRU | 64 | 91.74 | 90.47 |

Bester Nicht-WM-Arm und damit Comparator: **RC_only**.

## Primärergebnis (versiegelt 09:56:53, PRIMARY_RESULT.json `5841fba4…42e9`)
S = 1 − ΣE_WM / ΣE_comp; dD = mittel(D_WM − D_comp) in K·h/Tag; CI 95 % Bootstrap B = 10000.

| Gate | Ergebnis | Wert |
|---|---|---|
| P1 (S ≥ 5 %) | **fail** | S = −1.91 %, CI [−2.18, −1.63] % |
| P2 (CI-Untergrenze) | **fail** | s. o. |
| P3 Komfort (CI-Obergrenze < 0.25) | pass | dD = −1.147, CI [−1.250, −1.042] |
| P4 Sicherheit | pass | Safety-Exits WM 0, Comparator 0 |

**Lesart:** WM verbraucht auf Held-out-Interventionen ~1.9 % *mehr* Energie als RC_only (Testsummen E: WM 33252.9 vs. RC_only 32630.8), hält dafür den Komfort deutlich besser (dD negativ). Die Hypothese „≥ 5 % Energieersparnis bei nicht schlechterem Komfort“ ist widerlegt, nicht nur unbestätigt: das gesamte CI von S liegt unter 0.

## Sekundärergebnisse (SECONDARY_RESULTS.json `ab131c5d…5ecd`, nicht gate-relevant)
- Sekundärfenster 120–191: S = −3.00 %, CI [−3.38, −2.66] %; dD = −1.150.
- In-Distribution: S = −1.49 %, CI [−1.94, −1.04] %; dD = −0.002, CI [−0.040, 0.034] (Komfort gleich).
- Adaptation (Held-out, 100 Updates):

| K | S | S-CI | dD |
|---|---|---|---|
| 1 | −1.95 % | [−2.46, −1.52] % | −1.37 |
| 3 | −1.86 % | [−2.39, −1.38] % | −1.23 |
| 10 | −1.54 % | [−2.03, −1.10] % | −1.21 |

  Mehr Adaptationsdaten verkleinern den Energienachteil leicht, kein K erreicht S > 0.
- Per Typ (S / dD): H1 −2.53 % / −2.14; H2 −1.63 % / −2.26; H3 −1.49 % / +0.011; H4 −1.91 % / −0.196.
- 12-Schritt-Held-out-MSE: WM 2.241, GRU 2.244, RC_only 2.620, LIN 2.837. WM ist der beste Prädiktor, aber praktisch gleichauf mit GRU — passt zu SPARTAN (arXiv 2411.06890): bessere Vorhersage übersetzt sich nicht automatisch in besseres Planen.
- Testsummen (E / J): WM 33252.9 / 35725.0; GRU 33173.7 / 35918.9; RC_only 32630.8 / 38647.1; LIN 31678.8 / 45861.8; PI 32837.7 / 44219.3; Oracle 32783.1 / 33691.8.
- Keine Safety-Exits in irgendeinem Arm (Tuning, Test, In-Dist, Adaptation).

## Einordnung (Interpretation, nicht gate-relevant)
WM liegt beim Gesamtziel J vor RC_only, tauscht aber Energie gegen Komfort. Da λ = 0.5 für alle Arme auf J gewählt wurde, die Hypothese aber Energie bei Komfort-Nichtunterlegenheit misst, bevorzugt die Zielfunktion Komfort stärker, als das Primärkriterium belohnt. Das ist eine Beobachtung für einen eventuellen v8.2-Plan, keine Nachjustierung dieses Laufs.

Bekannter Störfaktor (dahill, Code-Review): In den Trainingsdaten fallen Onset 120, Anfangszustand S1 und PI-Anregung zusammen — für alle Arme gleich.

## Dateien (sha256)
```
fd4b1d30f22e9a3bc208918dd0c98dbb51ad0533fcc74c13df51f7c0a0f4f061  BOOT_INDEX_adapt.npy
7741d5f947f57b2cb8725f113d1d9e019fa440f2c12d3906b214a0368d09e3ff  BOOT_INDEX_primary.npy
e062498995f5051aff1b8d3987c5856da5ac54c339a93b46b9a1ba47bed75b99  BOOT_INDEX_secondary.npy
1c422d5b5a243a5929efb3bb7f43b79a40bda4f7e09458340722fd37490747b5  COMPARATOR_SELECTION.json
7330b5a5761ae245383d17edbc4ecc8a8f5dc7d34f73735d76d899a64bef1c30  FINAL_STATUS.json
a1a6796acca17ae53752c8cb08e497c8243b030824b42ba87ef0345cb04b54a4  LEDGER.jsonl
5841fba4e61265862ef1e948974b925eca577cb6440aeb227756618fdfd142e9  PRIMARY_RESULT.json
ab131c5da784724ecd7a4ee13a1265fcd17ed31954f06a5c5d8be5e13a035ecd  SECONDARY_RESULTS.json
873435b170fd70220a68501d4d1484af5225a6f0d7aafed3d151b7a85e0ea3e5  fit_info.json
a8f5b344bf93b9856e701379f71d5a669cf4acc73c3f80e36cbe4b621c332ebf  normalization.json
d7dac6b8b7ac92f95fa2097df4c797ca441831f61cf69464d386c0cb265e514f  test_results.npz
864ac87a67addcc727d1b1f8a655431600c866b8f9badfe6d3c10f567fffdce4  tuning_rows.npz
```
