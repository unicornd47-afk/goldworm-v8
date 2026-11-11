# v8 full1 Ergebnisse

Checkpoint: <own_coder>/v8/full1/sft_best.pt (Auswahl nur per Dev-CE: {'phase': 'sft', 'step': 809, 'dev_loss': 0.48391179191438777, 'train_loss_mean': 0.11254538188455626, 'phase_s': 60.32723708599224, 'presented_bytes': 908244, 'lr': 1e-05})

## eval2
Summary: {"p1": 0.004166666666666667, "p16": 0.06666666666666667, "valid": 0.9166666666666666, "hp1": 0.004166666666666667, "hp16": 0.06666666666666667, "hvalid": 0.93125, "solved_tasks": 2.0, "heal_solved_tasks": 2.0}
- vs gen0 p1: base 0.0000 new 0.0042 diff +0.0042 CI95 [+0.0000,+0.0104]
- vs gen0 p16: base 0.0000 new 0.0667 diff +0.0667 CI95 [+0.0000,+0.1667]
- vs gen0 valid: base 0.2729 new 0.9167 diff +0.6438 CI95 [+0.5687,+0.7083]
- vs v2 p1: base 0.0000 new 0.0042 diff +0.0042 CI95 [+0.0000,+0.0104]
- vs v2 p16: base 0.0000 new 0.0667 diff +0.0667 CI95 [+0.0000,+0.1667]
- vs v2 valid: base 0.5625 new 0.9167 diff +0.3542 CI95 [+0.2583,+0.4437]
- vs trial1 p1: base 0.0000 new 0.0042 diff +0.0042 CI95 [+0.0000,+0.0104]
- vs trial1 p16: base 0.0000 new 0.0667 diff +0.0667 CI95 [+0.0000,+0.1667]
- vs trial1 valid: base 0.8146 new 0.9167 diff +0.1021 CI95 [+0.0333,+0.1708]

## eval
Summary: {"p1": 0.07083333333333333, "p16": 0.16666666666666666, "valid": 0.9291666666666667, "hp1": 0.07083333333333333, "hp16": 0.16666666666666666, "hvalid": 0.93125, "solved_tasks": 5.0, "heal_solved_tasks": 5.0}
- vs gen0 p1: base 0.0000 new 0.0708 diff +0.0708 CI95 [+0.0063,+0.1542]
- vs gen0 p16: base 0.0000 new 0.1667 diff +0.1667 CI95 [+0.0333,+0.3000]
- vs gen0 valid: base 0.2354 new 0.9292 diff +0.6937 CI95 [+0.6125,+0.7562]
- vs v2 p1: base 0.0354 new 0.0708 diff +0.0354 CI95 [-0.0229,+0.1083]
- vs v2 p16: base 0.0667 new 0.1667 diff +0.1000 CI95 [+0.0000,+0.2000]
- vs v2 valid: base 0.7646 new 0.9292 diff +0.1646 CI95 [+0.0791,+0.2479]
- vs trial1 p1: base 0.0000 new 0.0708 diff +0.0708 CI95 [+0.0063,+0.1542]
- vs trial1 p16: base 0.0000 new 0.1667 diff +0.1667 CI95 [+0.0333,+0.3000]
- vs trial1 valid: base 0.8521 new 0.9292 diff +0.0771 CI95 [+0.0000,+0.1521]

Hinweis: teacher_v4 nutzt gelockerten Konzeptfilter; eval2 ist kein garantierter Test auf fremde Konzepte. Nichts gepusht.