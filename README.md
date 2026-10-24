# Knee MRI abnormality detection: model archive (Kaggle RSNA 2026 benchmark)

This archives our own models for the **RSNA Knee Abnormality Detection** benchmark on Kaggle (October 2026). The task is to predict 12 knee findings per MRI study, scored by mean AUC.

It holds our two best entries built only from our own models, one per track:

| Entry | What it is | Public | Private |
|---|---|---|---|
| **Main track**: CNXS-v2, label v3 | ConvNeXt-S (timm `convnext_small.in12k_ft_in1k_384`), 2.5D 3-slice windows at 352 px, full field plus a 100 mm local view. Each of 4 fold models is fine-tuned for 3 epochs on label v3, with its own fold held out. | 0.947 | 0.937 |
| **Efficiency track**: FT-SF3 | Folds 0 and 1 of CNXS-v2 fine-tuned for 3 epochs on all training studies (label v3); fp16 serving on 2× T4. | 0.946 | 0.935 |

The scores are Kaggle's public and private leaderboard scores for these exact submissions.

## Layout

```
main-track/
  code/                     serving + inference package, exactly as submitted (kernel/, serving/, infer/, models/, pipeline/, trainer/)
  trainer_base_cnxs_v2/     training code for the base CNXS-v2 4-fold models
  v3cv_launch.sh, ft_export_box_v3cv.sh   label-v3 fine-tune launch/export scripts
efficiency-track/
  code/                     serving + inference package, exactly as submitted
  trainer_finetune_sf3/     fine-tune training code (FT-SF3)
```

`code/pins_cnxs_v2.json` pins the sha256 of every serving file and every weight file. The serving entry point is `code/kernel/kernel_cnxs_v2.py`.

## Weights

The weights are attached to the **v1.0 release** as tar archives:

| File | Size | sha256 |
|---|---|---|
| `main_cnxs_v2_label_v3_4fold.tar` | 806 MB | `6c940e21149c8541b8985fa5ad856261a9103b28e7f65aa8540377c04913fc0a` |
| `efficiency_cnxs_v2_ftsf3.tar` | 806 MB | `4b1dc2e8914c195ebf43647dc282107580b523e82da4d6a0caecf5951a3c2989` |

- **Main track:** `fold{0..3}/ema_ep02.pt` + `config.json`.
- **Efficiency track:** `fold0` and `fold1` are the fine-tuned models (`ema_ep02.pt`). `fold2` and `fold3` are the base CNXS-v2 models (`ema_ep14.pt`), packaged alongside them. The efficiency entry served folds 0 and 1.

## Training data and labels

- **Data:** the benchmark's training studies only. Neither entry used external datasets.
- **Labels:** derived from the radiology reports with LLM assistance.
- **Not included:** label files, fold lists and any image data. They derive from the benchmark data, which its rules do not allow us to redistribute.
- **Pretrained start:** timm ConvNeXt-S (in12k → in1k, 384), Apache-2.0.
- **Not shipped:** the knee anatomy anchor model. Serving falls back to image-centred local views (`anchors_mode="centre"`).

## Credits

- **Study-level DICOM read pools:** the design follows a public notebook by nartaa.
- **Effusion volume feature:** `serving/eff_feature.py` and `fit_stacker.py` implement the `eff_sag_p99_open` definition from heliosli's public notebook. Our code re-implements the definition only. The stacker was not adopted in these two entries.

## Licence

- **Code:** MIT (`LICENSE`).
- **Weights:** CC BY-NC 4.0 (`WEIGHTS_LICENSE`). The models were trained on RSNA data used under the benchmark's rules and the RSNA MIRA licence.

Author: Syed Asad Ali.
