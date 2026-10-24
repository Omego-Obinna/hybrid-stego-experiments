# Environments

Use separate environments. Installing every study's dependencies into one
environment is neither required nor an exact reproduction of the recorded runs.

| Activity | Requirements |
|---|---|
| Release integrity/accounting check | Python >=3.10, standard library only |
| Saved HSMNet score/uncertainty check | NumPy; `requirements/analysis.txt` |
| Image protocol, fusion and fixed-observer studies | NumPy, SciPy, Pillow, scikit-image, scikit-learn, pandas, native `pystcstego`; inspect each configuration |
| HSMNet/component CNN training | Compatible PyTorch/CUDA, torchvision where imported, tqdm, psutil, OpenCV for Yedroudj; native GPU configuration must be recorded |
| Text naturalness | transformers/datasets/torch, MAUVE, selected model weights and local language-model service where used; component requirements retained beside scripts |
| Pulsar | Pinned upstream code, diffusers/generator weights and its dependency set |
| Assemblage | Its pinned native code, SageMath and submodule dependencies, separate from the PyTorch environment |
| Manuscript | pdfLaTeX, BibTeX, latexmk and standard TeX Live packages; Overleaf includes the common packages |

One recorded protocol/linkage stack has Python 3.12.14, NumPy 2.4.5, SciPy
1.17.1, Pillow 12.2.0, scikit-image 0.26.0, scikit-learn 1.8.0, pandas 3.0.3
and pystcstego 0.1.0. This is a record, not a guarantee that every platform
can install that exact stack. The per-run `run_config.json` and
`results/audit/recorded_parameters.json` have additional recorded settings.

`requirements/analysis.txt` is deliberately small and sufficient for the saved
statistics check. No claim of a fully locked original GPU/native environment
is made. No software installation is performed by repository verification.
