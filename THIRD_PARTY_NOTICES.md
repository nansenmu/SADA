# Third-party notices

## SurvPath

Upstream: https://github.com/mahmoodlab/SurvPath

Paper: Guillaume Jaume, Anurag Vaidya, Richard J. Chen, Drew F. K. Williamson, Paul Pu Liang, and Faisal Mahmood. Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction. CVPR, 2024.

The upstream README states:

> This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

The inherited data pipeline, attention, baseline compatibility modules, optimizer and utilities retain their upstream notices. `models/model_SurvPath.py`, `utils/core_utils.py`, `utils/sada_loss.py`, and the sampling modules contain the SADA study's extensions. The historical filenames are retained for compatibility.

The GPL version 3 text is supplied in `LICENSES/GPL-3.0.txt`. No MIT or other replacement license is applied to upstream code. Refer to the upstream repository for its full usage terms.
