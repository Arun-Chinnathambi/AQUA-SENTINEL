# Equation-to-code mapping

1. Adaptive log intensity: P1(x)=c log(1+I(x)), c=255/log(1+Imax). Implemented in `src/physics.py::physics_features`.
2. Normalized Laplacian: P2(x)=|∇²I(x)|/max_u|∇²I(u)|. Implemented in the same function.
3. Gated fusion: alpha=sigmoid(Wg*[Fr;Fs]); F=alpha Fr+(1-alpha)Fs. Implemented by `src/model.py::Gate` at four skip levels and bottleneck.
4. Lookalike gate: p_hat=sigmoid(s)*(1-rho), rho=sigmoid(z). The training model returns segmentation logits and z; deployment code can apply the gate before PIECL.
5. Loss: 0.5 Dice + 0.3 BCE + 0.2 focal + 0.3 auxiliary BCE. Implemented in `src/losses.py`.
6. Environmental plausibility: Phi=W*T, W=D/max(D), T=1-Lambda/max(Lambda). Implemented in `src/postprocess.py`.
7. Confidence adaptive threshold: theta=0.60-0.40*gamma, gamma=1-rho. Implemented in `apply_constraint`.
8. Pooled P/R/F1/FDR and per-image Dice/IoU are implemented in `src/metrics.py`; empty-empty Dice/IoU = 1.
