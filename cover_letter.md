# Cover Letter — Applied Acoustics

Dear Editor,

We submit our manuscript "Lightweight CPU-bound convolutional neural networks for
acoustic anomaly detection of industrial machinery under severe noise: spectral
representation, class imbalance, and hardware-decoupled deployment" for
consideration as a research article in Applied Acoustics.

The paper presents a systematic experimental study of supervised acoustic anomaly
detection on the MIMII benchmark at the extreme −6 dB signal-to-noise operating
point, where the published autoencoder baseline attains an AUC of only ≈0.64.
Its contributions are directly relevant to the journal's readership in
machinery acoustics and acoustic signal processing:

1. We isolate and explain a failure mode in which the most class-imbalanced
   machine category (pneumatic valves, 7.7:1) collapses to majority-class
   prediction, and show that recovery requires the joint application of
   log-Mel spectral input and class-weighted loss — cepstral (MFCC) features
   remain collapsed even when re-weighted, because the DCT discards the
   inter-band co-occurrence structure of broadband valve transients.

2. We demonstrate that single-run results for severely imbalanced categories
   are seed-unstable, and therefore report a ten-seed protocol with bootstrap
   confidence intervals, partial AUC, and calibrated decision thresholds —
   a methodological caution we believe is of broad value to the anomalous
   sound detection community.

3. We benchmark deployment on a commodity six-core CPU shared with a fully
   loaded GPU visual-inspection workload, showing ~500 frames per second of
   acoustic inference on two CPU threads with zero video-memory usage and at
   most single-digit interference — evidence for practical hardware-decoupled
   multimodal condition monitoring at the industrial edge.

The manuscript additionally reports architecture, representation, and SNR
ablations, a convolutional-autoencoder reproduction under identical data
handling, and leave-one-machine-ID-out cross-validation that transparently
delimits generalization to unseen machine units.

This work is original, has not been published, and is not under consideration
elsewhere. All authors approved the submission and declare no competing
interests. The MIMII dataset is publicly available; our code and experiment
logs are available to reviewers.

Suggested reviewers: researchers active in DCASE Task 2 / machinery anomalous
sound detection.

Sincerely,
Tao Zhang (corresponding author), on behalf of the authors
Department of Mechanical Engineering,
Huai'an University, Huai'an, Jiangsu, China
