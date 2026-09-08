# Electron-proton-classifier

## Exercises

In order — each one builds on the previous.

1. [Transformer particle-ID classifier](transformer/README.md) — train a
   transformer to identify e / p / C from the raw detector response.
2. [Neural optimal transport toy exercise](optimal_transport_toy/README.md) —
   implement an ICNN and the Makkuva dual on a 2-D toy problem.
3. [Latent-space optimal-transport calibration](latent_ot/README.md) — put the
   two together: extract the classifier's latent representation and learn an OT
   map that corrects simulation to real test-beam data.
