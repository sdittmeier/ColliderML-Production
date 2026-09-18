# Paired vs published event notebook

`paired_vs_published.ipynb` compares one new ttbar event with one independent
published event. It defaults to pu0 event 0 in each source. Change `SAMPLE` to
`'pu200'` in the first code cell for pileup 200, or edit the event IDs and
paths there. The output directory is named `paired-samples-features-n1`, but
the completed run contains two events per sample.

The notebook needs Python with `numpy`, `pyarrow`, `pandas`, `matplotlib`,
`ipykernel`, and `jupyterlab`. A separate regular conda environment keeps these
plotting dependencies out of the ACTS runtime:

```bash
conda create -n paired-validation -c conda-forge \
  python=3.10 numpy pyarrow pandas matplotlib ipykernel jupyterlab
conda activate paired-validation
cd /home/atlas/dittmeier/git/cluster_extraction/ColliderML-Production
python -m jupyter lab notebooks/validation/paired_vs_published.ipynb
```

If `paired-validation` already exists, install any missing packages into it
instead of creating it again. The default new-data path points to this host's
`/tmp/colliderml-paired-40167/output/paired-samples-features-n1`; adjust it
if your completed output was copied elsewhere.

Blue always represents the new paired sample and orange the published release.
The two events are independent, so compare distributions and detector coverage,
not equality of individual values. All displayed histograms share bins and are
normalized within each source. Spatial views sample at most 10,000 hits per
source for readability; integrity counts use the full selected events.
Panels with a dominant bin and sparse tails use a logarithmic y-axis; the
feature values and shared x-bins remain unchanged.
