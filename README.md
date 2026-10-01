# encoding-templates

Experiments with different encoding solvers in the THINGS dataset.

## Running experiments

To run experiments, we recommend using the defined entrypoint `encoding_experiment`.
For example, to run a local analysis of Reduced-Rank Ridge solver performance in the
pre-defined EBA ROI of sub-01:

.. code-block::

    $ pip install -e .
    $ encoding_experiment --sub_name="sub-01" --roi="EBA" \
        --cv_strategy="kfold" --scoring_metric="r2_score" \
        --solver="rrr" \
        --data_dir="/Users/emdupre/Desktop/UdeM-Projects/things-encode"


## Running on HPCs

Note that these analyses benefit from GPU availability;
as such, there is a containerized workflow tailed to
[Alliance Canada](https://www.alliancecan.ca/)'s [Rorqual cluster](https://docs.alliancecan.ca/wiki/Rorqual/).

To re-generate the supporting Apptainer image:

```
module load apptainer/1.3.4
apptainer cache list
apptainer cache clean
apptainer build --notest things-apptainer.sif apptainer.def
```

then, we can confirm it can access the GPU in a Rorqual HPC interactive job with:

```
module load apptainer/1.3.4
module load cuda
salloc --account=<DEF-ACCOUNT> --gpus-per-node=h100_3g.40gb:1 --time=00:05:00 --mem=1G

nvidia-smi
apptainer test --nv things-apptainer.sif
```

the actual analyses can be re-launched using:

```
sbatch sbatch_scripts/encoding.sbatch
```
