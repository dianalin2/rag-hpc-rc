+++
images = [""]
author = "Staff"
description = ""
date = "2022-12-06T00:00:00-05:00"
title = "Rivanna Maintenance: December 19, 2022"
# url = "/maintenance"
draft = false
tags = ["rivanna"]
categories = ["feature"]
+++

{{< alert-green >}}Rivanna will be down for maintenance on <strong>December 19, 2022</strong> beginning at 6 a.m.{{< /alert-green >}}

You may continue to submit jobs until the maintenance period begins, but if the system determines your job will not have time to finish, it will not start until Rivanna is returned to service. Users will not be able to access the Globus data transfer node during the maintenance period.

All systems are expected to return to service by **6 a.m. on Tuesday, December 20**. Globus users may need to rebuild their shared collections.

## IMPORTANT MAINTENANCE NOTES

Two new toolchains are now available: `goolf/11.2.0_4.1.4` and `intel/2022.11`. The former consists of GCC 11.2.0, OpenMPI 4.1.4, and math libraries. The latter is Intel oneAPI that consists of the Intel compilers, MKL, and MPI. (Note that there is no need to load `intelmpi` from this version onwards.) The default versions have not been upgraded this time. Users with their own compiled codes are encouraged to build them with the new toolchains and report any issues.

### Modules

1. The following software modules will be **removed** from Rivanna during the maintenance period:

| Module | Removed version | Replacement |
|---|---|---|
|blender     |2.78c | 3.2.1 |
|cellranger      |4.0.0 | 5.0.0, 6.0.1, 7.0.1 |
|cellranger-atac |1.2.0 | 2.0.0 |
|diamond     |0.9.13| 2.0.14|
|drmaa       |1.1.2 | 1.1.3 |
|fsl         |6.0.0 | 6.0.5 |
|go          |1.8.1, 1.13.4 | 1.18.4 |
|gparallel   |20170822 | parallel/20200322 |
|gurobi      |9.0.1, 9.1.1 | 9.5.0 |
|htslib      |1.4.1 | 1.7, 1.9 |
|igvtools    |2.8.9 | 2.12.0 |
|librmath    |3.6.2 | 3.6.3 |
|macs2       |2.1.2 | 2.2.7.1 |
|micromamba  |0.7.14| 0.24.0 |
|nextflow    |0.26.0.4715 | 20.10.0 |
|salmon      |1.2.1 | 1.5.1 |
|seqoutbias  |1.2.0 | 1.3.1 |
|sratoolkit  |2.8.0, 2.9.1 | 2.10.5 |
|trimmomatic |0.36  | 0.39 |
|vcftools    |0.1.15| 0.1.16 |

<sup>*</sup>Archived containers can be found in `/share/resources/containers/singularity/archive`.

2. **Upgrades**:
    - eccodes/2.26.0 - under `gcc/9.2.0 openmpi/3.1.6`
    - hic-pro/3.1.0
    - kraken2/2.1.2 - from `kraken/0.10.5`; note module name change
    - openfoam/v2206 - under `goolf/9.2.0_3.1.6`
    - R/4.2.1 - under `goolf/9.2.0_3.1.6`
    - ruby/3.1.2 - under `gcc/9.2.0`

    Default version changes:
    - alphafold/2.2.0 &rarr; 2.3.0
    - cellranger/5.0.0 &rarr; 7.0.1
    - cmake/3.16.5 &rarr; 3.23.3
    - deeplabcut/2.2 &rarr; 2.2.1.1-anipose
    - matlab/R2022a &rarr; R2022b
    - pytorch/1.10.0 &rarr; 1.12.0 (includes PyTorch Geometric)
    - qiime2/2020.8 &rarr; 2022.2
    - tensorflow/2.7.0 &rarr; 2.10.0

3. **New** modules:
    - clang/10.0.1
    - crossftp/1.99.9
    - gawk/5.1.1
    - isaacgym/1.0.preview4
    - nibabies/22.1.3
    - rnaeditor/1.1a
