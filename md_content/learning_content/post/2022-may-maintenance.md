+++
images = [""]
author = "Staff"
description = ""
date = "2022-05-02T00:00:00-05:00"
title = "Rivanna Maintenance: May 17, 2022"
# url = "/maintenance"
draft = false
tags = ["rivanna"]
categories = ["feature"]
+++

{{< alert-green >}}Rivanna will be down for maintenance on <strong>May 17, 2022</strong> beginning at 6 a.m.{{< /alert-green >}}

You may continue to submit jobs until the maintenance period begins, but if the system determines your job will not have time to finish, it will not start until Rivanna is returned to service. Users will not be able to access the Globus data transfer node (UVA Main-DTN) or Research Project storage during the maintenance period. All systems are expected to return to service by **6 a.m. on Wednesday, May 18**.

## IMPORTANT MAINTENANCE NOTES

- The operating system will be upgraded from CentOS 7.8 to 7.9. This should have no impact on the software built on Rivanna, whether it be modules or your own compiled codes. If you need assistance to rebuild your code, please contact hpc-support@virginia.edu.
- Slurm will be upgraded to 21.08.8, which requires us to rebuild the OpenMPI module. If your code was built with OpenMPI and it no longer works after maintenance, you may need to rebuild it.
- NVIDIA Driver will be upgraded to 470.103.01. You should not need to rebuild CUDA programs.

### Modules

1. The following software modules will be **removed** from Rivanna during the maintenance period:

| Module | Removed version | Replacement |
|---|---|---|
|gcc       |6.5.0 | 7.1.0, 9.2.0 |
|mvapich2  |2.3.1, 2.3.3 | Please use `gcc openmpi` or `intel intelmpi`. |
|nvhpc     |20.9 | 21.9 |
|alphafold<sup>*</sup> |2.0.0, 2.1.1 | 2.1.2, 2.2.0 |
|awscli    |2.1.10 | 2.4.12 |
|cellranger|2.2.0, 3.0.2, 3.1.0 | 4.0.0, 5.0.0, 6.0.1 |
|cmake     | 3.5.2, 3.12.3 | 3.6.1, 3.16.5 |
|gatk      |3.8.1.0, 4.0.0.0, 4.1.6.0 | 4.2.3.0 |
|metamorpheus<sup>*</sup>|0.0.311-dev, 0.0.317 | 0.0.320 |
|mpi4py    |3.0.0-py2.7, 3.0.3 | Load any MPI toolchain (e.g. `gcc openmpi`) plus `anaconda` and run `pip install --user mpi4py`; see [here](https://mpi4py.readthedocs.io/en/stable/install.html) |
|picard    |2.1.1, 2.18.5, 2.20.6 | 2.23.4 |
|rapidsai<sup>*</sup>  |0.19 | 21.10 |

<sup>*</sup>Archived containers can be found in `/share/resources/containers/singularity/archive`.

2. **Upgrades**:
    - Addition of Matplotlib widget ipympl/0.8.7 to JupyterLab
    - tensorflow/2.8.0
    - swig/4.0.2

   Default version changes:
    - alphafold/2.1.1 &rarr; 2.2.0
    - cellprofiler/3.1.8 &rarr; 4.2.1
    - cuda/11.0.228 &rarr; 11.4.2
    - diamond/0.9.13 &rarr; 2.0.14
    - igvtools/2.8.9 &rarr; 2.12.0
    - matlab/R2021b &rarr; R2022a
    - totalview/2019.0.4_linux_x86-64 &rarr; 2021.4.10
    - zlib/1.2.11 &rarr; 1.2.12 - hidden module; load `zlib/.1.2.12`

3. **New** modules:
    - nvompic/21.9_3.1.6_11.4.2 toolchain (nvhpc/21.9 + openmpi/3.1.6 + cuda/11.4.2)
        - libraries: scalapack, fftw, hdf5
        - berkeleygw/3.0.1
        - quantumespresso/7.0
        - yambo/5.0.4
    - pandoc/2.17
    - trinity/2.13.2
    - cufflinks/2.2.1
    - redis-cli/6.2.6
