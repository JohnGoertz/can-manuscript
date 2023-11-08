The following instructions are for a Linux system. For a Windows system, you should use the Windows Subsystem for Linux 2 (WSL2). For a Mac system, you may need to install libcairo2 using different commands, but the setup after that point should be the same.

You should start in the root directory of the project.

0. Ensure you have `mamba` installed (or change the commands below for conda): https://mamba.readthedocs.io/en/latest/installation.html
1. Unzip env/candas.zip and env/nupack-4.0.1.7.zip into the env/ directory
2. Use the following commands to set up the environment.
    ```bash
    # From the root directory.
    cd env
    sudo apt update
    sudo apt install libcairo2-dev  # needed only for saving SVGs
    mamba env create -f env.yml --prefix ./can_manuscript
    conda activate can_manuscript
    pip install -e candas
    pip install -U nupack -f nupack-4.0.1.7/package/
    ```
    

# ParameterSets Pipeline
Many of the figures in this paper rely on aggregate data gathered from ~ two dozen experiments. This data is contained if files named along the lines of "ADVI_ParameterSet_220528.pkl". These experimental files were processed in a Snakemake pipeline. All necessary data and code is included in the `pipeline` directory. To view a dry-run of the pipeline execution, activate the environment as above, navigate to the `pipeline` directory, and run the command `snakemake -n`. Snakemake will identify which generated files are out-of-date relative to their input files, which scripts need to be executed in order to build the target file, and execute them. However, this pipeline was intended to be run in a High Performance Computing environment with extensive computational resources. It will take a very long time to complete on a personal computer, potentially exceeding the resources available (and then crashing). If needed, you can drop the `-n` flag and run simply `snakemake`, but this is not recommended.