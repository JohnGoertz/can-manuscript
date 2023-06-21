The following instructions are for a Linux system. For a Windows system, you should use the Windows Subsystem for Linux 2 (WSL2). For a Mac system, you may need to install python3.9 using different commands, but the setup after that point should be the same.

0. Ensure you have `mamba` installed (or change the commands below for conda): https://mamba.readthedocs.io/en/latest/installation.html
1. Unzip env/candas.zip and env/nupack-4.0.1.7.zip
2. Use the following commands to set up the environment.
    ```bash
    # You should start in the can_paper/ directory
    cd env
    sudo apt update
    sudo apt install libcairo2-dev
    mamba env create -f env.yml --prefix ./can_manuscript
    conda activate can_manuscript
    pip install -e candas
    pip install -U nupack -f nupack-4.0.1.7/package/
    ```
3. Open Jupyter Lab in a browser from the root directory for the project
    ```bash
    cd ../  # You should now be back in the can_paper/ directory
    jupyter lab
    ```

    * A browser window should open; if not copy-paste the full hyperlink that appears into a browser.
    
4. Run each notebook
    * Open the .ipynb file in the `code/` subdirectory for each figure, and click "Run All". You may want to first remove all files from the "graphics" subdirectory to verify that they are freshly generated.
    * Keep in mind that some notebooks, in particular the ones for Fig 2, may require up to 64GB of RAM.