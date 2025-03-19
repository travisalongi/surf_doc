## Running and Testing SURF

The core algorithms reside in `SURF/Src`, handling data ingestion, clustering, merging, regression fitting, plotting, and 3D rendering.

To test the package, navigate to one of the example directories (e.g., `./Examples/Ex_RC/`). The workflows are numbered in the sequence they should be executed.

To run all example workflows at once:

```zsh
chmod +x ./Examples/Ex_Sjb/run_all_workflows.sh   # Allows you to execute the script
./Examples/Ex_Sjb/run_all_workflows.sh
```

Example scripts use our recommended default parameters in `params_file.py` that can be modified. Users should not need to change parameters that being with a underscore (e.g.         `"_init_cluster_dir"`)


