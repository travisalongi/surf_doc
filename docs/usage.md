##  Running and Testing SURF

The core algorithms reside in `surf/surf/`, handling data ingestion, regression fitting, plotting, and 3D rendering. Clustering, merging, and visualization examples are in `surf/examples/*`. 

To test the package, navigate to one of the example directories (e.g., `./examples/rc/`). The workflows are numbered in the sequence they should be executed.

To run all example workflows at once:

```zsh
cd examples/sjb/
chmod +x run_all_workflows.sh   # Allows you to execute the script
./run_all_workflows.sh
```

Example scripts use our recommended default parameters in `params_file.py` that can be modified. Users should not need to change parameters that begin with a underscore (e.g. `"_init_cluster_dir"`)


