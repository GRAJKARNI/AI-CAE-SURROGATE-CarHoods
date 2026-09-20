# Dataset

This project uses the **CarHoods10k** automotive hood dataset.

The dataset contains parametric CAD designs, 3D geometry representations and corresponding finite-element analysis (FEA) response quantities.

## Dataset Used in This Project

The condensed HDF5 dataset used for the surrogate-model development contains:

- 9,982 automotive hood designs
- 54 CAD/topological design parameters
- 8,192 XYZ surface points per design
- Maximum Von Mises stress [MPa]
- Maximum deformation [mm]
- Mass [kg]

The HDF5 file contains the following top-level datasets:

```text
design_parameters
points
stress
deformation
mass
```

with the principal shapes:

```text
design_parameters : (9982, 54)
points             : (9982, 8192, 3)
stress             : (9982,)
deformation        : (9982,)
mass               : (9982,)
```

## Geometry Validation

Before geometry-based modeling, the point clouds were checked for invalid or zero geometry.

51 records were excluded, leaving:

**9,931 valid designs**

for the main geometry-based surrogate-model development.

## CAD Parameter Names

The condensed HDF5 file contains the 54 CAD/topological parameter values but does not contain an authoritative mapping between the parameter columns and their original CAD feature names.

Therefore, this project refers to the parameters as:

**P1, P2, ..., P54**

rather than assigning unsupported physical definitions.

## Dataset Availability

The dataset itself is **not included in this GitHub repository** because of its size.

The original CarHoods10k dataset is available through the Dryad Digital Repository.

**DOI:** 10.5061/dryad.2fqz612pt

The original dataset is more than 400 GB. This project uses a condensed HDF5 representation containing the design parameters, downsampled point-cloud geometry and FEA response quantities.

## Expected Local File

The notebooks expect the condensed dataset to be available locally as:

```text
CarHoods_Extracted.h5
```

When using Google Colab, update the dataset path in the notebooks to match the location of the file in Google Drive.

## Important

Large dataset files are intentionally excluded from version control.

Files such as:

```text
*.h5
*.hdf5
```

should remain excluded through the repository `.gitignore`.