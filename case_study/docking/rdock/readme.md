# SLI A22/A23 docking files

This directory contains rDock inputs and docking results for the
AlphaFold 3-predicted SLI RNA structure, targeting the A22/A23 region.

## Shared inputs

| File | Description |
|---|---|
| `SLI_A22A23_bases.prm` | Receptor and binding-site configuration used for both the standard and SF5 runs. |
| `SLI_A22A23_bases.as` | Generated rDock cavity file. |
| `A22A23_base_atoms.sdf` | A22/A23 base-atom reference used to define the docking region. |

The receptor structure is stored outside this directory:

`/home/gavrilev/Thesis/src/case_study/docking/receptor/SLI_model_0.mol2`

Ligand inputs were read from:

`/home/gavrilev/Thesis/src/case_study/docking/ligands/<GROUP>/<id>.sdf`

The selected screening candidates were stored under `ligands/top5/`.

## Binding-site settings

The configuration uses `RbtLigandSiteMapper` with:

- Reference molecule: `A22A23_base_atoms.sdf`
- Radius: 5.0 Å
- Small sphere: 1.0 Å
- Minimum volume: 100 Å³
- Maximum cavities: 1
- Volume increment: 0.0
- Grid step: 0.5 Å
- Cavity scoring function: `RbtCavityGridSF`, weight 1.0

## Docking results

| Filename pattern | Description |
|---|---|
| `dock_<id>_bases.sd` | Poses generated using the standard `$RBT_ROOT/data/scripts/dock.prm` protocol. |
| `dock_<id>_bases_sorted.sd` | Standard-protocol poses sorted numerically by total `SCORE`, lowest first. |
| `dock_<id>_sf5.sd` | Poses generated using the later SF5 protocol, supplied through the shell variable `$SF5`. |
| `dock_<id>_sf5_sorted.sd` | SF5 poses sorted numerically by total `SCORE`, lowest first. |
| `best_pose_4554-4658.sdf` | Separately saved pose for compound 4554-4658. Its source run has not been verified. |

Both protocols used 50 docking runs per ligand and random seed 42.
The sorted files contain the generated poses, ordered by score; they
are not single-pose files.

The workflow selected the best pose using the lowest total `SCORE`.
Cross-compound interaction comparisons used `SCORE.INTER`.

## Protocol provenance

Shell history confirms that the SF5 runs occurred after the standard
runs and used the same `SLI_A22A23_bases.prm` configuration.

The exact file assigned to `$SF5` is not recorded here. Earlier project
notes mention `dock_solv.prm`; verify the original variable assignment
before reproducing the SF5 runs.

The filenames alone do not establish which results were used for the
final manuscript scores or ChimeraX figure. Preserve both result sets
until that provenance is verified.

The compounds shown in the case-study figure were:

- Y600-4636
- 8020-4333
- 4554-4658

## Earlier experiments

If still present, files containing `r9_5`, `two_cav`, `radius12`,
`finaltest`, `large5`, or `small1`, together with the initial
`SLI.prm` and unsuffixed `dock_4554-4658.sd` results, belong to earlier
site-definition or protocol experiments.
