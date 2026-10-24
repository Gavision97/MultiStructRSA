\# SLI Virtual Screening and Docking



This case study evaluated MultiStructRSA for compound prioritization, followed by docking to assess the structural plausibility of selected candidates.



\## Target



Stem-loop I (SLI) of the IncB plasmid replication system was selected based on strong cross-validation performance across 11 R-SIM compounds (PCC ≈ 0.92, SCC ≈ 0.91, RMSE ≈ 0.38) and prior experimental evidence implicating A22/A23 in aminoglycoside binding.



\*\*RNA sequence:\*\* `AUUUUUCCUCGAACUUGGCGGAACGCAGAAAAAU`



\## Pipeline



1\. \*\*Virtual screening:\*\* Screened approximately 24,000 compounds from the ChemDiv RNA-Binding Chemical Space Library using MultiStructRSA.

2\. \*\*Compound prioritization:\*\* Ranked compounds using `S = mean(pKd) − std(pKd)`, calculated over 100 Monte Carlo dropout passes, and selected the top five.

3\. \*\*Structure preparation:\*\* Used a fixed AlphaFold 3-predicted SLI structure as the receptor.

4\. \*\*Site definition:\*\* Defined an approximately 304 Å³ docking region using the 20 heavy atoms of the A22/A23 bases.

5\. \*\*Docking:\*\* Ran rDock with 50 runs per compound and random seed 42. Selected each best pose by lowest total `SCORE` and compared intermolecular interactions using `SCORE.INTER`.

6\. \*\*Visualization:\*\* Inspected representative poses, close contacts, and predicted hydrogen bonds in ChimeraX.



\## Interpretation



Docking provided complementary evidence of structural plausibility within the experimentally implicated A22/A23 region. It did not establish binding affinity or experimentally validate the model's predictions.

