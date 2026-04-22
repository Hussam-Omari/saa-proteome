# saa-proteome

A reproducible Python library for proteome-wide amino acid composition analysis, 
with dedicated support for sulfur-containing amino acids (methionine and cysteine) 
and sliding-window–based methionine enrichment metrics.

---

## Version

Current version: v0.3.0

---

## License

This project is licensed under the MIT License.

---

## Scientific Purpose

saa_proteome enables deterministic and reproducible analysis of:

- Protein-level amino acid composition  
- Proteome-level aggregated composition  
- Sulfur amino acid (SAA) metrics (Met%, Cys%, SAA%)  
- Essential amino acid (EAA) summaries  
- Sliding-window maximum methionine enrichment  
- Top-N protein ranking tables  
- Quality control (QC) and summary statistics  
- Publication-ready outputs  

All outputs are returned as pandas DataFrames and can be exported as CSV files or visualized as heatmaps and bar charts.

---

## Computational Workflow

![Workflow](docs/workflow.png)

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/<your-username>/saa-proteome.git
cd saa_proteome
pip install -e .
```

---

## Data Source

Proteome FASTA files used in this study were obtained from UniProt reference proteomes. 
Specific proteome identifiers and download links are provided in the repository or associated publication.

---

## Output Schema

A complete description of all output variables is provided in `output_dictionary.csv`, 
including variable names, definitions, and units.

---

## Minimal Example

```python
from saa_proteome import load_proteome_metrics, saa_summary

df = load_proteome_metrics(
    fasta_path="path/to/proteome.fasta",
    species="Glycine max"
)
# Compute species-level summary
summary = saa_summary(df)
print(summary)
