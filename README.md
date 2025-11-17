### ICOR Codon Optimization – Restriction Site Avoidance Fork

This repository is a fork of the original ICOR codon optimization tool. It adds a **restriction site avoidance** layer and a small **programmatic API** on top of the existing ICOR optimizer.

The core ICOR model and benchmarking assets are unchanged; this fork focuses only on post-optimization restriction handling and easier integration into other Python code.

---

### Features added in this fork

- **Restriction site avoidance for optimized sequences**
  - After codon optimization, the DNA sequence is scanned for common restriction enzyme recognition sites.
  - If any such site is found, one or more **synonymous codon substitutions** are applied to break the site:
    - The amino acid sequence is preserved.
    - Codon usage remains as close as possible to E. coli–biased usage (may slightly degrade if necessary).
  - This process is repeated until **no forbidden sites remain** or an informative error is raised.

- **Restriction site database**
  - The following sites are avoided by default:

    | Enzyme           | Recognition Site      |
    |------------------|-----------------------|
    | EcoRI            | GAATTC                |
    | HindIII          | AAGCTT                |
    | BamHI            | GGATCC                |
    | PstI             | CTGCAG                |
    | XbaI             | TCTAGA                |
    | SpeI             | ACTAGT                |
    | NheI             | GCTAGC                |
    | XhoI             | CTCGAG                |
    | SalI             | GTCGAC                |
    | NcoI             | CCATGG                |
    | KpnI             | GGTACC                |
    | ApaI             | GGGCCC                |
    | SacI             | GAGCTC                |
    | SmaI             | CCCGGG                |
    | NotI             | GCGGCCGC              |
    | BglII            | AGATCT                |
    | ClaI             | ATCGAT                |
    | BsaI             | GGTCTC                |
    | BsmBI            | CGTCTC                |
    | BbsI             | GAAGAC                |
    | BsaI-HF / SapI   | GCTCTTC               |
    | BstXI            | CCANNNNNNTGG          |

  - Degenerate bases such as **N** in BstXI are treated as wildcards (`[ACGT]`).
  - The site list is defined in `tool/optimizers/restriction_site_manager.py` and can be extended.

- **Programmatic API for ICOR optimization**
  - The ICOR optimizer can now be called directly from Python:

    ```python
    from tool.optimizers.icor_optimizer import optimize_sequence

    # Amino acid input (recommended)
    optimized_dna = optimize_sequence(
        sequence="MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKER*",
        sequence_type="AA",            # or "DNA"
        avoid_restriction_sites=True,  # default in this fork
    )
    ```

  - `avoid_restriction_sites`:
    - `True` (default): run restriction site avoidance.
    - `False`: return ICOR’s optimized DNA without post-processing.

---

### CLI usage in this fork

From the repository root:

```bash
pip3 install -r requirements.txt
python3 ./tool/optimizers/icor_optimizer.py
```

During the interactive run you will be prompted for:

- Input type: amino acid (`aa`) or DNA (`dna`).
- Sequence (or `demo` to use the built-in example).

Restriction site avoidance is **enabled by default**. You can explicitly control it via flags:

```bash
# Enable restriction site avoidance (default)
python3 ./tool/optimizers/icor_optimizer.py --avoid-restriction-sites

# Disable restriction site avoidance
python3 ./tool/optimizers/icor_optimizer.py --no-avoid-restriction-sites
```

The script will print the optimized DNA sequence and optionally write it to `output.txt`.

---

### Error handling

If a restriction site cannot be broken without changing the amino acid sequence, the optimizer will raise a clear error indicating:

- Which restriction site was found.
- The position in the sequence.
- That it cannot be repaired without altering the encoded protein.

This situation should be rare and usually indicates a highly constrained region where no synonymous codon choices are available to disrupt the site.


