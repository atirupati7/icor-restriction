import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


IUPAC_DNA_MAP: Dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    # Treat ambiguous bases as wildcards over all four bases unless otherwise specified
    "N": "[ACGT]",
}


def _pattern_to_regex(pattern: str) -> str:
    """Convert a restriction site pattern with optional ambiguous bases into a regex."""
    regex_parts: List[str] = []
    for ch in pattern.upper():
        if ch in IUPAC_DNA_MAP:
            regex_parts.append(IUPAC_DNA_MAP[ch])
        else:
            # Fallback: treat unknown codes as literal to avoid over-matching
            regex_parts.append(ch)
    return "".join(regex_parts)


@dataclass
class RestrictionSite:
    name: str
    pattern: str
    regex: re.Pattern


@dataclass
class RestrictionSiteMatch:
    site: RestrictionSite
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive


class RestrictionSiteManager:
    """
    Store and query restriction sites and provide helpers for avoidance.

    This class is intentionally independent of any particular optimizer so it can
    be reused by different optimization strategies.
    """

    def __init__(self, extra_sites: Dict[str, str] | None = None) -> None:
        # Core, commonly used restriction sites. This list is intentionally
        # small and easily extensible – additional sites can be provided via
        # `extra_sites` or by editing this dictionary.
        sites: Dict[str, str] = {
            # 6-bp and 8-bp palindromic cutters
            "EcoRI": "GAATTC",
            "HindIII": "AAGCTT",
            "BamHI": "GGATCC",
            "PstI": "CTGCAG",
            "XbaI": "TCTAGA",
            "SpeI": "ACTAGT",
            "NheI": "GCTAGC",
            "XhoI": "CTCGAG",
            "SalI": "GTCGAC",
            "NcoI": "CCATGG",
            "KpnI": "GGTACC",
            "ApaI": "GGGCCC",
            "SacI": "GAGCTC",
            "SmaI": "CCCGGG",
            "NotI": "GCGGCCGC",
            "BglII": "AGATCT",
            "ClaI": "ATCGAT",
            # Type IIS enzymes (recognition site only, ignoring cut offset)
            "BsaI": "GGTCTC",
            "BsmBI": "CGTCTC",
            "BbsI": "GAAGAC",
            "BsaI-HF/SapI": "GCTCTTC",
            # Degenerate-base cutter
            "BstXI": "CCANNNNNNTGG",
        }

        if extra_sites:
            # Allow extension/override of built-in sites
            sites.update(extra_sites)

        self._sites: List[RestrictionSite] = []
        for name, pattern in sites.items():
            regex = re.compile(_pattern_to_regex(pattern))
            self._sites.append(RestrictionSite(name=name, pattern=pattern, regex=regex))

    @property
    def sites(self) -> List[RestrictionSite]:
        return list(self._sites)

    def find_sites(self, dna_sequence: str) -> List[RestrictionSiteMatch]:
        """
        Scan the DNA sequence for exact matches to any configured restriction site.

        Returns a list of matches, each with site metadata and 0-based coordinates.
        """
        seq = dna_sequence.upper()
        matches: List[RestrictionSiteMatch] = []
        for site in self._sites:
            for m in site.regex.finditer(seq):
                matches.append(
                    RestrictionSiteMatch(site=site, start=m.start(), end=m.end())
                )
        return matches

    def has_sites(self, dna_sequence: str) -> bool:
        """Return True if any configured restriction site is present in the sequence."""
        return bool(self.find_sites(dna_sequence))

    def summary(self) -> List[Tuple[str, str]]:
        """Return a table-like list of (name, pattern) for documentation or display."""
        return [(s.name, s.pattern) for s in self._sites]


# --- Synonymous codon handling and site fixing helpers ---

# E. coli-biased codon frequencies (copied from BFC optimizer)
AA_CODON_FREQUENCIES: Dict[str, Tuple[List[str], List[float]]] = {
    "A": (["GCG", "GCA", "GCT", "GCC"], [0.34, 0.22, 0.17, 0.27]),
    "R": (["AGG", "AGA", "CGG", "CGA", "CGT", "CGC"], [0.03, 0.05, 0.1, 0.07, 0.37, 0.38]),
    "N": (["AAT", "AAC"], [0.46, 0.54]),
    "D": (["GAT", "GAC"], [0.63, 0.37]),
    "C": (["TGT", "TGC"], [0.45, 0.55]),
    "*": (["TGA", "TAG", "TAA"], [0.3, 0.08, 0.62]),
    "Q": (["CAG", "CAA"], [0.66, 0.34]),
    "E": (["GAG", "GAA"], [0.32, 0.68]),
    "G": (["GGG", "GGA", "GGT", "GGC"], [0.15, 0.12, 0.34, 0.39]),
    "H": (["CAT", "CAC"], [0.57, 0.43]),
    "I": (["ATA", "ATT", "ATC"], [0.09, 0.5, 0.41]),
    "L": (["TTG", "TTA", "CTG", "CTA", "CTT", "CTC"], [0.13, 0.13, 0.49, 0.04, 0.11, 0.1]),
    "K": (["AAG", "AAA"], [0.25, 0.75]),
    "M": (["ATG"], [1.0]),
    "F": (["TTT", "TTC"], [0.57, 0.43]),
    "P": (["CCG", "CCA", "CCT", "CCC"], [0.51, 0.2, 0.17, 0.12]),
    "S": (["AGT", "AGC", "TCG", "TCA", "TCT", "TCC"], [0.15, 0.26, 0.15, 0.13, 0.16, 0.15]),
    "T": (["ACG", "ACA", "ACT", "ACC"], [0.26, 0.15, 0.18, 0.42]),
    "W": (["TGG"], [1.0]),
    "Y": (["TAT", "TAC"], [0.58, 0.42]),
    "V": (["GTG", "GTA", "GTT", "GTC"], [0.36, 0.16, 0.27, 0.21]),
}


def _sorted_synonymous_codons(aa: str) -> List[str]:
    """
    Return synonymous codons for the given amino acid, sorted by descending
    frequency (most optimal first).
    """
    aa = aa.upper()
    if aa not in AA_CODON_FREQUENCIES:
        return []
    codons, freqs = AA_CODON_FREQUENCIES[aa]
    # Pair and sort by frequency descending
    paired = sorted(zip(codons, freqs), key=lambda x: x[1], reverse=True)
    return [c for c, _ in paired]


class RestrictionSiteFixError(RuntimeError):
    """Raised when a restriction site cannot be removed without changing the amino acid sequence."""


def fix_restriction_sites(
    dna_sequence: str,
    amino_acid_sequence: str,
    manager: RestrictionSiteManager | None = None,
) -> str:
    """
    Iteratively break all restriction sites using synonymous codon substitutions.

    Parameters
    ----------
    dna_sequence:
        Optimized DNA sequence (must be in-frame starting at position 0).
    amino_acid_sequence:
        The amino-acid sequence encoded by `dna_sequence`. This is used to
        ensure only synonymous substitutions are considered.
    manager:
        Optional existing `RestrictionSiteManager` instance. If not provided a
        default manager with common sites is created.

    Returns
    -------
    str
        A new DNA sequence with all configured restriction sites removed.

    Raises
    ------
    RestrictionSiteFixError
        If no synonymous codon change can be found to remove a given site.
    """
    if manager is None:
        manager = RestrictionSiteManager()

    seq = dna_sequence.upper()
    aa_seq = amino_acid_sequence

    if len(seq) % 3 != 0 or len(seq) // 3 != len(aa_seq):
        raise ValueError(
            "DNA sequence length must be a multiple of 3 and consistent with the amino acid sequence length."
        )

    # Keep iterating until no sites remain
    while True:
        matches = manager.find_sites(seq)
        if not matches:
            return seq

        # Process matches one by one; recompute from scratch after each change
        match = matches[0]
        site = match.site

        site_fixed = False

        # Try changing each codon that overlaps the restriction site
        for nt_index in range(match.start, match.end):
            codon_index = nt_index // 3
            codon_start = codon_index * 3
            codon_end = codon_start + 3

            current_codon = seq[codon_start:codon_end]
            aa = aa_seq[codon_index]
            candidates = _sorted_synonymous_codons(aa)

            # If we don't know any synonyms (e.g., non-standard AA), skip this codon
            if not candidates:
                continue

            for cand in candidates:
                if cand == current_codon:
                    continue

                new_seq = seq[:codon_start] + cand + seq[codon_end:]

                # Check if the original match window still contains this site
                window = new_seq[match.start : match.end]
                if not site.regex.search(window):
                    seq = new_seq
                    site_fixed = True
                    break

            if site_fixed:
                break

        if not site_fixed:
            raise RestrictionSiteFixError(
                f"Unable to remove restriction site {site.name} ({site.pattern}) at positions "
                f"{match.start}-{match.end} without altering the amino acid sequence."
            )



