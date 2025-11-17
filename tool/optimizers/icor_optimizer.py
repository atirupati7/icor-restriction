"""
ICOR optimizer CLI and programmatic API with restriction site avoidance.
"""

from Bio.Seq import Seq
import sys
import argparse
import onnxruntime as rt
import numpy as np
from typing import List, Tuple
import os

from .restriction_site_manager import RestrictionSiteManager, fix_restriction_sites


# Set path to the ONNX model
model_path = os.path.join(os.getcwd(), "tool", "models", "icor.onnx")


# Define categorical labels from when model was trained.
labels = [
    "AAA",
    "AAC",
    "AAG",
    "AAT",
    "ACA",
    "ACG",
    "ACT",
    "AGC",
    "ATA",
    "ATC",
    "ATG",
    "ATT",
    "CAA",
    "CAC",
    "CAG",
    "CCG",
    "CCT",
    "CTA",
    "CTC",
    "CTG",
    "CTT",
    "GAA",
    "GAT",
    "GCA",
    "GCC",
    "GCG",
    "GCT",
    "GGA",
    "GGC",
    "GTC",
    "GTG",
    "GTT",
    "TAA",
    "TAT",
    "TCA",
    "TCG",
    "TCT",
    "TGG",
    "TGT",
    "TTA",
    "TTC",
    "TTG",
    "TTT",
    "ACC",
    "CAT",
    "CCA",
    "CGG",
    "CGT",
    "GAC",
    "GAG",
    "GGT",
    "AGT",
    "GGG",
    "GTA",
    "TGC",
    "CCC",
    "CGA",
    "CGC",
    "TAC",
    "TAG",
    "TCC",
    "AGA",
    "AGG",
    "TGA",
]


def aa2int(seq: str) -> List[int]:
    """Map amino-acid characters to integer indices used by the ICOR model."""
    _aa2int = {
        "A": 1,
        "R": 2,
        "N": 3,
        "D": 4,
        "C": 5,
        "Q": 6,
        "E": 7,
        "G": 8,
        "H": 9,
        "I": 10,
        "L": 11,
        "K": 12,
        "M": 13,
        "F": 14,
        "P": 15,
        "S": 16,
        "T": 17,
        "W": 18,
        "Y": 19,
        "V": 20,
        "B": 21,
        "Z": 22,
        "X": 23,
        "*": 24,
        "-": 25,
        "?": 26,
    }
    return [_aa2int[i] for i in seq]


def _run_icor_model(amino_acid_sequence: str) -> str:
    """Run the ICOR ONNX model on an amino-acid sequence and return optimized DNA."""
    encoded = aa2int(amino_acid_sequence)
    oh_array = np.zeros(shape=(26, len(amino_acid_sequence)))

    for i, aa_index in enumerate(encoded):
        oh_array[aa_index, i] = 1

    oh_array = [oh_array]
    x = np.array(np.transpose(oh_array))

    y = x.astype(np.float32)
    y = np.reshape(y, (y.shape[0], 1, 26))

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    pred_onx = sess.run(None, {input_name: y})

    pred_indices = [int(np.argmax(pred)) for pred in pred_onx[0]]

    return "".join(labels[index] for index in pred_indices)


def _prepare_amino_acid_sequence(sequence: str, sequence_type: str) -> Tuple[str, str]:
    """
    Validate and, if needed, translate an input sequence into amino acids.

    Returns (amino_acid_sequence, normalized_sequence_type).
    """
    sequence_type = sequence_type.strip().upper()
    seq = sequence.strip().upper()

    if sequence_type == "AA":
        if seq == "DEMO":
            seq = (
                "MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLMKTERPRPNTF"
                "IIRCLQWTTVIERTFHVETPEEREEWTTAIQTVADGLKKQEEEEMDFRSGSPSDNSGAEEMEVSLAKPKHRVTM"
                "NEFEYLKLLGKGTFGKVILVKEKATGRYYAMKILKKEVIVAKDEVAHTLTENRVLQNSRHPFLTALKYSFQTHD"
                "RLCFVMEYANGGELFFHLSRERVFSEDRARFYGAEIVSALDYLHSEKNVVYRDLKLENLMLDKDGHIKITDFGL"
                "CKEGIKDGATMKTFCGTPEYLAPEVLEDNDYGRAVDWWGLGVVMYEMMCGRLPFYNQDHEKLFELILMEEIRFP"
                "RTLGPEAKSLLSGLLKKDPKQRLGGGSEDAKEIMQHRFFAGIVWQHVYEKKLSPPFKPQVTSETDTRYFDEEFT"
                "AQMITITPPDQDDSMECVDSERRPHFPQFSYSASGTA*"
            )
        if not seq.startswith("M") or not seq.endswith("*"):
            sys.exit(
                "Invalid amino acid sequence detected.\n"
                "The sequence must start with M and end with * because ICOR only optimizes the codon-sequence region!\n"
                "Please try again.\nRead more: http://www.hgvs.org/mutnomen/references.html#aalist"
            )
        return seq, "AA"

    if sequence_type == "DNA":
        if seq == "DEMO":
            seq = (
                "ATGAGCGACGTGGCTATTGTGAAGGAGGGTTGGCTGCACAAACGAGGGGAGTACATCAAGACCTGGCGGCCACG"
                "CTACTTCCTCCTCAAGAATGATGGCACCTTCATTGGCTACAAGGAGCGGCCGCAGGATGTGGACCAACGTGAGG"
                "CTCCCCTCAACAACTTCTCTGTGGCGCAGTGCCAGCTGATGAAGACGGAGCGGCCCCGGCCCAACACCTTCATC"
                "ATCCGCTGCCTGCAGTGGACCACTGTCATCGAACGCACCTTCCATGTGGAGACTCCTGAGGAGCGGGAGGAGTG"
                "GACAACCGCCATCCAGACTGTGGCTGACGGCCTCAAGAAGCAGGAGGAGGAGGAGATGGACTTCCGGTCGGGCT"
                "CACCCAGTGACAACTCAGGGGCTGAAGAGATGGAGGTGTCCCTGGCCAAGCCCAAGCACCGCGTGACCATGAAC"
                "GAGTTTGAGTACCTGAAGCTGCTGGGCAAGGGCACTTTCGGCAAGGTGATCCTGGTGAAGGAGAAGGCCACAGG"
                "CCGCTACTACGCCATGAAGATCCTCAAGAAGGAAGTCATCGTGGCCAAGGACGAGGTGGCCCACACACTCACCG"
                "AGAACCGCGTCCTGCAGAACTCCAGGCACCCCTTCCTCACAGCCCTGAAGTACTCTTTCCAGACCCACGACCGC"
                "CTCTGCTTTGTCATGGAGTACGCCAACGGGGGCGAGCTGTTCTTCCACCTGTCCCGGGAGCGTGTGTTCTCCGA"
                "GGACCGGGCCCGCTTCTATGGCGCTGAGATTGTGTCAGCCCTGGACTACCTGCACTCGGAGAAGAACGTGGTGT"
                "ACCGGGACCTCAAGCTGGAGAACCTCATGCTGGACAAGGACGGGCACATTAAGATCACAGACTTCGGGCTGTGC"
                "AAGGAGGGGATCAAGGACGGTGCCACCATGAAGACCTTTTGCGGCACACCTGAGTACCTGGCCCCCGAGGTGCT"
                "GGAGGACAATGACTACGGCCGTGCAGTGGACTGGTGGGGGCTGGGCGTGGTCATGTACGAGATGATGTGCGGTC"
                "GCCTGCCCTTCTACAACCAGGACCATGAGAAGCTTTTTGAGCTCATCCTCATGGAGGAGATCCGCTTCCCGCGC"
                "ACGCTTGGTCCCGAGGCCAAGTCCTTGCTTTCAGGGCTGCTCAAGAAGGACCCCAAGCAGAGGCTTGGCGGGGG"
                "CTCCGAGGACGCCAAGGAGATCATGCAGCATCGCTTCTTTGCCGGTATCGTGTGGCAGCACGTGTACGAGAAGA"
                "AGCTCAGCCCACCCTTCAAGCCCCAGGTCACGTCGGAGACTGACACCAGGTATTTTGATGAGGAGTTCACGGCC"
                "CAGATGATCACCATCACACCACCTGACCAAGATGACAGCATGGAGTGTGTGGACAGCGAGCGCAGGCCCCACTT"
                "CCCCCAGTTCTCCTACTCGGCCAGCGGCACGGCCTGA"
            )
        if "U" in seq:
            sys.exit(
                "Invalid DNA sequence detected.\n"
                'The sequence must be in DNA form. A "U" was found in your sequence.\n'
                "Please try again.\nRead more: http://www.hgvs.org/mutnomen/references.html#aalist"
            )
        if not seq.startswith("ATG"):
            sys.exit(
                "Invalid DNA sequence detected.\n"
                "The sequence must start with ATG because ICOR only optimizes the codon-sequence region! Please try again.\n"
                "Read more: http://www.hgvs.org/mutnomen/references.html#aalist"
            )
        if not (seq.endswith("TAA") or seq.endswith("TGA") or seq.endswith("TAG")):
            sys.exit(
                "Invalid DNA sequence detected.\n"
                "The sequence must end with a stop codon of TAA, TGA, or TAG because ICOR only optimizes the codon-sequence region! Please try again.\n"
                "Read more: http://www.hgvs.org/mutnomen/references.html#aalist"
            )

        aa_seq = str(Seq(seq).translate())
        return aa_seq, "DNA"

    sys.exit(f"Invalid sequence type {sequence_type}. Expected 'aa' or 'dna'")


def optimize_sequence(
    sequence: str,
    sequence_type: str = "AA",
    avoid_restriction_sites: bool = True,
) -> str:
    """
    Programmatic API for ICOR codon optimization.

    Parameters
    ----------
    sequence:
        Input sequence, either amino acid (including leading M and trailing *)
        or DNA (including start codon and stop codon).
    sequence_type:
        'AA' for amino acid input or 'DNA' for nucleotide input.
    avoid_restriction_sites:
        If True (default), automatically remove common restriction sites from
        the optimized sequence using synonymous codon substitutions.

    Returns
    -------
    str
        Optimized DNA sequence.
    """
    amino_acid_sequence, _ = _prepare_amino_acid_sequence(sequence, sequence_type)
    optimized_dna = _run_icor_model(amino_acid_sequence)

    if avoid_restriction_sites:
        manager = RestrictionSiteManager()
        optimized_dna = fix_restriction_sites(
            optimized_dna, amino_acid_sequence, manager=manager
        )

    return optimized_dna


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ICOR codon optimizer. Accepts an amino acid or DNA sequence and "
            "returns an optimized DNA sequence. Restriction site avoidance is "
            "enabled by default."
        )
    )
    parser.add_argument(
        "--avoid-restriction-sites",
        dest="avoid_restriction_sites",
        action="store_true",
        help="Enable restriction site avoidance (default).",
    )
    parser.add_argument(
        "--no-avoid-restriction-sites",
        dest="avoid_restriction_sites",
        action="store_false",
        help="Disable restriction site avoidance.",
    )
    parser.set_defaults(avoid_restriction_sites=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    sequence_type = input(
        "Welcome to ICOR! Are you optimizing an amino acid sequence (enter in 'aa' below) or a dna/codon sequence (enter in 'dna' below)?\n\n"
    ).strip()
    input_seq = input(
        "Enter the coding sequence only.\nEnter in 'demo' to use demo sequence.\n\n"
    ).strip()

    optimized = optimize_sequence(
        input_seq,
        sequence_type=sequence_type,
        avoid_restriction_sites=args.avoid_restriction_sites,
    )

    print("==== OUTPUT ====\n" + optimized)

    output = input(
        "Would you like to write this into a file? (Y or N)\n\n"
    ).strip().upper()

    while True:
        if output == "Y":
            with open("output.txt", "w") as f:
                f.write(optimized)
                print("\nOutput written to output.txt")
            break
        elif output == "N":
            print("\nNo output written. Done!")
            break
        else:
            print("Error! Expected Y/N")
            break


if __name__ == "__main__":
    main()


