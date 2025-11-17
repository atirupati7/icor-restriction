import unittest
from unittest.mock import patch

from tool.optimizers.restriction_site_manager import (
    RestrictionSiteManager,
    fix_restriction_sites,
    RestrictionSiteFixError,
)
from tool.optimizers.icor_optimizer import optimize_sequence


class RestrictionSiteManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = RestrictionSiteManager()

    def test_detect_single_site(self):
        # Contains EcoRI (GAATTC) once
        seq = "ATGGAATTCTAA"
        matches = self.manager.find_sites(seq)
        names = {m.site.name for m in matches}
        self.assertIn("EcoRI", names)
        self.assertEqual(len(matches), 1)

    def test_detect_multiple_and_overlapping_sites(self):
        # Engineered sequence with overlapping XbaI / SpeI-like patterns
        seq = "TCTAGAACTAGT"
        matches = self.manager.find_sites(seq)
        self.assertGreaterEqual(len(matches), 2)

    def test_degenerate_site_detection(self):
        # BstXI: CCANNNNNNTGG – this sequence matches with N=ACGT...
        seq = "ATGCCACCCCCCTGGTAATAG"
        matches = [m for m in self.manager.find_sites(seq) if m.site.name == "BstXI"]
        self.assertGreaterEqual(len(matches), 1)

    def test_fix_single_site_synonymous(self):
        # Simple construct: M + EcoRI + stop
        # AA: M (ATG), then two amino acids encoded by GAATTC, then stop
        aa_seq = "MEX*"  # arbitrary amino acids, length 3
        # Corresponding DNA with EcoRI in positions 3–8
        dna_seq = "ATGGAATTCTAA"
        fixed = fix_restriction_sites(dna_seq, aa_seq, manager=self.manager)
        self.assertNotIn("GAATTC", fixed)
        self.assertFalse(self.manager.has_sites(fixed))
        # Ensure length preserved
        self.assertEqual(len(fixed), len(dna_seq))

    def test_fix_multiple_sites(self):
        # Two EcoRI sites back-to-back, in-frame
        aa_seq = "MEEEEEE*"  # 7 amino acids
        dna_seq = "ATG" + "GAATTC" * 2 + "TAA"
        fixed = fix_restriction_sites(dna_seq, aa_seq, manager=self.manager)
        self.assertFalse(self.manager.has_sites(fixed))

    def test_unfixable_site_raises(self):
        # Construct a pathological case where no synonyms are known for an AA,
        # reusing a placeholder amino acid that is not in the standard table.
        manager = RestrictionSiteManager(extra_sites={"TestSite": "GAATTC"})
        dna = "GAATTC"
        aa = "?" * (len(dna) // 3)
        with self.assertRaises(RestrictionSiteFixError):
            fix_restriction_sites(dna, aa, manager=manager)


class IcorIntegrationTests(unittest.TestCase):
    def test_full_optimization_pipeline_removes_sites(self):
        """
        Integration test: simulate a full ICOR run that initially introduces
        a restriction site and confirm the final output has no forbidden sites.
        The ICOR model call is patched to avoid requiring the ONNX runtime.
        """
        manager = RestrictionSiteManager()

        # Fake model output that includes an EcoRI site
        fake_output_with_site = "ATGGAATTCTAA"  # ATG + GAATTC + TAA

        with patch(
            "tool.optimizers.icor_optimizer._run_icor_model",
            return_value=fake_output_with_site,
        ):
            aa_seq = "MEX*"
            optimized = optimize_sequence(
                sequence=aa_seq,
                sequence_type="AA",
                avoid_restriction_sites=True,
            )

        self.assertNotIn("GAATTC", optimized)
        self.assertFalse(manager.has_sites(optimized))


if __name__ == "__main__":
    unittest.main()


