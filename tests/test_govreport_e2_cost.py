import unittest

from scripts.audit.run_govreport_e2_cost import PLM_SYSTEMS, REPO_ROOT, _load_config, _specs


class GovReportE2CostContractTests(unittest.TestCase):
    def test_semantic_route_selector_comparators_are_cache_aware(self):
        for system in (
            "D2_matched_greedy_tfidf_anchor",
            "D2_matched_nsga2_tfidf",
        ):
            with self.subTest(system=system):
                spec = _specs()[system]
                config = _load_config(REPO_ROOT / spec["config_path"])
                self.assertIn("semantic", config["compute_budget"]["enabled_routes"])
                self.assertIsInstance(config["routes"]["semantic"], dict)
                self.assertIn(system, PLM_SYSTEMS)

    def test_non_embedding_baselines_are_not_cache_aware(self):
        for system in ("lead", "lexrank", "pacsum_tfidf_P07"):
            with self.subTest(system=system):
                self.assertNotIn(system, PLM_SYSTEMS)


if __name__ == "__main__":
    unittest.main()
