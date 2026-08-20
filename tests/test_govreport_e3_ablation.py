import copy
import unittest

from scripts.audit.run_govreport_e3_ablation import _validated_dev_partition
from src.data.partitions import selected_ids_sha256


class GovReportE3PartitionContractTests(unittest.TestCase):
    def _study(self):
        ids = ["validation_001", "validation_002"]
        return {
            "manifest_object": {
                "partitions": {
                    "dev": {
                        "rows": len(ids),
                        "selected_ids": ids,
                        "selected_ids_sha256": selected_ids_sha256(ids),
                    }
                }
            }
        }

    def test_reads_identity_from_manifest_dev_object(self):
        partition, ids = _validated_dev_partition(self._study())
        self.assertEqual(ids, ["validation_001", "validation_002"])
        self.assertEqual(partition["rows"], 2)

    def test_rejects_membership_digest_drift(self):
        study = copy.deepcopy(self._study())
        study["manifest_object"]["partitions"]["dev"]["selected_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "membership drifted"):
            _validated_dev_partition(study)

    def test_rejects_row_count_drift(self):
        study = copy.deepcopy(self._study())
        study["manifest_object"]["partitions"]["dev"]["rows"] = 3
        with self.assertRaisesRegex(ValueError, "row count drifted"):
            _validated_dev_partition(study)


if __name__ == "__main__":
    unittest.main()
