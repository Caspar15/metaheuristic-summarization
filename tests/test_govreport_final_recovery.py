from scripts.audit.recover_govreport_final_analysis import derive_internal_random_rows
from scripts.audit.run_govreport_final_test import RANDOM_SEEDS


def test_internal_random_recovery_derives_macro_without_source_macro_field():
    seeds = [
        {"row": {"rouge1": float(i), "rouge2": float(i + 1), "rougeLsum": float(i + 2)}}
        for i, _ in enumerate(RANDOM_SEEDS)
    ]
    rows = derive_internal_random_rows(seeds, ["row"])
    assert rows == [
        {"id": "row", "rouge1": 4.5, "rouge2": 5.5, "rougeLsum": 6.5, "macro_rouge": 5.5}
    ]
