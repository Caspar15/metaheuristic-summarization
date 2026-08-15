import pytest

from scripts.audit.run_govreport_official_evaluator import parse_rouge_output


SAMPLE = """\
1 ROUGE-1 Average_R: 0.50000 (95%-conf.int. 0.40000 - 0.60000)
1 ROUGE-1 Average_P: 0.40000 (95%-conf.int. 0.30000 - 0.50000)
1 ROUGE-1 Average_F: 0.44444 (95%-conf.int. 0.35000 - 0.53000)
1 ROUGE-1 Eval 1.1 R:0.50000 P:0.40000 F:0.44444
1 ROUGE-2 Average_F: 0.22222 (95%-conf.int. 0.10000 - 0.30000)
1 ROUGE-2 Eval 1.1 R:0.25000 P:0.20000 F:0.22222
1 ROUGE-L Average_F: 0.33333 (95%-conf.int. 0.20000 - 0.40000)
1 ROUGE-L Eval 1.1 R:0.37500 P:0.30000 F:0.33333
"""


def test_parse_official_rouge_average_and_per_eval():
    average, per_eval = parse_rouge_output(SAMPLE)
    assert average == {"rouge1": 0.44444, "rouge2": 0.22222, "rougeL": 0.33333}
    assert per_eval == {
        1: {"rouge1": 0.44444, "rouge2": 0.22222, "rougeL": 0.33333}
    }


def test_parse_official_rouge_fails_on_partial_corpus_output():
    with pytest.raises(ValueError, match="all corpus"):
        parse_rouge_output("1 ROUGE-1 Average_F: 0.5")
