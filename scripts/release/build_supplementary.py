"""Render the English supplement from archived analyses; never run experiments.

Requires reportlab. The existing analyses remain authoritative and unchanged.
Outputs PDF, an English Markdown companion, and a source-hash manifest.
"""
from pathlib import Path
from html import escape
import hashlib
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission"
TAG = "v1.0.0-ieee-access"
SOURCES = {}
LABELS = {"govreport": "GovReport", "multinews": "Multi-News"}
METRICS = {"rouge1": "R-1", "rouge2": "R-2", "rougeL": "R-L", "rougeLsum": "R-Lsum", "macro_rouge": "Mean", "macro": "Mean"}
SYSTEMS = {"proposed": "PAMR-ES", "lead": "Lead", "random_mean_10_seeds": "Random (10 seeds)", "textrank": "TextRank", "lexrank": "LexRank", "sbert_centroid": "SBERT centroid"}


def load(path):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def name(key):
    if key in SYSTEMS:
        return SYSTEMS[key]
    if key.startswith("pacsum_tfidf"):
        return "PacSum-TFIDF"
    if key.startswith("pacsum_sbert"):
        return "PacSum-SBERT"
    if key.startswith("sbert_mmr") or key.startswith("full_source_sbert_mmr"):
        return "SBERT+MMR"
    return key


def f(value):
    return f"{value:.6f}"


def paired_rows(comparisons, adjusted):
    return [[METRICS[k], f(v["mean_difference"]),
             f"[{v['ci_lower']:+.6f}, {v['ci_upper']:+.6f}]",
             f(v["p_value_two_sided"]), f(v.get(adjusted, v["p_value_two_sided"]))]
            for k, v in comparisons.items()]


def build():
    OUT.mkdir(exist_ok=True)
    # Use embedded TrueType fonts when available; Liberation is the Linux alternative.
    candidates = [(Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
                  (Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
                   Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"))]
    regular, bold = next((a, b) for a, b in candidates if a.exists() and b.exists())
    pdfmetrics.registerFont(TTFont("Body", str(regular)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(bold)))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyTextCustom", fontName="Body", fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle("CellCustom", fontName="Body", fontSize=8.2, leading=10.5, splitLongWords=True))
    styles.add(ParagraphStyle("TitleCustom", fontName="BodyBold", fontSize=20, leading=25, spaceAfter=16))
    styles.add(ParagraphStyle("HeadingCustom", fontName="BodyBold", fontSize=14, leading=18, spaceAfter=10, keepWithNext=True))
    styles.add(ParagraphStyle("SubCustom", fontName="BodyBold", fontSize=10.5, leading=14, spaceBefore=7, spaceAfter=7, keepWithNext=True))
    story, markdown = [], []
    width = A4[0] - 96
    table_number = 0

    def p(text):
        story.append(Paragraph(escape(text), styles["BodyTextCustom"]))
        markdown.append(text + "\n")

    def heading(text, page=True):
        if page and story: story.append(PageBreak())
        story.append(Paragraph(escape(text), styles["HeadingCustom"]))
        markdown.append("## " + text + "\n")

    def sub(text):
        story.append(Paragraph(escape(text), styles["SubCustom"]))
        markdown.append("### " + text + "\n")

    def table(caption, headers, rows, widths=None):
        nonlocal table_number
        table_number += 1
        sub(f"Table S{table_number}. {caption}")
        data = [[Paragraph(escape(str(x)), styles["CellCustom"]) for x in row] for row in [headers] + rows]
        t = Table(data, colWidths=[width * w for w in widths] if widths else [width / len(headers)] * len(headers), repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#687889")),
            ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#d6dce3")),
        ]))
        story.extend([t, Spacer(1, 9)])
        markdown.append("| " + " | ".join(headers) + " |")
        markdown.append("| " + " | ".join(["---"] * len(headers)) + " |")
        markdown.extend("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows)
        markdown.append("")

    story.append(Paragraph("Supplementary Material", styles["TitleCustom"]))
    markdown.append("# Supplementary Material\n")
    p("Provenance-Aware Multi-Route Fusion for Long-Document Extractive Summarization Without Task-Specific Fine-Tuning")
    p("Shih-Wei Yang, Bo-Yu Chen, Shao-Chi Kuan, Sy-Yen Kuo, and Jiann-Liang Chen")
    p(f"PAMR-ES | IEEE Access submission companion | Version {TAG}")
    heading("S1. Scope, protocols and final configuration", page=False)
    p("This supplement reports archived experimental evidence and details needed to inspect the final method. It adds no new model training, test prediction, evaluator run or configuration selection. Prespecified development ablations and post-test development diagnostics are reported as separate analysis families. Neither group replaces the frozen official test result.")
    p("Official quality tables use Stanza tokenize,mwt followed by Perl ROUGE-1.5.5. Development selector/ablation tables use the internal multi-sentence ROUGE-Lsum protocol. Scores are on the 0-1 scale. Mean denotes the arithmetic mean of the three stated ROUGE metrics. A corpus-score difference need not equal a mean paired difference because the archived official corpus outputs and per-document scores have different rounding/aggregation paths.")
    table("Final method settings", ["Setting", "GovReport", "Multi-News"], [
        ["Official test / frozen development rows", "973 / 681", "5,621 / 3,935"],
        ["Routes", "Lexical, semantic, sparse graph", "Lexical, semantic, sparse graph"],
        ["Per-route proposal limit / reservation", "40 / 20", "40 / 20"],
        ["Candidate-pool cap / RRF constant", "80 / 60", "80 / 60"],
        ["RRF weights: lexical, semantic, graph", "0.5, 1, 1", "1, 1, 2"],
        ["Final selector", "TF-IDF MMR, lambda=0.7", "Greedy-TFIDF"],
        ["Requested summary words", "500-650", "200-250"],
        ["Semantic encoder / input limit", "all-MiniLM-L6-v2 / 256 tokens", "Same"],
        ["Task-specific fine-tuning", "None", "None"],
    ], [.40, .30, .30])
    p("The semantic encoder is pinned to revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf. Route weights and selectors were selected using development data and fixed before test execution. This is not a claim of zero configuration tuning. Full settings, policy identities and environment versions are in the two configs/final YAML files and their execution freezes.")
    for dataset in LABELS:
        for path in [f"configs/final/{dataset}_final_v1.yaml", f"configs/preregistrations/{dataset}_final_execution_freeze_v1.json"]:
            SOURCES[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

    heading("S2. Official test results and paired comparisons")
    finals = {d: load(f"runs_v2/{d}_final_test_v1/analysis.json") for d in LABELS}
    for d, a in finals.items():
        table(f"{LABELS[d]} official quality ({a['rows']:,} rows)", ["System", "R-1", "R-2", "R-L", "Mean"],
              [[name(r['system']), f(r['rouge1']), f(r['rouge2']), f(r['rougeL']), f(r['macro_rouge'])] for r in a['corpus_ranking']], [.32, .17, .17, .17, .17])
    p("The Random row averages the ten fixed seeds; it is not the best seed. All system outputs are retained for the primary all-row analysis, including length shortfalls.")
    heading("S2.1 Prespecified paired comparisons")
    for d, a in finals.items():
        primary = a[next(k for k in a if k.startswith('primary_proposed_vs_'))]
        rows = paired_rows(primary['components'], 'p_value_holm_3') + paired_rows({'macro': primary['macro']}, 'p_value_two_sided')
        table(f"{LABELS[d]}: PAMR-ES minus {'SBERT+MMR' if d == 'govreport' else 'PacSum-TFIDF'}", ['Metric', 'Difference', '95% paired CI', 'Raw p', 'Reported p'], rows, [.12,.17,.35,.18,.18])
    p("Each comparison uses 100,000 paired bootstrap resamples. Component p-values use Holm-3; the prespecified Mean endpoint reports its two-sided p-value. GovReport supports the scoped Mean advantage; its R-L component is not significant. Multi-News does not show a significant overall Mean advantage, with higher R-1/R-L and lower R-2. Full exploratory baseline comparisons and their Holm-32 family remain in the linked final analyses.")

    supplement = load('runs_v2/manuscript_supplemental_analysis_v1/analysis.json')
    heading("S3. Output length and feasibility")
    p("Values are means with population standard deviations. Random pools document-seed observations over ten fixed seeds; the remaining systems pool documents. Infeasible means the requested length contract was not satisfied, not that the output was omitted from evaluation.")
    for d, a in supplement['output_length'].items():
        rows = [[('PAMR-ES' if s == 'Proposed' else s), f"{v['mean_words']:.2f} +/- {v['sd_words_population']:.2f}", f"{v['mean_sentences']:.2f} +/- {v['sd_sentences_population']:.2f}", str(v['infeasible_observations'])] for s,v in a['systems'].items()]
        table(f"{LABELS[d]} output statistics", ['System', 'Words: mean +/- SD', 'Sentences: mean +/- SD', 'Shortfalls'], rows, [.30,.26,.28,.16])
    p("PAMR-ES does not produce systematically longer summaries than all strong baselines. Its comparative R-1/R-L performance cannot be attributed simply to filling more of the permitted word budget.")

    variants = {'A01_no_semantic':'A01: no semantic route', 'A02_no_graph':'A02: no graph route', 'A03_lexical_only_capacity_80':'A03: lexical only, capacity 80', 'A04_exact_pool_equal_rrf':'A04: exact pool, equal-weight RRF', 'A05_exact_pool_lexical_salience':'A05: exact pool, lexical salience'}
    def variant_name(k):
        return variants.get(k, {'A03':'A03: lexical only, capacity 80', 'A04':'A04: exact pool, equal-weight RRF', 'A05':'A05: exact pool, lexical salience'}.get(k[:3],k))
    for d in LABELS:
        heading(f"S4. Prespecified ablations: {LABELS[d]}")
        a = load(f'runs_v2/{d}_e3_route_provenance_v1/analysis.json')
        study = load(f'runs_v2/{d}_e3_route_provenance_v1/study_summary.json')
        p(f"Frozen development only ({a['rows']:,} rows); internal ROUGE-Lsum protocol. Each difference is full PAMR-ES minus the stated ablation. The five variants and four endpoints form a dataset-specific Holm-20 family with 100,000 paired bootstrap resamples.")
        rows = [['Full', f(a['anchor_metrics']['macro_rouge'])]]
        rows += [[variant_name(k),f(v['metrics']['macro_rouge'])] for k,v in study['variants'].items()]
        table('Mean ROUGE by variant', ['Variant','Mean'], rows, [.76,.24])
        rows = []
        for key, comps in a['comparisons_full_minus_ablation'].items():
            rows += [[key[:3]] + row for row in paired_rows(comps,'p_value_holm_20')]
        table('Complete prespecified paired endpoints', ['Variant','Metric','Difference','95% CI','Raw p','Holm-20 p'], rows, [.09,.11,.15,.33,.16,.16])
        p("A01/A02 remove one candidate-generation route. A03 retains lexical-only candidates at the matched capacity. A04 and A05 fix full-system candidate membership, changing only route weighting or selector salience. These contrasts support the studied semantic/graph and fusion-ranking effects within the frozen configuration; they do not establish that every route or reservation mechanism independently improves quality.")

    heading('S5. Post-test development mechanism diagnostics')
    p("The following diagnostics were registered after final testing, before scoring their respective new variants. They use only frozen development membership and do not access dev-test/test for method selection. They are separate from the prespecified E3 family and did not alter the final system.")
    for identifier, title in [('postfreeze_no_reservation_v1','No route reservation'),('postfreeze_no_lexical_route_v1','No lexical candidate route')]:
        a = load(f'runs_v2/{identifier}/analysis.json')
        rows = []
        for d, v in a['datasets'].items():
            rows += [[LABELS[d]] + row for row in paired_rows(v['comparisons'],'p_value_holm_4')]
        table(f'{title}: full minus variant', ['Dataset','Metric','Difference','95% CI','Raw p','Holm-4 p'], rows, [.15,.10,.15,.30,.15,.15])
    p("Removing reservation changes min_per_route from 20 to zero with other settings held fixed. No multiplicity-corrected quality gain for retaining reservation is established. It is therefore described as a source-balancing and audit mechanism. Removing the lexical candidate route improves development Mean ROUGE in both datasets. TF-IDF still supplies selector features, so this is not removal of all lexical information.")

    heading('S5.1 Exact-pool zero lexical ranking weight')
    p("Candidate membership is fixed to the full system for every document. The lexical RRF weight alone is set to zero. Contrast A is full minus exact-pool zero-weight; contrast B is exact-pool zero-weight minus no lexical candidate route. The 16 endpoints across both datasets use global Holm-16 correction and 100,000 paired bootstrap resamples.")
    a = load('runs_v2/postfreeze_zero_lexical_weight_v1/analysis.json')
    for d,v in a['datasets'].items():
        rows = []
        for letter, (_, comps) in zip(['A','B'], v['comparisons'].items()):
            rows += [[letter] + row for row in paired_rows(comps,'p_value_holm_16_global')]
        table(f'{LABELS[d]} exact-pool contrasts', ['Contrast','Metric','Difference','95% CI','Raw p','Holm-16 p'], rows, [.10,.10,.16,.32,.16,.16])
    p("In GovReport, the negative lexical effect is associated primarily with its ranking vote; retaining lexical-proposed pool members has no detectable additional effect in this contrast. Multi-News shows effects of both the ranking vote and candidate membership. These are configuration-specific mechanism diagnostics, not evidence for a newly selected final method.")

    heading('S6. Matched-row sensitivity and configuration budget')
    sens = supplement['multinews_feasible_row_sensitivity']
    p("The 12 PAMR-ES shortfalls in Multi-News arise from whole-sentence packing under the 250-word ceiling. Their eligible source capacity exceeds 200 words; they are not short source documents. SBERT+MMR has seven shortfalls, six overlapping PAMR-ES. Actual outputs remain in the primary 5,621-row evaluation.")
    p(f"The sensitivity analysis removes the same {sens['excluded_rows']} PAMR-ES-shortfall IDs from both PAMR-ES and PacSum-TFIDF, leaving {sens['matched_rows']:,} paired rows. Only existing per-document scores are reaggregated; summaries are not regenerated.")
    table('Matched-row PAMR-ES minus PacSum-TFIDF', ['Metric','Difference','95% CI','Raw p','Reported p'], paired_rows(sens['proposed_minus_pacsum_tfidf'],'p_value_holm_3'), [.12,.17,.35,.18,.18])
    p("The component endpoints use Holm-3 and Mean reports its two-sided p-value. The conclusion is unchanged: no significant overall Mean difference, higher R-1/R-L, and lower R-2. Excluded IDs: " + ', '.join(sens['infeasible_ids']) + '.')
    budget = supplement['configuration_budget']
    rows = [[k,str(v),str(budget['baseline_validation_candidates']['Multi-News'][k])] for k,v in budget['baseline_validation_candidates']['GovReport'].items()]
    rows.append(['PAMR-ES: unique development config hashes',str(budget['proposed_development_program']['GovReport']['unique_config_hashes']),str(budget['proposed_development_program']['Multi-News']['unique_config_hashes'])])
    rows.append(['PAMR-ES: held-out dev-test score observations','4','4'])
    table('Configuration comparisons', ['Method / measure','GovReport','Multi-News'], rows, [.50,.25,.25])
    p("These counts describe configuration selection, not neural fine-tuning. The PAMR-ES program includes length, capacity, route, selector and combination studies that could inform its final profile; it excludes E3, cost, oracle and engineering checks. Methods did not receive identical search budgets. Search records and frozen data partitions make that scope auditable.")

    heading('S7. Fixed-rule decision-provenance example')
    case = load('runs_v2/manuscript_supplemental_analysis_v1/provenance_case.json')
    p(f"The example is {case['example_id']}, the first document in the frozen GovReport development prediction order, selected without consulting ROUGE or qualitative favorability. The summary has {case['summary_words']} words and {case['selected_sentence_count']} sentences. All selected sentences are provided below, followed by the first five reserved but unselected candidates in record order. Indices refer to the original canonical sentence order.")
    p("L/S/G denote lexical/semantic/graph route ranks. Retention reasons distinguish route nomination, reservation, guard or fusion fill from final selection. This example demonstrates a traceable selection record, not factuality or causal explanation. Source text is from the GovReport CRS document identified above; it is not original prose by the authors.")
    for group, title in [('selected_sentences','Selected sentences'),('first_five_reserved_not_selected','Reserved candidates not selected')]:
        sub(title)
        for item in case[group]:
            ranks=item['route_ranks']
            p(f"Sentence index {item['original_index']} | L/S/G ranks {ranks.get('lexical','-')}/{ranks.get('semantic','-')}/{ranks.get('graph','-')} | fusion rank {item['fusion_rank']} | route agreement {item.get('route_agreement','-')} | retained: {', '.join(item['retention_reasons'])}")
            p(item['text'])
            story[-2:] = [KeepTogether(story[-2:])]

    heading('S8. Historical selector comparison and controlled cost')
    p("NSGA-II is a historical comparator, not a component of final PAMR-ES. The development selector comparison precedes final fusion configuration and holds candidate pools, TF-IDF similarities, length constraints and random seeds fixed. The shared comparison uses equal route weights and zero positional weight; its scores must not be substituted for the final development configuration.")
    d2=load('runs_v2/d2_selector_full_dev_v1/analysis/paired_summary.json')
    rows=[]
    for d,v in d2['datasets'].items():
        for key, value in v['macro_means'].items():
            selected_mmr = 'S05_' if d == 'govreport' else 'S03_'
            if key.startswith(('S00_', selected_mmr, 'S12_')):
                rows.append([LABELS[d], key, f(value)])
    table('Archived matched selector scores (candidate IDs preserve provenance)', ['Dataset','Candidate','Mean'], rows, [.20,.58,.22])
    p("The complete selector candidates and paired analyses remain in the D2 source JSON. NSGA-II was not the highest-quality selector on either dataset. The final selectors are MMR for GovReport and Greedy for Multi-News.")
    rows=[]
    for d in LABELS:
        a=load(f'runs_v2/{d}_cost_scaling_v1/analysis.json')
        for key,v in a['systems'].items():
            if key not in ['frozen_C01_proposed','D2_matched_nsga2_tfidf','matched_nsga2_tfidf']: continue
            c,w=v['cold']['total_run'],v['warm_cache']['total_run']
            rows.append([LABELS[d], 'Final PAMR-ES' if key=='frozen_C01_proposed' else 'NSGA-II comparator', f"{c['wall_seconds']['median']:.2f}", f"{w['wall_seconds']['median']:.2f}", f"{w['wall_seconds']['median']/30:.3f}", f"{c['peak_process_tree_rss_bytes']['median']/2**20:.1f} / {w['peak_process_tree_rss_bytes']['median']/2**20:.1f}"])
    table('Controlled CPU cost: medians of three measured repetitions', ['Dataset','System','Cold total s','Warm total s','Warm s/doc','Cold/warm RSS MiB'], rows, [.15,.21,.14,.14,.14,.22])
    p("Times are totals for the fixed 30-document sample, with a derived warm per-document average. Sampling is reference-blind across the 10th, 50th and 90th percentiles of log(1 + source sentence count), ten documents per stratum. Cold and warm-cache runs use fresh subprocesses, three measured repetitions, and process-tree RSS. Full system-level quartiles, CPU measurements, stratum measurements and descriptive scaling slopes are in the cost-analysis JSON files. Hardware timing is not a cross-machine speed claim.")

    heading('S9. Artifact access and source index')
    p(f"Code, configurations and compact evidence are versioned at https://github.com/Caspar15/metaheuristic-summarization/tree/{TAG}. The matching release provides this supplement and the numerical-score/selected-index artifact. Raw input documents and reference summaries remain with their original providers; the artifact does not relicense or redistribute full source corpora.")
    p("The source package includes a standard-library snapshot verifier and result-table export. The numerical artifact includes a content manifest with per-file hashes and original-source hashes. Full output regeneration additionally requires the frozen canonical data, pinned semantic and Stanza model assets, and the specified Perl evaluator environment. Source-only verification is distinct from rerunning both full benchmarks.")
    p("Tables in this supplement are generated from archived JSON. Decimal presentation is rounded; machine-readable source values retain full precision. The following source paths and the companion source manifest identify the evidence used. Complete configuration paths are recorded by the individual execution and variant evidence files.")
    for path in sorted(SOURCES): p(path)
    p("AI assistance was used to organize and translate the supplementary text and implement its evidence-to-table formatting. No new experimental results were generated for this supplement; the authors retain responsibility for verification and the final content.")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Body',8)
        canvas.setFillColor(colors.HexColor('#526171'))
        canvas.drawString(48, 27, 'PAMR-ES | Supplementary Material')
        canvas.drawRightString(A4[0]-48,27,f'S{doc.page}')
        canvas.restoreState()

    doc=SimpleDocTemplate(str(OUT/'PAMR_ES_Supplementary_Material.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=45,title='PAMR-ES Supplementary Material',author='PAMR-ES authors')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'Supplementary_Material.md').write_text('\n'.join(markdown).rstrip()+'\n',encoding='utf-8',newline='\n')
    (OUT/'supplement_sources.json').write_text(json.dumps({'release_tag':TAG,'new_experiments':False,'table_count':table_number,'sources':SOURCES},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'tables':table_number,'sources':len(SOURCES),'output':str(OUT/'PAMR_ES_Supplementary_Material.pdf')}))


if __name__ == '__main__':
    build()
