"""
Detailed Query-by-Query Comparison

Shows exactly what each parser extracted for every query.
Useful for understanding differences and debugging.

"""

from src.property_search_nlp.parser import PropertySearchParser
from src.property_search_nlp.llm_parser import LLMPropertyParser
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import json
from typing import Dict, List

def load_test_cases(path: str = "examples/test_cases.json"):
    """Load queries and ground truths from a JSON file."""
    base = Path(__file__).resolve().parent
    json_path = base / path

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries: List[str] = []
    truths: List[Dict] = []

    for item in data:
        queries.append(item["query"])
        truths.append(item.get("truth", {}))

    return queries, truths

def _values_match(field_key: str, pred, truth) -> bool:
    """Robust comparison between prediction and ground truth."""
    if truth is None:
        # No label for this field -> don't count as wrong
        return True

    if pred is None:
        return False

    # Budget: numeric comparison
    if field_key == "budget":
        try:
            return abs(float(pred) - float(truth)) < 1e-6
        except (TypeError, ValueError):
            return False

    # Features: compare as case-insensitive sets
    if field_key == "features":
        pred_list = pred or []
        truth_list = truth or []
        pred_set = {str(s).strip().lower() for s in pred_list}
        truth_set = {str(s).strip().lower() for s in truth_list}
        return pred_set == truth_set

    # Strings: case-insensitive
    if isinstance(truth, str):
        return str(pred).strip().lower() == truth.strip().lower()

    # Fallback
    return pred == truth

def compare_single_query(query: str, truth: Dict, rule_parser, llm_parser):
    """Compare both parsers on a single query against ground truth."""
    print("=" * 100)
    print(f"Query: '{query}'")
    print("=" * 100)

    rule_result = rule_parser.parse(query)
    llm_result = llm_parser.parse(query)

    fields = [
        ("Property Type", "property_type"),
        ("Location", "location"),
        ("Budget", "budget"),
        ("Budget Type", "budget_type"),
        ("Bedrooms", "bedrooms"),
        ("Bathrooms", "bathrooms"),
        ("Parking", "parking"),
        ("Features", "features"),
        ("Confidence", "confidence_score"),
    ]

    comparison = []

    def fmt(val, key):
        if val is None:
            return "—"
        if key == "budget":
            return f"${val:,.0f}"
        if key == "features":
            return ", ".join(val) if val else "—"
        if key == "confidence_score":
            return f"{val:.1f}%"
        return str(val)

    for field_name, field_key in fields:
        rule_raw = getattr(rule_result, field_key)
        llm_raw = getattr(llm_result, field_key)
        truth_raw = truth.get(field_key)

        truth_val = fmt(truth_raw, field_key) if truth_raw is not None else "—"
        rule_val = fmt(rule_raw, field_key)
        llm_val = fmt(llm_raw, field_key)

        rule_ok = "✅" if _values_match(field_key, rule_raw, truth_raw) else "❌"
        llm_ok = "✅" if _values_match(field_key, llm_raw, truth_raw) else "❌"

        comparison.append([
            field_name,
            truth_val,
            rule_val,
            llm_val,
            f"{rule_ok} / {llm_ok}",
        ])

    comparison.append(["-" * 20, "-" * 20, "-" * 20, "-" * 20, "-" * 5])
    comparison.append([
        "Processing Time",
        "—",
        f"{0.02:.2f}ms",
        f"{llm_result.processing_time_ms:.2f}ms",
        "—",
    ])
    comparison.append([
        "Cost",
        "—",
        "$0.00",
        f"${llm_result.cost_usd:.6f}",
        "—",
    ])
    comparison.append([
        "Tokens Used",
        "—",
        "0",
        str(llm_result.tokens_used),
        "—",
    ])

    headers = ["Field", "Ground Truth", "Rule-Based", "LLM (GPT)", "Correct (Rule / LLM)"]
    print(tabulate(comparison, headers=headers, tablefmt="grid"))
    print()

    return rule_result, llm_result

def save_detailed_log(queries, truths, results, filename="detailed_comparison_log.txt"):
    """Save detailed comparison (rule vs LLM vs ground truth) to file."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("DETAILED QUERY-BY-QUERY COMPARISON LOG\n")
        f.write("=" * 100 + "\n\n")

        fields = [
            ("Property Type", "property_type"),
            ("Location", "location"),
            ("Budget", "budget"),
            ("Budget Type", "budget_type"),
            ("Bedrooms", "bedrooms"),
            ("Bathrooms", "bathrooms"),
            ("Parking", "parking"),
            ("Features", "features"),
            ("Confidence", "confidence_score"),
        ]

        def fmt(val, key):
            if val is None:
                return "—"
            if key == "budget":
                return f"${val:,.0f}"
            if key == "features":
                return ", ".join(val) if val else "—"
            if key == "confidence_score":
                return f"{val:.1f}%"
            return str(val)

        for i, (query, truth, rule_res, llm_res) in enumerate(results, 1):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"Query #{i}: '{query}'\n")
            f.write(f"{'=' * 100}\n\n")

            f.write(
                f"{'Field':<20} {'Ground Truth':<30} "
                f"{'Rule-Based':<30} {'LLM-Based':<30} {'Correct (R/L)'}\n"
            )
            f.write(f"{'-' * 120}\n")

            for field_name, field_key in fields:
                truth_raw = truth.get(field_key)
                rule_raw = getattr(rule_res, field_key)
                llm_raw = getattr(llm_res, field_key)

                truth_val = fmt(truth_raw, field_key) if truth_raw is not None else "—"
                rule_val = fmt(rule_raw, field_key)
                llm_val = fmt(llm_raw, field_key)

                rule_ok = "✅" if _values_match(field_key, rule_raw, truth_raw) else "❌"
                llm_ok = "✅" if _values_match(field_key, llm_raw, truth_raw) else "❌"

                f.write(
                    f"{field_name:<20} "
                    f"{truth_val:<30} "
                    f"{rule_val:<30} "
                    f"{llm_val:<30} "
                    f"{rule_ok}/{llm_ok}\n"
                )

            # Performance metrics section
            f.write(f"\n{'Performance Metrics':<20}\n")
            f.write(f"{'-' * 85}\n")
            f.write(
                f"{'Time':<20} "
                f"{'~0.02ms':<30} "
                f"{f'{llm_res.processing_time_ms:.2f}ms':<30}\n"
            )
            f.write(
                f"{'Cost':<20} "
                f"{'$0.00':<30} "
                f"{f'${llm_res.cost_usd:.6f}':<30}\n"
            )
            f.write(
                f"{'Tokens':<20} "
                f"{'0':<30} "
                f"{str(llm_res.tokens_used):<30}\n"
            )
            f.write("\n")

    print(f"\n✅ Detailed log saved to: {filename}")

def analyze_differences(results):
    """Analyze accuracy of each parser vs ground truth."""

    print("\n" + "=" * 100)
    print("GROUND TRUTH ACCURACY ANALYSIS")
    print("=" * 100 + "\n")

    # Fields to score
    fields = [
        "property_type",
        "location",
        "budget",
        "budget_type",
        "bedrooms",
        "bathrooms",
        "parking",
        "features",
    ]

    stats = {
        "rule_correct": {field: 0 for field in fields},
        "llm_correct": {field: 0 for field in fields},
        "labeled": {field: 0 for field in fields},
    }

    # Per-query scoring
    for query, truth, rule_res, llm_res in results:
        for field in fields:
            truth_val = truth.get(field)
            if truth_val is None:
                continue  # no label → skip this field for this query

            stats["labeled"][field] += 1

            rule_val = getattr(rule_res, field)
            llm_val = getattr(llm_res, field)

            if _values_match(field, rule_val, truth_val):
                stats["rule_correct"][field] += 1
            if _values_match(field, llm_val, truth_val):
                stats["llm_correct"][field] += 1

    # Print per-field accuracy
    print("Accuracy by field (vs ground truth):")
    print("-" * 70)

    total_labeled_all = 0
    total_rule_correct_all = 0
    total_llm_correct_all = 0

    for field in fields:
        labeled = stats["labeled"][field]
        if labeled == 0:
            continue

        rule_correct = stats["rule_correct"][field]
        llm_correct = stats["llm_correct"][field]

        total_labeled_all += labeled
        total_rule_correct_all += rule_correct
        total_llm_correct_all += llm_correct

        rule_acc = 100.0 * rule_correct / labeled
        llm_acc = 100.0 * llm_correct / labeled

        pretty_name = field.replace("_", " ").title()
        print(
            f"{pretty_name:<18} "
            f"Rule: {rule_acc:6.1f}%  "
            f"LLM: {llm_acc:6.1f}%  "
            f"(n={labeled})"
        )

    # Micro-average across all labeled fields
    if total_labeled_all > 0:
        overall_rule = 100.0 * total_rule_correct_all / total_labeled_all
        overall_llm = 100.0 * total_llm_correct_all / total_labeled_all

        print("\n" + "-" * 70)
        print(
            f"Overall (micro-avg)  "
            f"Rule: {overall_rule:6.1f}%  "
            f"LLM: {overall_llm:6.1f}%  "
            f"(total labeled fields = {total_labeled_all})"
        )
    else:
        print("No labeled fields found in ground truth.")

def main():
    """Run detailed comparison."""
    print("=" * 100)
    print("DETAILED QUERY-BY-QUERY COMPARISON")
    print("=" * 100)
    print()

    # Initialize parsers
    rule_parser = PropertySearchParser()
    llm_parser = LLMPropertyParser(model="gpt-3.5-turbo")

    # Load test cases from JSON
    test_queries, ground_truths = load_test_cases()

    print(f"Testing {len(test_queries)} queries...\n")
    print("⚠️  This will cost approximately ${:.4f}\n".format(len(test_queries) * 0.0002))

    input("Press Enter to continue...")
    print()

    results = []  # list of (query, truth, rule_res, llm_res)

    for i, (query, truth) in enumerate(zip(test_queries, ground_truths), 1):
        print(f"\n[{i}/{len(test_queries)}]")
        rule_res, llm_res = compare_single_query(query, truth, rule_parser, llm_parser)
        results.append((query, truth, rule_res, llm_res))

        if i < len(test_queries):
            input("Press Enter for next query...\n")

    # Save detailed log & analyze (you can adapt your existing functions)
    save_detailed_log(test_queries, ground_truths, results)
    analyze_differences(results)

    # Summary stats as you already had them
    total_cost = sum(llm_res.cost_usd for _, _, _, llm_res in results)
    total_time = sum(llm_res.processing_time_ms for _, _, _, llm_res in results)
    total_tokens = sum(llm_res.tokens_used for _, _, _, llm_res in results)

    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"\nTotal Queries: {len(test_queries)}")
    print(f"Total LLM Cost: ${total_cost:.6f}")
    print(f"Total LLM Time: {total_time:.1f}ms")
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Cost/Query: ${total_cost/len(test_queries):.6f}")
    print(f"Average Time/Query: {total_time/len(test_queries):.1f}ms")
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()
