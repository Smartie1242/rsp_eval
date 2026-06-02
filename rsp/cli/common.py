"""Small CLI formatting helpers."""


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def print_metrics_table(metrics, labels) -> None:
    """Pretty-print a metrics table."""
    print(f"\n{'Language':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 61)
    for label in labels:
        if label in metrics:
            metric = metrics[label]
            print(
                f"{label:<15} "
                f"{metric['precision']:<12.3f} "
                f"{metric['recall']:<12.3f} "
                f"{metric['f1']:<12.3f} "
                f"{metric.get('support', 0):<10.0f}"
            )

