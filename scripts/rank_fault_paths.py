"""Build a four-layer fault graph and Balanced Risk ranking without plotting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from engine_fault_nlp.io_utils import read_json, read_jsonl
from engine_fault_nlp.paper_ranking import build_ranking, document_id, export_ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Model-output JSONL, not reference labels")
    parser.add_argument("--work-orders", default="data/work_orders.jsonl")
    parser.add_argument("--cohort", help="Optional JSONL whose IDs select a cohort; labels are not used")
    parser.add_argument("--knowledge-base", default="data/knowledge_base.json")
    parser.add_argument("--output-dir", default="outputs/path_ranking")
    parser.add_argument("--root-name", default="Equipment")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--shrinkage", type=float, default=5.0)
    parser.add_argument("--canonicalize", action="store_true", help="Apply historical graph component aliases, after prediction")
    parser.add_argument("--screen-materials", action="store_true", help="Remove historical non-fault material chains and material-only orders")
    parser.add_argument("--fault-only", action="store_true", help="Require an explicit FAULT record_type in the supplied prediction")
    args = parser.parse_args()
    cohort = [document_id(row) for row in read_jsonl(args.cohort)] if args.cohort else None
    result = build_ranking(read_jsonl(args.work_orders), read_jsonl(args.predictions), cohort_ids=cohort,
                           knowledge_base=read_json(args.knowledge_base), root_name=args.root_name,
                           threshold=args.threshold, shrinkage=args.shrinkage, canonicalize=args.canonicalize,
                           screen_materials=args.screen_materials, fault_only=args.fault_only)
    export_ranking(result, args.output_dir)
    print(f"Exported {len(result.ranking)} ranked paths to {args.output_dir}")


if __name__ == "__main__":
    main()
