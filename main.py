"""
AgentFGL Main Entry Point
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(__file__))

from config import config
from agentfgl_pipeline import AgentFGLPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="AgentFGL: Multi-Agent Semantic Disentanglement for Fair TAG Learning")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory path"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input CSV file name (default: person.csv)"
    )

    parser.add_argument(
        "--input_edges",
        type=str,
        default=None,
        help="Input edges file name (default: edges.txt)"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file name (default: person_agentfgl.csv)"
    )

    parser.add_argument(
        "--output_edges",
        type=str,
        default=None,
        help="Output edges file name (default: edges_agentfgl.txt)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max concurrent workers (default: 30)"
    )

    parser.add_argument(
        "--max_rounds",
        type=int,
        default=None,
        help="Max refinement rounds (default: 3)"
    )

    parser.add_argument(
        "--edit_budget_ratio",
        type=float,
        default=None,
        help="Graph editing budget ratio (default: 0.1)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="BERT encoding device (default: cuda)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 80)
    print("AgentFGL: Multi-Agent Semantic Disentanglement for Fair TAG Learning")
    print("=" * 80)

    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.input_csv:
        config.INPUT_CSV = args.input_csv
    if args.input_edges:
        config.INPUT_EDGES = args.input_edges
    if args.output_csv:
        config.OUTPUT_CSV = args.output_csv
    if args.output_edges:
        config.OUTPUT_EDGES = args.output_edges
    if args.model:
        config.MODEL_NAME = args.model
    if args.max_workers:
        config.MAX_WORKERS = args.max_workers
    if args.max_rounds:
        config.MAX_REFINEMENT_ROUNDS = args.max_rounds
    if args.edit_budget_ratio:
        config.EDIT_BUDGET_RATIO = args.edit_budget_ratio
    if args.device:
        config.DEVICE = args.device
    if args.verbose:
        config.VERBOSE = True
        config.LOG_LEVEL = "DEBUG"

    try:
        config.validate()
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

    print("\nConfiguration:")
    print(f"  Data directory: {config.DATA_DIR}")
    print(f"  Input CSV: {config.INPUT_CSV}")
    print(f"  Input edges: {config.INPUT_EDGES}")
    print(f"  Output CSV: {config.OUTPUT_CSV}")
    print(f"  Output edges: {config.OUTPUT_EDGES}")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Max workers: {config.MAX_WORKERS}")
    print(f"  Max refinement rounds: {config.MAX_REFINEMENT_ROUNDS}")
    print(f"  Edit budget ratio: {config.EDIT_BUDGET_RATIO}")
    print(f"  BERT device: {config.DEVICE}")
    print()

    try:
        pipeline = AgentFGLPipeline(config)
        pipeline.run()

        print("\n" + "=" * 80)
        print("✅ AgentFGL executed successfully!")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  - Refined node data: {config.get_output_csv_path()}")
        print(f"  - Refined edges: {config.get_output_edges_path()}")
        print(f"  - Complete log: {config.get_output_log_path()}")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
