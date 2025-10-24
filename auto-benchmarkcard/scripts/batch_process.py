#!/usr/bin/env python3
"""
Batch job script to run agents.py on all unitxt cards in the catalog.
Provides statistics and saves failed cards to a file.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from unitxt.catalog import get_from_catalog
from unitxt.ui.load_catalog_data import get_catalog_items

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BatchJobRunner:
    """Runs agents.py on all unitxt cards and tracks statistics."""

    def __init__(self, output_dir: str = None, limit: int = None, skip_existing: bool = True):
        self.output_dir = Path(output_dir) if output_dir else Path("batch_output")
        self.limit = limit
        self.skip_existing = skip_existing
        self.stats = {
            "total_cards": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None,
        }
        self.failed_cards = []

    def get_all_cards(self) -> List[str]:
        """Get all unitxt cards from catalog."""
        logger.info("Loading all unitxt cards from catalog...")

        try:
            # Get cards using get_catalog_items - returns [cards_list, cards_dict]
            cards_data = get_catalog_items("cards")
            cards_list = cards_data[0]  # First element is the list of card names

            # Remove 'cards.' prefix from each card name
            clean_cards = []
            for card in cards_list:
                if card.startswith("cards."):
                    clean_name = card.replace("cards.", "", 1)
                    clean_cards.append(clean_name)
                else:
                    clean_cards.append(card)

            logger.info(f"Found {len(clean_cards)} cards in catalog")
            return clean_cards

        except Exception as e:
            logger.error(f"Failed to load cards from catalog: {e}")
            sys.exit(1)

    def card_already_processed(self, card_name: str) -> bool:
        """Check if a card has already been processed."""
        if not self.skip_existing:
            return False

        # Check for existing output directory for this card
        # Look for pattern: output/{card_name}_{timestamp}/
        output_base = Path("output")
        if not output_base.exists():
            return False

        for dir_path in output_base.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith(f"{card_name}_"):
                # Check if it has both tool_output and benchmarkcard directories
                tool_output = dir_path / "tool_output"
                benchmark_card = dir_path / "benchmarkcard"
                if tool_output.exists() and benchmark_card.exists():
                    logger.debug(f"Card {card_name} already processed in {dir_path}")
                    return True

        return False

    def run_agents_for_card(self, card_name: str) -> Tuple[bool, str]:
        """Run agents.py for a single card."""
        try:
            logger.info(f"Processing card: {card_name}")

            # Run agents.py with the card name
            cmd = [sys.executable, "agents.py", card_name]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✓ Successfully processed: {card_name}")
                return True, "Success"
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.warning(f"✗ Failed to process {card_name}: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"✗ Exception processing {card_name}: {error_msg}")
            return False, error_msg

    def save_failed_cards(self, failed_cards: List[Dict]):
        """Save failed cards to a JSON file."""
        if not failed_cards:
            logger.info("No failed cards to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_file = self.output_dir / f"failed_cards_{timestamp}.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(failed_file, "w") as f:
            json.dump(failed_cards, f, indent=2)

        logger.info(f"Failed cards saved to: {failed_file}")

    def save_summary_stats(self):
        """Save summary statistics to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = self.output_dir / f"batch_job_stats_{timestamp}.json"

        # Calculate additional stats
        if self.stats["total_cards"] > 0:
            success_rate = (
                (self.stats["successful"] / (self.stats["successful"] + self.stats["failed"])) * 100
                if (self.stats["successful"] + self.stats["failed"]) > 0
                else 0
            )
            self.stats["success_rate_percent"] = round(success_rate, 1)

        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]
            self.stats["duration_seconds"] = duration.total_seconds()
            self.stats["duration_human"] = str(duration).split(".")[0]  # Remove microseconds

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)

        logger.info(f"Summary statistics saved to: {stats_file}")

    def print_summary(self):
        """Print summary statistics."""
        total_attempted = self.stats["successful"] + self.stats["failed"]
        success_rate = (
            (self.stats["successful"] / total_attempted * 100) if total_attempted > 0 else 0
        )

        print("\n" + "=" * 60)
        print("BATCH JOB SUMMARY")
        print("=" * 60)
        print(f"Total cards in catalog: {self.stats['total_cards']}")
        print(f"Cards attempted: {total_attempted}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped (already processed): {self.stats['skipped']}")
        print(f"Success rate: {success_rate:.1f}%")

        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]
            print(f"Total runtime: {str(duration).split('.')[0]}")

        print("=" * 60)

        if self.failed_cards:
            print(f"\nFailed cards ({len(self.failed_cards)}):")
            for failed in self.failed_cards[:10]:  # Show first 10
                print(f"  - {failed['card']}: {failed['error'][:50]}...")
            if len(self.failed_cards) > 10:
                print(f"  ... and {len(self.failed_cards) - 10} more")

    def run_batch_job(self):
        """Run the complete batch job."""
        logger.info("Starting batch job for all unitxt cards")

        # Get all cards
        all_cards = self.get_all_cards()
        self.stats["total_cards"] = len(all_cards)

        # Apply limit if specified
        if self.limit:
            all_cards = all_cards[: self.limit]
            logger.info(f"Limiting to first {self.limit} cards")

        self.stats["start_time"] = datetime.now()

        # Process each card
        for i, card in enumerate(all_cards, 1):
            logger.info(f"\n[{i}/{len(all_cards)}] Processing: {card}")

            # Check if already processed
            if self.card_already_processed(card):
                logger.info(f"Skipping {card} (already processed)")
                self.stats["skipped"] += 1
                continue

            # Run agents.py for this card
            success, error_msg = self.run_agents_for_card(card)

            if success:
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1
                self.failed_cards.append(
                    {
                        "card": card,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Progress update
            if i % 10 == 0:
                attempted = self.stats["successful"] + self.stats["failed"]
                success_rate = (self.stats["successful"] / attempted * 100) if attempted > 0 else 0
                logger.info(
                    f"Progress: {i}/{len(all_cards)} cards, {success_rate:.1f}% success rate"
                )

        self.stats["end_time"] = datetime.now()

        # Save results
        self.save_failed_cards(self.failed_cards)
        self.save_summary_stats()
        self.print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run agents.py on all unitxt cards")
    parser.add_argument("--limit", type=int, help="Limit number of cards to process (for testing)")
    parser.add_argument("--output-dir", default="batch_output", help="Output directory for results")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip already processed cards")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run batch job
    runner = BatchJobRunner(
        output_dir=args.output_dir, limit=args.limit, skip_existing=not args.no_skip
    )

    try:
        runner.run_batch_job()
    except KeyboardInterrupt:
        logger.info("Batch job interrupted by user")
        runner.stats["end_time"] = datetime.now()
        runner.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch job failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
