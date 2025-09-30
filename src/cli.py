"""
Command-line interface (CLI) for Credit Card Fraud Detection project.

Usage:
    python -m src.cli train         # Train models
    python -m src.cli evaluate      # Evaluate saved models
    python -m src.cli app           # Launch Streamlit demo app
"""

import subprocess
import sys

import click

from src import __version__, logger


@click.group()
@click.version_option(__version__)
def cli():
    """Credit Card Fraud Detection — CLI entrypoint."""
    pass


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file.")
def train(config):
    """Train models (XGBoost + Neural Net)."""
    logger.info("Starting training...")
    subprocess.run(
        [sys.executable, "src/train_model.py", "--config", config], check=True
    )


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file.")
def evaluate(config):
    """Evaluate saved models on test set."""
    logger.info("Running evaluation...")
    subprocess.run([sys.executable, "src/evaluate.py", "--config", config], check=True)


@cli.command()
def app():
    """Run Streamlit app for inference demo."""
    logger.info("Launching Streamlit app...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"], check=True)


@cli.command()
def lint():
    """Run code format & lint checks."""
    logger.info("Running black + flake8...")
    subprocess.run([sys.executable, "-m", "black", "."], check=True)
    subprocess.run([sys.executable, "-m", "flake8", "src"], check=True)


if __name__ == "__main__":
    cli()
