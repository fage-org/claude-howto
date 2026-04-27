#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["markdown", "beautifulsoup4"]
# ///
"""
Build a PDF from the Claude How-To markdown files using pandoc.

Usage:
    Run from the repository root directory:
        ./scripts/build_pdf.py

    Or run directly with Python/uv:
        uv run scripts/build_pdf.py
        python scripts/build_pdf.py

    Command-line options:
        --root, -r          Root directory containing markdown files (default: repo root)
        --output, -o        Output PDF file path (default: <repo_root>/claude-howto-guide.pdf)
        --verbose, -v       Enable verbose logging
        --lang              Language: en | vi | zh (default: en)
        --mmdc-path         Path to mmdc binary (default: mmdc from PATH)
        --puppeteer-config  Path to Puppeteer config JSON for mmdc in CI
        --pandoc-path       Path to pandoc binary (default: pandoc from PATH)

Requirements:
    - pandoc >= 2.11 installed (https://pandoc.org/installing.html)
    - For PDF: xelatex (TeX Live / MiKTeX) OR weasyprint (pip install weasyprint)
      xelatex is used by default when available; weasyprint is the fallback.
    - For CJK: fonts-noto-cjk (Linux) / Source Han Serif (macOS) installed
    - @mermaid-js/mermaid-cli for Mermaid diagrams:
        npm install -g @mermaid-js/mermaid-cli
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-use mermaid rendering + chapter-collection from build_epub
# ---------------------------------------------------------------------------
# We import selectively to avoid pulling in the heavy ebooklib/pillow deps
# for PDF builds.  The shared helpers live in build_epub.py in the same dir.
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from build_epub import (  # noqa: E402
    BuildState,
    EPUBConfig,
    MermaidRenderer,
    ChapterCollector,
    extract_all_mermaid_blocks,
    get_chapter_order,
    sanitize_mermaid,
    setup_logging,
)


# =============================================================================
# Exceptions
# =============================================================================


class PDFBuildError(Exception):
    """Base exception for PDF build errors."""


class PandocNotFoundError(PDFBuildError):
    """pandoc binary not found."""


class PDFEngineNotFoundError(PDFBuildError):
    """No suitable PDF engine found."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PDFConfig:
    """Configuration for PDF generation."""

    root_path: Path
    output_path: Path
    language: str = "en"
    title: str = "Claude Code How-To Guide"
    author: str = "Claude Code Community"

    mmdc_path: str = "mmdc"
    puppeteer_config: str | None = None
    pandoc_path: str = "pandoc"

    # Language-specific titles (mirrors EPUBConfig for consistency)
    en_title: str = "Claude Code How-To Guide"
    vi_title: str = "Hướng Dẫn Claude Code"
    zh_title: str = "Claude Code 使用指南"

    # CJK font names used with xelatex
    cjk_main_font: str = "Noto Serif CJK SC"
    cjk_sans_font: str = "Noto Sans CJK SC"
    cjk_mono_font: str = "Noto Sans Mono CJK SC"

    # Latin font names used with xelatex
    main_font: str = "DejaVu Serif"
    sans_font: str = "DejaVu Sans"
    mono_font: str = "DejaVu Sans Mono"


# =============================================================================
# Engine detection
# =============================================================================


def _which(binary: str) -> str | None:
    return shutil.which(binary)


def detect_pdf_engine(logger: logging.Logger) -> str:
    """Return the best available PDF engine name for pandoc."""
    for engine in ("xelatex", "lualatex", "pdflatex", "weasyprint", "wkhtmltopdf"):
        if _which(engine):
            logger.info(f"PDF engine: {engine}")
            return engine
    raise PDFEngineNotFoundError(
        "No PDF engine found. Install one of: xelatex (texlive-xetex), "
        "weasyprint (pip install weasyprint), or wkhtmltopdf."
    )


# =============================================================================
# Mermaid pre-processing (render diagrams → PNG, rewrite md blocks)
# =============================================================================


def preprocess_markdown(
    md_content: str,
    mermaid_state: BuildState,
    image_dir: Path,
    logger: logging.Logger,
) -> str:
    """Replace ```mermaid blocks with ![Diagram](path/to/png) references."""
    pattern = r"```mermaid\n(.*?)```"

    def replace(match: re.Match[str]) -> str:
        raw_code = match.group(1)
        sanitized = sanitize_mermaid(raw_code)
        cache_key = sanitized.strip()

        if cache_key not in mermaid_state.mermaid_cache:
            logger.warning("Mermaid diagram missing from pre-rendered cache — skipping")
            return ""

        img_data, img_name = mermaid_state.mermaid_cache[cache_key]
        img_path = image_dir / img_name
        if not img_path.exists():
            img_path.write_bytes(img_data)

        return f"\n![Diagram]({img_path})\n"

    return re.sub(pattern, replace, md_content, flags=re.DOTALL)


# =============================================================================
# Markdown collection helpers
# =============================================================================


def collect_all_md_files(
    root_path: Path, language: str
) -> list[tuple[Path, str]]:
    """Return ordered (file_path, display_name) pairs for the given language root."""
    # Reuse the same EPUB chapter order/collection logic
    dummy_state = BuildState()
    collector = ChapterCollector(root_path, dummy_state)
    chapter_infos = collector.collect_all_chapters(get_chapter_order())
    return [(ch.file_path, ch.file_title) for ch in chapter_infos]


# =============================================================================
# pandoc helpers
# =============================================================================


_LATEX_CJK_HEADER = r"""
\usepackage{xeCJK}
\xeCJKsetup{AutoFakeBold=true, AutoFakeSlant=true}
"""


def _build_pandoc_cmd(
    input_files: list[str],
    output_path: Path,
    config: PDFConfig,
    engine: str,
    logger: logging.Logger,
) -> list[str]:
    """Construct the pandoc command list."""
    pandoc = _which(config.pandoc_path) or config.pandoc_path
    if not _which(pandoc):
        raise PandocNotFoundError(
            f"pandoc not found at '{config.pandoc_path}'. "
            "Install it from https://pandoc.org/installing.html"
        )

    is_cjk = config.language in ("zh", "ja", "ko")

    cmd: list[str] = [
        pandoc,
        "--pdf-engine", engine,
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "--file-scope",
        "-V", f"lang={config.language}",
        "-V", "papersize=a4",
        "-V", "geometry:top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm",
        "-V", "colorlinks=true",
        "-V", "linkcolor=NavyBlue",
        "-V", "urlcolor=NavyBlue",
        # Title block variables
        "-M", f"title={config.title}",
        "-M", f"author={config.author}",
        # Standalone document
        "--standalone",
        "-o", str(output_path),
    ]

    if engine in ("xelatex", "lualatex"):
        if is_cjk:
            cmd += [
                "-V", f"CJKmainfont={config.cjk_main_font}",
                "-V", f"CJKsansfont={config.cjk_sans_font}",
                "-V", f"CJKmonofont={config.cjk_mono_font}",
            ]
        else:
            cmd += [
                "-V", f"mainfont={config.main_font}",
                "-V", f"sansfont={config.sans_font}",
                "-V", f"monofont={config.mono_font}",
            ]

    cmd += input_files
    return cmd


# =============================================================================
# Main build function
# =============================================================================


def build_pdf(config: PDFConfig, logger: logging.Logger) -> Path:
    """Build a PDF from the markdown files in config.root_path."""
    # Validate inputs
    if not config.root_path.exists():
        raise PDFBuildError(f"Root path does not exist: {config.root_path}")

    engine = detect_pdf_engine(logger)

    # Collect markdown files
    logger.info("Collecting chapters...")
    md_files = collect_all_md_files(config.root_path, config.language)
    if not md_files:
        raise PDFBuildError(f"No markdown files found in {config.root_path}")
    logger.info(f"Found {len(md_files)} chapters")

    # Build a minimal EPUBConfig just for the MermaidRenderer (it only needs
    # mmdc_path and puppeteer_config from the config object).
    epub_config = EPUBConfig(
        root_path=config.root_path,
        output_path=config.output_path.with_suffix(".epub"),  # dummy, not written
        language=config.language,
        mmdc_path=config.mmdc_path,
        puppeteer_config=config.puppeteer_config,
    )
    state = BuildState()

    # Extract and render all Mermaid diagrams
    logger.info("Extracting Mermaid diagrams...")
    all_diagrams = extract_all_mermaid_blocks(md_files, logger)
    if all_diagrams:
        renderer = MermaidRenderer(epub_config, state, logger)
        renderer.render_all(all_diagrams)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        image_dir = tmp / "images"
        image_dir.mkdir()
        processed_files: list[str] = []

        for i, (file_path, _title) in enumerate(md_files):
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                logger.warning(f"Skipping unreadable file {file_path}: {exc}")
                continue

            # Replace Mermaid code-blocks with rendered image references
            content = preprocess_markdown(content, state, image_dir, logger)

            out_md = tmp / f"chapter_{i:04d}.md"
            out_md.write_text(content, encoding="utf-8")
            processed_files.append(str(out_md))

        if not processed_files:
            raise PDFBuildError("No processable markdown files found")

        # Run pandoc
        cmd = _build_pandoc_cmd(processed_files, config.output_path, config, engine, logger)
        logger.info(f"Running pandoc ({engine}) → {config.output_path}")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"pandoc stderr:\n{result.stderr}")
            raise PDFBuildError(f"pandoc failed (exit {result.returncode}):\n{result.stderr.strip()}")

        if result.stderr:
            logger.debug(f"pandoc stderr:\n{result.stderr}")

    if not config.output_path.exists():
        raise PDFBuildError("pandoc exited successfully but no PDF was produced")

    logger.info(f"PDF created successfully: {config.output_path}")
    return config.output_path


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build a PDF from Claude How-To markdown files."
    )
    parser.add_argument("--root", "-r", type=Path, default=None,
                        help="Root directory of markdown files (default: repo root)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output PDF path (default: <repo_root>/claude-howto-guide[-lang].pdf)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "vi", "zh"],
                        help="Language: en | vi | zh (default: en)")
    parser.add_argument("--mmdc-path", type=str, default="mmdc",
                        help="Path to mmdc binary (default: mmdc)")
    parser.add_argument("--puppeteer-config", type=str, default=None,
                        help="Path to Puppeteer config JSON for mmdc (CI use)")
    parser.add_argument("--pandoc-path", type=str, default="pandoc",
                        help="Path to pandoc binary (default: pandoc)")

    args = parser.parse_args()

    repo_root = (args.root or Path(__file__).parent.parent).resolve()

    lang_map: dict[str, tuple[Path, str, str]] = {
        "en": (repo_root,          "claude-howto-guide.pdf",    "Claude Code How-To Guide"),
        "vi": (repo_root / "vi",   "claude-howto-guide-vi.pdf", "Hướng Dẫn Claude Code"),
        "zh": (repo_root / "zh",   "claude-howto-guide-zh.pdf", "Claude Code 使用指南"),
    }
    root, default_name, title = lang_map[args.lang]
    output = (args.output or (repo_root / default_name)).resolve()

    logger = setup_logging(args.verbose)
    config = PDFConfig(
        root_path=root.resolve(),
        output_path=output,
        language=args.lang,
        title=title,
        mmdc_path=args.mmdc_path,
        puppeteer_config=args.puppeteer_config,
        pandoc_path=args.pandoc_path,
    )

    try:
        result = build_pdf(config, logger)
        print(f"Successfully created: {result}")
        return 0
    except PDFBuildError as e:
        logger.error(f"Build failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
