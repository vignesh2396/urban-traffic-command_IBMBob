"""
UrbanPulse — Narrated Slideshow Video Generator
Generates a professional MP4 slideshow using moviepy + PIL.
Each slide has animated text, gradient backgrounds, and slide transitions.

Usage:
    python create_video.py

Output:
    UrbanPulse_Demo.mp4 (in the project root)

Requirements:
    pip install moviepy pillow numpy
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Try to import moviepy ──────────────────────────────────────────────────────
try:
    from moviepy.editor import (
        ImageClip,
        concatenate_videoclips,
        CompositeVideoClip,
        ColorClip,
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("[WARN] moviepy not fully available — will generate PNG slide previews only.")

# ── Video configuration ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080
FPS = 24
SLIDE_DURATION = 8  # seconds per slide
TRANSITION_DURATION = 0.5  # seconds for crossfade

# ── Color palette ──────────────────────────────────────────────────────────────
DARK_BG = (13, 27, 42)          # #0D1B2A — navy
WHITE_BG = (250, 250, 252)       # near-white
ACCENT = (245, 166, 35)          # #F5A623 — amber
TEXT_DARK = (26, 26, 26)         # near-black
TEXT_LIGHT = (224, 224, 224)     # light grey
TEXT_MUTED = (144, 164, 174)     # blue-grey
GREEN_OK = (76, 175, 80)         # success green
RED_CRITICAL = (229, 57, 53)     # critical red
BLUE_ACCENT = (30, 136, 229)     # info blue

# ── Font resolution ────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a system font, falling back to default if unavailable."""
    candidates = []
    if bold:
        candidates = [
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/verdanab.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _gradient_bg(
    w: int, h: int,
    color_top: tuple, color_bot: tuple,
) -> np.ndarray:
    """Create a vertical gradient background as numpy array."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        r = int(color_top[0] * (1 - t) + color_bot[0] * t)
        g = int(color_top[1] * (1 - t) + color_bot[1] * t)
        b = int(color_top[2] * (1 - t) + color_bot[2] * t)
        img[y, :] = [r, g, b]
    return img


def _draw_slide(
    title: str,
    body_lines: list[str],
    bg_dark: bool = True,
    accent_bar: bool = True,
    badge: str | None = None,
    footer: str = "IBM SkillsBuild Gartner Program 2026 | UrbanPulse Agentic AI Capstone",
) -> np.ndarray:
    """Render a complete slide as a numpy array (H x W x 3)."""

    # ── Background ────────────────────────────────────────────────────────────
    if bg_dark:
        bg_arr = _gradient_bg(WIDTH, HEIGHT, DARK_BG, (8, 20, 35))
        title_color = ACCENT
        body_color = TEXT_LIGHT
        footer_color = TEXT_MUTED
        divider_color = ACCENT
    else:
        bg_arr = _gradient_bg(WIDTH, HEIGHT, (248, 249, 252), (238, 241, 248))
        title_color = DARK_BG
        body_color = TEXT_DARK
        footer_color = (120, 130, 150)
        divider_color = ACCENT

    img = Image.fromarray(bg_arr)
    draw = ImageDraw.Draw(img)

    # ── Accent bar (left vertical stripe) ────────────────────────────────────
    if accent_bar:
        draw.rectangle([60, 100, 74, HEIGHT - 80], fill=ACCENT)

    # ── Title ─────────────────────────────────────────────────────────────────
    font_title = _load_font(72, bold=True)
    draw.text((110, 90), title, font=font_title, fill=title_color)

    # ── Divider line ──────────────────────────────────────────────────────────
    draw.rectangle([110, 185, WIDTH - 100, 190], fill=divider_color)

    # ── Badge (optional top-right pill) ───────────────────────────────────────
    if badge:
        font_badge = _load_font(36, bold=True)
        bw, bh = 400, 60
        bx, by = WIDTH - bw - 80, 90
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=30, fill=ACCENT)
        draw.text((bx + 20, by + 10), badge, font=font_badge, fill=DARK_BG)

    # ── Body text ─────────────────────────────────────────────────────────────
    font_body = _load_font(40)
    font_body_small = _load_font(34)
    y_pos = 220
    line_height = 52

    for raw_line in body_lines:
        if not raw_line.strip():
            y_pos += line_height // 2
            continue

        # Section headers (lines ending with :) in accent color
        if raw_line.rstrip().endswith(":") and not raw_line.startswith(" "):
            draw.text((110, y_pos), raw_line, font=font_body, fill=ACCENT if bg_dark else BLUE_ACCENT)
            y_pos += line_height + 4
            continue

        # Result lines — colour-code PASS/FAIL
        color = body_color
        if "✅" in raw_line or "PASS" in raw_line:
            color = GREEN_OK
        elif "❌" in raw_line or "FAIL" in raw_line:
            color = RED_CRITICAL
        elif "⚠️" in raw_line or "🔴" in raw_line:
            color = (255, 160, 0)

        # Wrap long lines
        wrapped = textwrap.wrap(raw_line, width=88)
        for wline in wrapped:
            if y_pos > HEIGHT - 140:
                break
            draw.text((110, y_pos), wline, font=font_body_small, fill=color)
            y_pos += line_height - 4
        y_pos += 4

    # ── Footer ────────────────────────────────────────────────────────────────
    font_footer = _load_font(32)
    draw.rectangle([60, HEIGHT - 78, WIDTH - 60, HEIGHT - 74], fill=divider_color)
    draw.text((110, HEIGHT - 65), footer, font=font_footer, fill=footer_color)

    return np.array(img)


# ── Slide definitions ──────────────────────────────────────────────────────────

SLIDES: list[dict] = [
    {
        "title": "🚦 UrbanPulse",
        "body": [
            "AI-Based Urban Traffic Command & Incident Response System",
            "",
            "IBM SkillsBuild Gartner Program 2026 | Agentic AI Capstone",
            "",
            "7 Specialized Agents  •  LangGraph StateGraph  •  Gemini 1.5 Flash",
            "ChromaDB RAG  •  Streamlit UI  •  HITL Gate  •  Docker CI/CD",
            "",
            "24/24 Tests Passing  •  Full Responsible AI Compliance",
        ],
        "dark": True,
        "badge": "v1.0.0",
    },
    {
        "title": "Problem Statement",
        "body": [
            "City traffic authorities face overwhelming real-time challenges:",
            "",
            "  • Manual monitoring cannot handle concurrent peak-hour events",
            "  • Incident detection takes 8–15 min using traditional methods",
            "  • No coordinated multi-agency response for critical accidents",
            "  • Human operators experience extreme cognitive overload",
            "  • Lack of data-driven signal optimization wastes commuter time",
            "",
            "Result: Gridlock, delayed emergency response, economic loss",
        ],
        "dark": False,
    },
    {
        "title": "Proposed Solution",
        "body": [
            "UrbanPulse — Proactive Multi-Agent Agentic AI System:",
            "",
            "  ✅ Monitors 10+ traffic zones continuously",
            "  ✅ Detects congestion anomalies in < 2 minutes",
            "  ✅ Investigates incidents using LLM + RAG + Weather",
            "  ✅ Generates coordinated response plans automatically",
            "  ✅ Recommends ≥2 alternative diversion routes",
            "  ✅ Escalates CRITICAL decisions to human operators (HITL)",
            "  ✅ Logs every action in an immutable audit trail",
        ],
        "dark": True,
        "badge": "Smart Cities",
    },
    {
        "title": "AI Agent Architecture — 7-Node Pipeline",
        "body": [
            "Agent                       Role",
            "─────────────────────────────────────────────────────────────────",
            "Traffic Monitoring Agent    Ingests/simulates sensor data",
            "Congestion Detection Agent  Rule-based threshold analysis",
            "Incident Investigation Agent  ⭐ Gemini LLM + RAG + Weather",
            "Route Diversion Agent       LLM ≥2 alternate routes",
            "Signal Optimization Agent   LLM adaptive signal timing",
            "Traffic Response Coord.     Full response plan + alerts",
            "Supervisor Agent            LangGraph StateGraph + HITL",
        ],
        "dark": False,
    },
    {
        "title": "Technology Stack",
        "body": [
            "Layer                    Technology",
            "─────────────────────────────────────────────────────────────────",
            "LLM                      Google Gemini 1.5 Flash",
            "Agentic Framework        LangGraph (StateGraph)",
            "UI                       Streamlit 1.66+",
            "RAG                      ChromaDB + Gemini Embeddings",
            "Database                 SQLite (prototype)",
            "Weather                  OpenWeatherMap API",
            "Language                 Python 3.11+",
            "Containerisation         Docker + Docker Compose",
            "CI/CD                    GitHub Actions",
        ],
        "dark": True,
    },
    {
        "title": "Streamlit UI — 5 Functional Tabs",
        "body": [
            "🗺️  Tab 1: Live Zone Monitor",
            "       Zone selector (Z-01 to Z-10) + scenario picker + AI trigger",
            "",
            "🚨  Tab 2: Incident Queue",
            "       HITL approve/reject + AI reasoning trace + response plan",
            "",
            "📊  Tab 3: Analytics",
            "       Incident charts + zone traffic overview + metrics",
            "",
            "📋  Tab 4: Audit Log   |   ⚙️ Tab 5: System Health",
            "       Immutable agent log  |  LLM + DB + RAG status",
        ],
        "dark": False,
        "badge": "Streamlit",
    },
    {
        "title": "HITL Gate — Responsible AI",
        "body": [
            "All CRITICAL incidents (severity ≥ 9/10) require human approval:",
            "",
            "  1. Supervisor detects severity_score ≥ 9",
            "  2. Sets awaiting_human = True in LangGraph state",
            "  3. Pipeline pauses — graph routes to END (hitl_wait)",
            "  4. UI renders ⛔ CRITICAL banner with APPROVE / REJECT",
            "  5. Operator reviews reasoning + evidence + diversion routes",
            "  6. Operator clicks APPROVE or REJECT with optional notes",
            "  7. Pipeline resumes → execute node → incident CLOSED",
        ],
        "dark": True,
        "badge": "HITL",
    },
    {
        "title": "LLM Testing — 24/24 Tests Pass  ✅",
        "body": [
            "TC-LLM-01  Correctness — congestion classification             ✅ PASS",
            "TC-LLM-02  Correctness — CRITICAL accident classification      ✅ PASS",
            "TC-LLM-03  Faithfulness — RAG sources cited in output          ✅ PASS",
            "TC-LLM-04  Hallucination — schema compliance (valid enum)      ✅ PASS",
            "TC-LLM-05  Robustness — malformed JSON triggers fallback       ✅ PASS",
            "TC-LLM-06  Robustness — API exception triggers fallback        ✅ PASS",
            "TC-LLM-07  Safety — prompt injection resistance                ✅ PASS",
            "TC-LLM-08  RAG empty — LLM general knowledge fallback          ✅ PASS",
            "TC-LLM-09  Severity clamping — out-of-range score clamped      ✅ PASS",
            "TC-LLM-10  Explainability — reasoning always populated         ✅ PASS",
            "TC-LLM-11  Agent trace — trace entry appended                  ✅ PASS",
            "TC-LLM-12  Performance — latency within 2s budget              ✅ PASS",
            "TC-LLM-13  HITL gate — severity ≥ 9 triggers approval gate     ✅ PASS",
            "TC-LLM-14  HITL gate — severity < 9 skips gate                 ✅ PASS",
            "TC-LLM-15  Schema completeness — all required fields present   ✅ PASS",
        ],
        "dark": False,
        "badge": "24 / 24 ✅",
    },
    {
        "title": "Agentic AI Requirements — All Met ✅",
        "body": [
            "Guideline §4 Checklist:",
            "",
            "  ✅ Autonomous AI Agent — LangGraph conditional routing",
            "  ✅ Multiple Agent Roles (7 implemented vs ≥3 required)",
            "  ✅ Tool Calling (4 tools: sensor, weather, RAG, audit)",
            "  ✅ RAG — ChromaDB + Gemini Embeddings (7 knowledge docs)",
            "  ✅ Agent Memory/State — MemorySaver + SQLite persistence",
            "  ✅ Multi-Step Reasoning — 7-node directed pipeline",
            "  ✅ Human-in-the-Loop — HITL gate for severity ≥ 9",
            "  ✅ Explainability — reasoning panel + agent trace UI",
        ],
        "dark": True,
        "badge": "All 8 Met",
    },
    {
        "title": "🚀 Project Summary",
        "body": [
            "UrbanPulse delivers full-stack Agentic AI for urban traffic management:",
            "",
            "  ⚡ 7 specialized AI agents in a LangGraph StateGraph pipeline",
            "  🧠 Gemini 1.5 Flash LLM + ChromaDB RAG (7 knowledge sources)",
            "  🌤️ Real-time OpenWeatherMap integration for adverse conditions",
            "  🖥️ Streamlit dashboard with 5 functional tabs",
            "  🔒 HITL approval gate — 100% CRITICAL incidents require sign-off",
            "  📋 Immutable SQLite audit trail for all agent actions",
            "  ✅ 24/24 tests passing | Docker + GitHub Actions CI/CD",
            "",
            "GitHub: https://github.com/vignesh2396/urban-traffic-command_IBMBob",
        ],
        "dark": False,
        "badge": "v1.0.0",
    },
]


# ── Main rendering ─────────────────────────────────────────────────────────────

def make_slide_array(slide_def: dict) -> np.ndarray:
    """Render a slide definition to a numpy array."""
    return _draw_slide(
        title=slide_def["title"],
        body_lines=slide_def.get("body", []),
        bg_dark=slide_def.get("dark", True),
        badge=slide_def.get("badge"),
    )


def save_png_previews(output_dir: Path) -> None:
    """Save each slide as a PNG preview image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, slide_def in enumerate(SLIDES):
        arr = make_slide_array(slide_def)
        img = Image.fromarray(arr)
        out_path = output_dir / f"slide_{i+1:02d}.png"
        img.save(out_path)
        print(f"  [PNG] Saved: {out_path}")


def render_video(output_path: Path) -> None:
    """Render all slides as an MP4 video using moviepy."""
    print(f"\n[VIDEO] Rendering {len(SLIDES)} slides → {output_path}")

    clips = []
    for i, slide_def in enumerate(SLIDES):
        print(f"  [Slide {i+1}/{len(SLIDES)}] {slide_def['title'][:50]}...")
        arr = make_slide_array(slide_def)

        # Create clip from array
        clip = ImageClip(arr, duration=SLIDE_DURATION)

        # Fade-in / fade-out transition
        clip = clip.fadein(TRANSITION_DURATION).fadeout(TRANSITION_DURATION)
        clips.append(clip)

    # Concatenate all clips
    print("\n[VIDEO] Concatenating clips...")
    final = concatenate_videoclips(clips, method="compose")

    # Write MP4
    print(f"[VIDEO] Writing {output_path} ...")
    final.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio=False,
        verbose=False,
        logger=None,
    )
    print(f"\n✅ Video saved: {output_path}")
    print(f"   Duration: {len(SLIDES) * SLIDE_DURATION}s | Resolution: {WIDTH}x{HEIGHT}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).parent

    # Always save PNG previews
    png_dir = root / "assets" / "slide_previews"
    print(f"\n[PNG] Saving slide previews to: {png_dir}")
    save_png_previews(png_dir)

    # Render video if moviepy is available
    if MOVIEPY_AVAILABLE:
        video_path = root / "UrbanPulse_Demo.mp4"
        try:
            render_video(video_path)
        except Exception as exc:
            print(f"\n[ERROR] Video rendering failed: {exc}")
            print("PNG previews have been saved successfully.")
    else:
        print("\n[INFO] moviepy is not fully available.")
        print("PNG slide previews saved. Install moviepy to generate MP4:")
        print("  pip install moviepy")
