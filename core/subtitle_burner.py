#!/usr/bin/env python3
"""
SubtitleBurner — burn subtitles into video clips, with optional translation.

Reads paired .mp4 + .srt files from a clips directory, optionally translates
the SRT to a target language via Qwen API, generates an ASS subtitle file,
then burns it into the output video with ffmpeg.

Usage (standalone):
    from core.subtitle_burner import SubtitleBurner

    # Burn original subtitles only (no API key needed):
    burner = SubtitleBurner()
    result = burner.burn_subtitles_for_clips("clips/", "clips_post_processed/")

    # Burn original + translated subtitles:
    burner = SubtitleBurner(api_key=os.environ["QWEN_API_KEY"])
    result = burner.burn_subtitles_for_clips(
        "clips/", "clips_post_processed/", subtitle_translation="Simplified Chinese"
    )
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from core.config import API_KEY_ENV_VARS

logger = logging.getLogger(__name__)

# Matches diarization speaker tags at the start of a subtitle line, e.g.:
#   "[Sam Altman] text"  →  "text"
#   "[SPEAKER_01] >> text"  →  "text"
_SPEAKER_TAG_RE = re.compile(r"^\[.*?\]\s*(?:>>\s*)?", re.UNICODE)


# ASS subtitle header — two styles:
#   Original:    bottom center (Alignment=2), 30px margin from bottom  — white
#   Translation: bottom center (Alignment=2), 80px margin from bottom  — yellow
ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Original,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1
Style: Translation,Arial,48,&H0000FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class SubtitleBurner:
    """Burn SRT subtitles into video clips, with optional LLM-powered translation."""

    def __init__(self, api_key: str = None, provider: str = "qwen", model: str = None):
        self.model = model  # None → each client uses its config default
        if api_key:
            provider = provider.lower()
            if provider == "openrouter":
                from core.llm.openrouter_api_client import OpenRouterAPIClient
                self.client = OpenRouterAPIClient(api_key=api_key)
            else:
                from core.llm.qwen_api_client import QwenAPIClient
                self.client = QwenAPIClient(api_key=api_key)
        else:
            self.client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def burn_subtitles_for_clips(
        self, clips_dir, output_dir, subtitle_translation: str = None
    ) -> dict:
        """
        Process all MP4+SRT pairs in clips_dir and write subtitle-burned
        versions to output_dir.

        Args:
            clips_dir: Directory containing .mp4 and .srt files.
            output_dir: Directory to write burned clips to.
            subtitle_translation: If set (e.g. "Simplified Chinese"), translate
                the SRT to that language and burn both tracks. Requires api_key.

        Returns a summary dict with success, total_clips, successful_clips,
        failed_clips, and output_dir.
        """
        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for mp4 in sorted(clips_dir.glob("*.mp4")):
            srt = mp4.with_suffix(".srt")
            if not srt.exists():
                logger.warning(f"No SRT for {mp4.name}, skipping subtitle burn")
                continue
            logger.info(f"  Burning subtitles: {mp4.name}")
            ok = self._process_clip(
                mp4, srt, output_dir / mp4.name, subtitle_translation
            )
            results.append(ok)

        successful = sum(results)
        logger.info(f"  Subtitle burning complete: {successful}/{len(results)} clips")
        return {
            "success": True,
            "output_dir": str(output_dir),
            "total_clips": len(results),
            "successful_clips": successful,
            "failed_clips": len(results) - successful,
        }

    def prepare_ass_for_clip(
        self, srt_path: Path, ass_path: Path, subtitle_translation: str = None
    ) -> bool:
        """
        Write an ASS subtitle file for a single clip.
        Used by TitleAdder for single-pass title+subtitle burns.

        Args:
            srt_path: Source SRT file.
            ass_path: Destination ASS file to write.
            subtitle_translation: If set, translate and include a second track.

        Returns True on success, False if the SRT is empty.
        """
        segments = self._parse_srt(srt_path)
        if not segments:
            return False
        translated = self._translate_srt(segments, subtitle_translation) if subtitle_translation and self.client else None
        ass_path.write_text(self._generate_ass(segments, translated), encoding="utf-8")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_clip(
        self, mp4: Path, srt: Path, output: Path, subtitle_translation: str = None
    ) -> bool:
        """Generate ASS + burn into output MP4."""
        ass_path = output.with_suffix(".ass")
        ok = self.prepare_ass_for_clip(srt, ass_path, subtitle_translation)
        if not ok:
            logger.warning(f"  Skipping {mp4.name}: no subtitle segments found")
            return False
        success = self._burn_ass(mp4, ass_path, output)
        ass_path.unlink(missing_ok=True)
        return success

    def _parse_srt(self, srt_path: Path) -> list:
        """Parse SRT file into list of {start, end, text} dicts."""
        return self._parse_srt_text(srt_path.read_text(encoding="utf-8"))

    def _parse_srt_text(self, text: str) -> list:
        """Parse SRT text into list of {start, end, text} dicts.

        Clips each segment's end to the next segment's start to prevent
        overlapping subtitle entries (common in Whisper-generated SRTs).
        """
        segments = []
        for block in re.split(r"\n\s*\n", text.strip()):
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            m = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                lines[1],
            )
            if not m:
                continue
            segments.append(
                {
                    "start": m.group(1),
                    "end": m.group(2),
                    "text": " ".join(lines[2:]),
                }
            )
        # Clip each segment's end to the start of the next to remove overlaps
        for i in range(len(segments) - 1):
            if segments[i]["end"] > segments[i + 1]["start"]:
                segments[i]["end"] = segments[i + 1]["start"]
        return segments

    def _translate_srt(self, segments: list, target_lang: str) -> list | None:
        """
        Translate segments via LLM by sending/receiving SRT format directly.
        Returns translated segments on success, None on failure (caller burns original-only).
        """
        # Build SRT text to send (blank line between blocks is required SRT format)
        srt_text = "\n\n".join(
            f"{i}\n{seg['start']} --> {seg['end']}\n{seg['text']}"
            for i, seg in enumerate(segments, 1)
        )
        prompt = (
            f"Translate this SRT subtitle file to {target_lang}.\n"
            "Return ONLY the translated SRT in the exact same format.\n"
            "Keep all line numbers and timestamps unchanged. Only translate the text lines.\n\n"
            + srt_text
        )
        try:
            response = self.client.simple_chat(prompt, model=self.model)
            # Strip markdown code fences the LLM may wrap the response in
            response = re.sub(r"^```[a-z]*\n?", "", response.strip(), flags=re.MULTILINE)
            response = re.sub(r"^```$", "", response, flags=re.MULTILINE)
            translated = self._parse_srt_text(response)
            if len(translated) != len(segments):
                logger.warning(
                    f"Translation returned {len(translated)}/{len(segments)} segments; "
                    "burning original subtitles only."
                )
                return None
            return translated
        except Exception as e:
            logger.warning(f"Translation failed ({e}); burning original subtitles only.")
            return None

    def _srt_time_to_ass(self, t: str) -> str:
        """Convert SRT time 'HH:MM:SS,mmm' to ASS time 'H:MM:SS.cc'."""
        h, m, s_ms = t.split(":")
        s, ms = s_ms.split(",")
        cc = int(ms) // 10
        return f"{int(h)}:{m}:{s}.{cc:02d}"

    def _generate_ass(self, segments: list, translated: list = None) -> str:
        """Build full ASS file content. Includes translation track if provided."""
        lines = [ASS_HEADER]
        for i, seg in enumerate(segments):
            start = self._srt_time_to_ass(seg["start"])
            end = self._srt_time_to_ass(seg["end"])
            text = _SPEAKER_TAG_RE.sub("", seg["text"].replace("\n", " "))
            lines.append(f"Dialogue: 0,{start},{end},Original,,0,0,0,,{text}")
            if translated and i < len(translated):
                tr_text = _SPEAKER_TAG_RE.sub("", translated[i]["text"].replace("\n", " "))
                if tr_text:
                    lines.append(f"Dialogue: 0,{start},{end},Translation,,0,0,0,,{tr_text}")
        return "\n".join(lines) + "\n"

    def _burn_ass(self, mp4: Path, ass: Path, output: Path) -> bool:
        """Run ffmpeg to burn the ASS subtitle file into the video."""
        # ffmpeg's ass= filter struggles with long paths — copy to a short /tmp path
        tmp_fd, tmp_ass_str = tempfile.mkstemp(suffix=".ass")
        os.close(tmp_fd)
        tmp_ass = Path(tmp_ass_str)
        try:
            tmp_ass.write_bytes(ass.read_bytes())
            cmd = [
                "ffmpeg",
                "-i", str(mp4.resolve()),
                "-vf", f"ass={tmp_ass}",
                "-c:a", "copy",
                "-y", str(output.resolve()),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                logger.error(f"ffmpeg subtitle error for {mp4.name}: {r.stderr[-500:]}")
            return r.returncode == 0
        finally:
            tmp_ass.unlink(missing_ok=True)