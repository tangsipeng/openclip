import asyncio
from pathlib import Path

from video_orchestrator import VideoOrchestrator


def test_skip_transcript_uses_existing_local_subtitle_for_single_part_video(tmp_path, monkeypatch):
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"fake-video")
    source_subtitle = tmp_path / "input.srt"
    source_subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
        encoding="utf-8",
    )

    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        skip_analysis=True,
        generate_clips=False,
        generate_cover=False,
    )

    async def fake_is_local_video_file(_source: str) -> bool:
        return True

    async def fake_process_local_video(_video_path: str, _progress_callback):
        return {
            "video_path": str(source_video),
            "video_info": {
                "title": "input",
                "duration": 60,
                "uploader": "Local File",
            },
            "subtitle_path": str(source_subtitle),
        }

    monkeypatch.setattr(orchestrator, "_is_local_video_file", fake_is_local_video_file)
    monkeypatch.setattr(orchestrator, "_process_local_video", fake_process_local_video)

    result = asyncio.run(
        orchestrator.process_video(
            str(source_video),
            skip_transcript=True,
            progress_callback=None,
        )
    )

    expected_subtitle = (
        Path(orchestrator.output_dir)
        / "input"
        / "splits"
        / "input_part01.srt"
    )

    assert result.success is True
    assert result.transcript_source == "existing"
    assert result.transcript_parts == [str(expected_subtitle)]
    assert expected_subtitle.exists()
