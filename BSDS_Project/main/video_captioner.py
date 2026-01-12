#!/usr/bin/env python3
"""
LLaVA-NeXT Video Captioner for Animation Analysis
Generates detailed captions capturing narrative progression and emotional arcs.

Strategy:
1. First pass: Analyze entire video (64 frames) to capture overall narrative arc
2. Second pass: Detailed segment analysis with cumulative context from previous segments
3. Final synthesis: Combine all information for comprehensive understanding
"""

import argparse
import gc
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm


def load_model(model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2"):
    """Load LLaVA-Video model and processors."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)

    print(f"Loading model: {model_path}")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        None,
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",  # Auto-distributes across available GPUs
        attn_implementation="eager",  # Disable Flash Attention for compatibility
    )
    model.eval()

    return tokenizer, model, image_processor, max_length


def extract_frames_uniform(
    video_path: str,
    num_frames: int = 64,
) -> tuple[np.ndarray, list[float], float]:
    """
    Extract uniformly sampled frames from entire video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        frames: numpy array of frames
        frame_times: list of timestamps for each frame
        total_duration: total video duration
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    total_duration = total_frames / video_fps

    # Uniform sampling across entire video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(frame_indices).asnumpy()
    frame_times = [(idx / video_fps) for idx in frame_indices]

    return frames, frame_times, total_duration


def extract_frames_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    max_frames: int = 32,
    target_fps: float = 2.0,
) -> tuple[np.ndarray, list[float], float]:
    """
    Extract frames from a specific video segment.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = int(start_time * video_fps)
    end_frame = min(int(end_time * video_fps), total_frames)

    segment_frames = end_frame - start_frame
    segment_duration = segment_frames / video_fps

    if target_fps > 0:
        sample_interval = int(video_fps / target_fps)
        frame_indices = list(range(start_frame, end_frame, max(1, sample_interval)))
    else:
        frame_indices = list(range(start_frame, end_frame))

    if len(frame_indices) > max_frames:
        indices = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in indices]

    frames = vr.get_batch(frame_indices).asnumpy()
    frame_times = [(idx / video_fps) for idx in frame_indices]

    return frames, frame_times, segment_duration


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    return str(timedelta(seconds=int(seconds)))[2:7]


def generate_caption(
    model,
    tokenizer,
    image_processor,
    frames: np.ndarray,
    prompt: str,
    max_new_tokens: int = 1024,
) -> str:
    """Generate caption for video frames using LLaVA-Video."""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    processed_frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    processed_frames = processed_frames.to(model.device, dtype=torch.bfloat16)

    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    # Create attention mask to avoid warning
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=[processed_frames],
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extract response after assistant marker
    if "<|im_start|>assistant" in output:
        output = output.split("<|im_start|>assistant")[-1].strip()
    if output.endswith("<|im_end|>"):
        output = output[:-len("<|im_end|>")].strip()

    return output


def analyze_full_video(
    model,
    tokenizer,
    image_processor,
    video_path: str,
    num_frames: int = 64,
) -> tuple[str, float]:
    """
    First pass: Analyze entire video to capture overall narrative arc.

    Returns:
        overall_narrative: High-level narrative summary
        total_duration: Video duration in seconds
    """
    print("\n[Phase 1] Analyzing full video for overall narrative context...")

    frames, frame_times, total_duration = extract_frames_uniform(video_path, num_frames)
    print(f"  Sampled {len(frames)} frames across {format_timestamp(total_duration)}")

    prompt = """You are an expert animation analyst. Watch this entire animation carefully and provide a comprehensive narrative overview.

Describe:
1. **Opening Setup**: How the story begins, initial setting, and characters introduced
2. **Narrative Arc**: The main plot progression, key events, and turning points
3. **Character Dynamics**: Who are the main characters, their relationships, and how they develop
4. **Emotional Journey**: The emotional beats throughout - moments of tension, joy, sadness, resolution
5. **Visual Style**: The animation's artistic approach, color palette, and visual storytelling techniques
6. **Conclusion**: How the story resolves and the final emotional state

This overview will serve as context for more detailed scene-by-scene analysis. Be thorough but concise.

Narrative Overview:"""

    narrative = generate_caption(
        model, tokenizer, image_processor, frames, prompt, max_new_tokens=1500
    )

    del frames
    torch.cuda.empty_cache()

    return narrative, total_duration


def analyze_segment_with_context(
    model,
    tokenizer,
    image_processor,
    video_path: str,
    segment_info: dict,
    overall_narrative: str,
    previous_captions: list[str],
    max_frames: int = 32,
    target_fps: float = 2.0,
) -> str:
    """
    Analyze a segment with full context from overall narrative and previous segments.

    Args:
        segment_info: Contains start_time, end_time, segment_num, total_segments
        overall_narrative: High-level narrative from first pass
        previous_captions: List of captions from previous segments (for continuity)
    """
    frames, frame_times, duration = extract_frames_segment(
        video_path,
        segment_info["start_time"],
        segment_info["end_time"],
        max_frames=max_frames,
        target_fps=target_fps,
    )

    # Build context from previous segments
    context_parts = []

    # Add overall narrative context
    context_parts.append(f"""Overall Story Context:
{overall_narrative}
""")

    # Add previous segment summaries (last 2-3 for continuity)
    if previous_captions:
        recent_context = previous_captions[-3:]  # Last 3 segments
        context_parts.append("Recent Events (for narrative continuity):")
        for i, caption in enumerate(recent_context):
            seg_num = segment_info["segment_num"] - len(recent_context) + i
            context_parts.append(f"[Previous Segment {seg_num}]: {caption[:500]}...")  # Truncate for token efficiency

    context = "\n".join(context_parts)

    # Position-specific instructions
    position = segment_info["segment_num"] / segment_info["total_segments"]
    if position <= 0.2:
        position_hint = "This is early in the story - focus on character introduction, setting establishment, and initial conflicts."
    elif position <= 0.4:
        position_hint = "This is the rising action - focus on developing tensions, character relationships, and building toward conflict."
    elif position <= 0.6:
        position_hint = "This is the middle of the story - focus on major developments, turning points, and emotional peaks."
    elif position <= 0.8:
        position_hint = "This is the falling action - focus on consequences, character reactions, and movement toward resolution."
    else:
        position_hint = "This is the conclusion - focus on resolution, emotional payoff, and final character states."

    prompt = f"""You are analyzing segment {segment_info['segment_num']}/{segment_info['total_segments']} ({format_timestamp(segment_info['start_time'])} - {format_timestamp(segment_info['end_time'])}) of an animation.

{context}

{position_hint}

For this specific segment, describe in detail:

1. **Scene Description**: What is happening visually? Describe the setting, characters present, and their positions/poses.

2. **Actions & Events**: What specific actions occur? What plot developments happen?

3. **Emotional Beats**: What emotions are conveyed? How do characters' expressions and body language communicate feelings? What is the mood?

4. **Continuity**: How does this segment connect to what came before? What narrative threads continue or change?

5. **Visual Storytelling**: Note any significant camera angles, lighting changes, color shifts, or visual metaphors.

Write a flowing, detailed caption that would allow someone to vividly imagine this segment while understanding how it fits into the larger narrative:

Segment Caption:"""

    caption = generate_caption(
        model, tokenizer, image_processor, frames, prompt, max_new_tokens=1024
    )

    del frames
    torch.cuda.empty_cache()

    return caption


def process_video(
    video_path: str,
    output_path: str,
    segment_duration: float = 30.0,
    max_frames_per_segment: int = 32,
    target_fps: float = 2.0,
    model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
    full_video_frames: int = 64,
):
    """
    Process video with two-pass approach:
    1. Full video analysis for overall narrative
    2. Segment-by-segment analysis with cumulative context
    """
    # Get video info
    vr = VideoReader(video_path, ctx=cpu(0))
    total_duration = len(vr) / vr.get_avg_fps()
    video_fps = vr.get_avg_fps()
    del vr

    print(f"Video: {video_path}")
    print(f"Duration: {format_timestamp(total_duration)} ({total_duration:.1f}s)")
    print(f"FPS: {video_fps:.2f}")

    # Load model
    tokenizer, model, image_processor, _ = load_model(model_path)

    # Initialize results
    results = {
        "video_path": str(video_path),
        "total_duration": total_duration,
        "model": model_path,
        "settings": {
            "segment_duration": segment_duration,
            "max_frames_per_segment": max_frames_per_segment,
            "target_fps": target_fps,
            "full_video_frames": full_video_frames,
        },
        "overall_narrative": "",
        "segments": [],
    }

    # ============================================
    # Phase 1: Full Video Analysis
    # ============================================
    overall_narrative, _ = analyze_full_video(
        model, tokenizer, image_processor, video_path, full_video_frames
    )
    results["overall_narrative"] = overall_narrative
    print(f"\nOverall Narrative:\n{overall_narrative[:500]}...\n")

    # Save intermediate results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ============================================
    # Phase 2: Segment-by-Segment with Context
    # ============================================
    print(f"\n[Phase 2] Detailed segment analysis with narrative context...")

    num_segments = int(np.ceil(total_duration / segment_duration))
    previous_captions = []

    for i in tqdm(range(num_segments), desc="Processing segments"):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)

        segment_info = {
            "segment_num": i + 1,
            "total_segments": num_segments,
            "start_time": start_time,
            "end_time": end_time,
        }

        print(f"\nSegment {i+1}/{num_segments}: {format_timestamp(start_time)} - {format_timestamp(end_time)}")

        # Analyze segment with full context
        caption = analyze_segment_with_context(
            model, tokenizer, image_processor,
            video_path,
            segment_info,
            overall_narrative,
            previous_captions,
            max_frames=max_frames_per_segment,
            target_fps=target_fps,
        )

        # Store result
        segment_result = {
            "segment_id": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": f"{format_timestamp(start_time)} - {format_timestamp(end_time)}",
            "caption": caption,
        }
        results["segments"].append(segment_result)
        previous_captions.append(caption)

        # Save intermediate results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Clean up memory after each segment
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================
    # Phase 3: Final Synthesis (TEXT-ONLY - No video frames)
    # ============================================
    print("\n[Phase 3] Generating final synthesis (text-only mode)...")

    # Clear GPU memory before Phase 3
    gc.collect()
    torch.cuda.empty_cache()

    # Combine segment captions (truncated for memory)
    all_captions_text = "\n".join([
        f"[{seg['timestamp']}]: {seg['caption'][:200]}"  # Shorter truncation
        for seg in results["segments"]
    ])

    # TEXT-ONLY synthesis using the model's text generation capability
    # No video frames = much less GPU memory
    synthesis_prompt = f"""Based on the following animation analysis, write a cohesive summary.

OVERALL NARRATIVE:
{overall_narrative[:1000]}

SEGMENT DETAILS:
{all_captions_text[:1500]}

Write a 2-3 paragraph summary covering the story arc, main characters, emotional journey, and themes:"""

    # Use minimal frames (just 4) as visual anchor, or skip entirely
    try:
        # Try with minimal frames first
        frames, _, _ = extract_frames_uniform(video_path, 4)
        print("  Using 4 frames as visual anchor")

        final_synthesis = generate_caption(
            model, tokenizer, image_processor, frames, synthesis_prompt, max_new_tokens=512
        )
        del frames
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  OOM with frames, falling back to text-only synthesis")
            torch.cuda.empty_cache()
            gc.collect()

            # Fallback: Generate synthesis from text alone (no model call)
            final_synthesis = f"""[Text-based synthesis from collected analysis]

{overall_narrative[:800]}

The animation progresses through {len(results['segments'])} distinct segments, each building on the narrative established in the overall story. The detailed segment analysis reveals the emotional progression and visual storytelling techniques employed throughout.

Key segments include:
{chr(10).join([f"- {seg['timestamp']}: {seg['caption'][:150]}..." for seg in results['segments'][:5]])}

This analysis provides a comprehensive understanding of the animation's narrative structure, character development, and thematic content."""
        else:
            raise e

    gc.collect()
    torch.cuda.empty_cache()

    results["final_synthesis"] = final_synthesis

    # Save final results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save readable text version
    text_output_path = output_path.replace(".json", ".txt")
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"ANIMATION CAPTION ANALYSIS\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Duration: {format_timestamp(total_duration)}\n\n")

        f.write(f"{'=' * 60}\n")
        f.write(f"FINAL SYNTHESIS\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"{final_synthesis}\n\n")

        f.write(f"{'=' * 60}\n")
        f.write(f"OVERALL NARRATIVE (First Pass)\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"{overall_narrative}\n\n")

        f.write(f"{'=' * 60}\n")
        f.write(f"DETAILED SEGMENT-BY-SEGMENT CAPTIONS\n")
        f.write(f"{'=' * 60}\n\n")
        for segment in results["segments"]:
            f.write(f"[{segment['timestamp']}]\n")
            f.write(f"{segment['caption']}\n\n")
            f.write(f"{'-' * 40}\n\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {output_path}")
    print(f"  Text: {text_output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed captions for animation videos using LLaVA-NeXT"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON path (default: video_name_captions.json)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=30.0,
        help="Duration of each segment in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum frames per segment (default: 32)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=2.0,
        help="Target FPS for frame sampling (default: 2.0)",
    )
    parser.add_argument(
        "--full-video-frames",
        type=int,
        default=64,
        help="Frames for full video analysis (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lmms-lab/LLaVA-Video-7B-Qwen2",
        help="Model path (default: lmms-lab/LLaVA-Video-7B-Qwen2)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    if args.output is None:
        video_name = Path(args.video_path).stem
        args.output = f"{video_name}_captions.json"

    process_video(
        video_path=args.video_path,
        output_path=args.output,
        segment_duration=args.segment_duration,
        max_frames_per_segment=args.max_frames,
        target_fps=args.target_fps,
        model_path=args.model,
        full_video_frames=args.full_video_frames,
    )


if __name__ == "__main__":
    main()
