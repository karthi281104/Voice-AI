"""
VoiceDesk Real-Time Voice Engine v0.2 — Latency Benchmark
Measures turn-by-turn timestamps (T0 to T7) and calculates:
- First audible latency: T7 - T0
- STT latency: T2 - T0
- LLM first-token latency: T4 - T3
- TTS startup latency: T6 - T5
- Interruption cancellation: < 200ms
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass


@dataclass
class VoiceTurnTimestamps:
    turn_id: int
    language: str
    t0_speech_ends: float = 0.0
    t1_vad_confirmed: float = 0.0
    t2_stt_ready: float = 0.0
    t3_llm_starts: float = 0.0
    t4_llm_first_token: float = 0.0
    t5_first_phrase_ready: float = 0.0
    t6_tts_audio_ready: float = 0.0
    t7_playback_begins: float = 0.0

    @property
    def vad_endpointing_ms(self) -> float:
        return max(0.0, (self.t1_vad_confirmed - self.t0_speech_ends) * 1000)

    @property
    def stt_latency_ms(self) -> float:
        return max(0.0, (self.t2_stt_ready - self.t0_speech_ends) * 1000)

    @property
    def llm_first_token_ms(self) -> float:
        return max(0.0, (self.t4_llm_first_token - self.t3_llm_starts) * 1000)

    @property
    def tts_startup_ms(self) -> float:
        return max(0.0, (self.t6_tts_audio_ready - self.t5_first_phrase_ready) * 1000)

    @property
    def first_audible_latency_ms(self) -> float:
        return max(0.0, (self.t7_playback_begins - self.t0_speech_ends) * 1000)


def print_benchmark_summary(turns: list[VoiceTurnTimestamps]) -> None:
    if not turns:
        print("No turns recorded.")
        return

    vad_times = [t.vad_endpointing_ms for t in turns]
    stt_times = [t.stt_latency_ms for t in turns]
    llm_times = [t.llm_first_token_ms for t in turns]
    tts_times = [t.tts_startup_ms for t in turns]
    audible_times = [t.first_audible_latency_ms for t in turns]

    print("\n" + "=" * 65)
    print(" VoiceDesk AI — Real-Time Voice Engine v0.2 Latency Benchmark")
    print("=" * 65)
    print(f"Total turns evaluated: {len(turns)}")
    print("-" * 65)
    print(f"{'Metric':<30} | {'Target':<14} | {'Median':<10} | {'P90':<10}")
    print("-" * 65)

    def p90(arr):
        return sorted(arr)[int(len(arr) * 0.9)] if arr else 0.0

    print(f"{'VAD Endpointing (T1-T0)':<30} | {'150-300 ms':<14} | {statistics.median(vad_times):<10.1f} | {p90(vad_times):<10.1f}")
    print(f"{'STT Latency (T2-T0)':<30} | {'100-400 ms':<14} | {statistics.median(stt_times):<10.1f} | {p90(stt_times):<10.1f}")
    print(f"{'LLM First Token (T4-T3)':<30} | {'< 500 ms':<14} | {statistics.median(llm_times):<10.1f} | {p90(llm_times):<10.1f}")
    print(f"{'TTS Startup (T6-T5)':<30} | {'< 300 ms':<14} | {statistics.median(tts_times):<10.1f} | {p90(tts_times):<10.1f}")
    print(f"{'First Audible Latency (T7-T0)':<30} | {'< 1200 ms':<14} | {statistics.median(audible_times):<10.1f} | {p90(audible_times):<10.1f}")
    print("=" * 65 + "\n")


def simulate_sample_benchmark():
    """Generates synthetic benchmark turns representing target milestones."""
    turns = []
    for i in range(1, 11):
        t0 = time.perf_counter()
        t1 = t0 + 0.220  # 220 ms VAD
        t2 = t0 + 0.380  # 380 ms STT
        t3 = t2 + 0.010
        t4 = t3 + 0.320  # 320 ms LLM first token
        t5 = t4 + 0.080  # first phrase ready
        t6 = t5 + 0.090  # TTS audio chunk ready
        t7 = t6 + 0.040  # browser playback starts: ~860ms total audible latency!

        turns.append(VoiceTurnTimestamps(
            turn_id=i,
            language="en-IN" if i % 2 == 0 else "ta-IN",
            t0_speech_ends=t0,
            t1_vad_confirmed=t1,
            t2_stt_ready=t2,
            t3_llm_starts=t3,
            t4_llm_first_token=t4,
            t5_first_phrase_ready=t5,
            t6_tts_audio_ready=t6,
            t7_playback_begins=t7,
        ))
    print_benchmark_summary(turns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceDesk Voice Latency Benchmark")
    parser.add_argument("--simulate", action="store_true", help="Run simulated benchmark test")
    args = parser.parse_args()

    if args.simulate:
        simulate_sample_benchmark()
    else:
        print("Run with --simulate to view target benchmark breakdown, or connect WebSocket client to capture live turns.")
