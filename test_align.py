import asyncio
from subs_diff.types import Segment, Config, LLMMode
from subs_diff.align import align_segments

stt_segs = [
    Segment(index=0, start_ms=1000, end_ms=4000, text="Они отправились на просмотр фильма, однако мест не оказалось.", tokens=[]),
    Segment(index=1, start_ms=4500, end_ms=8000, text="Следовательно, компания направилась перекусить в заведение.", tokens=[])
]

ref_segs = [
    Segment(index=0, start_ms=1000, end_ms=2500, text="Ребята пошли в кино."),
    Segment(index=1, start_ms=2500, end_ms=4000, text="Но билеты были распроданы."),
    Segment(index=2, start_ms=4500, end_ms=8000, text="В итоге они поужинали в кафе.")
]

def test():
    res = align_segments(stt_segs, ref_segs, time_tol=1.0, max_merge=5)
    
    print("ALIGNED PAIRS:")
    for a, b in res.aligned_pairs:
        print(f"A: [{a.start_ms}-{a.end_ms}] {a.text}")
        print(f"B: [{b.start_ms}-{b.end_ms}] {b.text}")
        print("---")
        
    print(f"Unmatched A: {len(res.unmatched_a)}")
    print(f"Unmatched B: {len(res.unmatched_b)}")

if __name__ == "__main__":
    test()
