import asyncio
import edge_tts

VOICE = "en-US-AriaNeural"
RATE  = "+15%"

WARNINGS = {
    "ph_high_very_close":   "Brake now, severe pothole directly ahead.",
    "ph_high_close":        "Reduce speed immediately, severe pothole ahead.",
    "ph_high_medium":       "Slow down, large pothole in your path.",
    "ph_high_far":          "Hazard ahead, prepare to slow down.",
    "ph_medium_very_close": "Avoid this lane if safe, moderate pothole.",
    "ph_medium_close":      "Slow down, moderate pothole ahead.",
    "ph_medium_medium":     "Caution, moderate pothole ahead.",
    "ph_medium_far":        "Monitor road, pothole detected in distance.",
    "ph_low_very_close":    "Minor pothole, proceed carefully.",
    "ph_low_close":         "Minor pothole ahead.",
    "ph_low_medium":        "Minor road damage detected.",
    "ph_low_far":           "Minor road damage far ahead.",
    "ph_fallback":          "Pothole detected, exercise caution.",
    "tl_red_lane":          "Stop, red light, do not proceed.",
    "tl_red_adjacent":      "Red light in adjacent lane, stay alert.",
    "tl_yellow_lane":       "Slow down, yellow light, prepare to stop.",
    "tl_yellow_adjacent":   "Yellow light in adjacent lane.",
    "tl_green_lane":        "Green light, check for crossing hazards.",
    "tl_green_adjacent":    "Green light in adjacent lane.",
    "tl_unknown_lane":      "Caution, traffic light state unclear.",
    "tl_unknown_adjacent":  "Unrelated signal detected.",
    "tl_fallback":          "Traffic light detected, exercise caution.",
}

async def generate():
    out_dir = "frontend/public/audio"
    for key, text in WARNINGS.items():
        path = f"{out_dir}/{key}.mp3"
        tts = edge_tts.Communicate(text, VOICE, rate=RATE)
        await tts.save(path)
        print(f"  saved: {key}.mp3")
    print(f"\nDone — {len(WARNINGS)} files in {out_dir}/")

asyncio.run(generate())
