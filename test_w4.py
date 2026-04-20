"""Run W4 pipeline smoke test and save output frames."""
import cv2, traceback
from src.pipeline import process_video

print("Running TacticsAI W4 pipeline on tryolabs_demo.mp4...")

results_list = []
try:
    for result in process_video(
        "data/tryolabs_demo.mp4",
        model_path="yolov8x.pt",
        device="cuda",
        skip_frames=2,
        max_frames=20,
        show_progress=True,
        calibration_frames=40,
        enable_team_classification=True,
        enable_bird_eye=True,
    ):
        fi = result["frame_idx"]
        tc = result["team_counts"]
        ps = result["pressing_state"]
        fo = result["formations"]
        print(f"  Frame {fi} | players={result['player_count']} | "
              "A={} B={} | Press={} | Form={}".format(
                  tc.get(0,0), tc.get(1,0), ps, fo.get(0,"?")))

        if result["pitch_view"] is not None and len(results_list) == 0:
            cv2.imwrite("output/w4_birdeye.jpg", result["pitch_view"])
            cv2.imwrite("output/w4_annotated.jpg", result["annotated_frame"])
            if result.get("heatmap_view") is not None:
                cv2.imwrite("output/w4_heatmap.jpg", result["heatmap_view"])
            print("  -> Saved w4_birdeye.jpg, w4_annotated.jpg, w4_heatmap.jpg")
        results_list.append(result)

except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()

if results_list:
    last = results_list[-1]
    print()
    print("Speed summary:", last.get("team_speed_summary", {}))
    print("Pressing events:", [(e.timestamp, e.team_id) for e in last.get("pressing_events", [])])
    print("Done!")
