#!/usr/bin/env bash
#
# preprocess_fps.sh
#
# Normalize the videos in media/videos/raw for timeline CV work (YOLO joint
# tracking + frame-to-seconds mapping) and save 30fps and 60fps copies into
# media/videos/30fps and media/videos/60fps.
#
# Each output is guaranteed to be:
#   * Constant frame rate (CFR)  -> frame_index / fps == real seconds
#   * Physically upright         -> rotation baked into pixels, flag cleared,
#                                   so OpenCV VideoCapture cannot read it sideways
#   * Square pixels (SAR 1:1)    -> no anamorphic distortion of joint coordinates
#
# Usage:
#   ./preprocess_fps.sh                  # process all videos in raw/
#   ./preprocess_fps.sh -f               # force re-encode even if output exists
#   ./preprocess_fps.sh clip.mp4         # process only the named file(s) in raw/
#   ./preprocess_fps.sh --keep-audio     # keep the audio track (dropped by default)
#   ./preprocess_fps.sh --probe          # report fps / VFR / rotation, encode nothing
#   ./preprocess_fps.sh --help
#
# Requires: ffmpeg and ffprobe (on PATH).

set -euo pipefail

# Resolve directories relative to this script's location so it can be run
# from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/raw"
OUT_30="$SCRIPT_DIR/30fps"
OUT_60="$SCRIPT_DIR/60fps"

# Target frame rates and matching output folders.
FPS_LIST=(30 60)
OUT_LIST=("$OUT_30" "$OUT_60")

FORCE=0
KEEP_AUDIO=0
PROBE_ONLY=0
FILES=()

for arg in "$@"; do
  case "$arg" in
    -f|--force)      FORCE=1 ;;
    --keep-audio)    KEEP_AUDIO=1 ;;
    --probe)         PROBE_ONLY=1 ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) FILES+=("$arg") ;;
  esac
done

command -v ffmpeg  >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found on PATH."  >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ERROR: ffprobe not found on PATH." >&2; exit 1; }

mkdir -p "$OUT_30" "$OUT_60"

# Print fps / VFR / rotation for one video. Used by --probe and as a
# post-encode sanity check.
probe_video() {
  local file="$1"
  local r avg rot
  r=$(  ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate   -of csv=p=0 "$file" | tr -d ', ')
  avg=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$file" | tr -d ', ')
  # Rotation can live in a stream tag (older files) or in side data (newer files).
  rot=$(ffprobe -v error -select_streams v:0 \
          -show_entries stream_tags=rotate -show_entries stream_side_data=rotation \
          -of default=nokey=1:noprint_wrappers=1 "$file" | tr -d '[:space:]')

  local cfr="CFR"
  [ "$r" != "$avg" ] && cfr="VFR(!)"
  local rotmsg="upright"
  [ -n "$rot" ] && [ "$rot" != "0" ] && rotmsg="rotation=${rot}(!)"

  printf '   r_fps=%-12s avg_fps=%-14s %-8s %s\n' "$r" "$avg" "$cfr" "$rotmsg"
}

# If no explicit files were given, process every video in raw/.
if [ "${#FILES[@]}" -eq 0 ]; then
  shopt -s nullglob nocaseglob
  for f in "$RAW_DIR"/*.{mp4,mov,mkv,avi,m4v,webm}; do
    FILES+=("$(basename "$f")")
  done
  shopt -u nullglob nocaseglob
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "No videos found in $RAW_DIR" >&2
  exit 1
fi

# --probe: just report what the raw files look like and exit.
if [ "$PROBE_ONLY" -eq 1 ]; then
  echo "Probing $RAW_DIR"
  echo "(VFR(!) means frame_index/fps will NOT equal real seconds until re-encoded;"
  echo " rotation=(!) means OpenCV may read the frames sideways.)"
  echo
  for name in "${FILES[@]}"; do
    [ -f "$RAW_DIR/$name" ] || continue
    echo "$name"
    probe_video "$RAW_DIR/$name"
  done
  exit 0
fi

# Audio handling: CV pipelines never use the audio track, so drop it by default.
if [ "$KEEP_AUDIO" -eq 1 ]; then
  AUDIO_ARGS=(-c:a aac -b:a 192k)
else
  AUDIO_ARGS=(-an)
fi

for name in "${FILES[@]}"; do
  src="$RAW_DIR/$name"
  if [ ! -f "$src" ]; then
    echo "SKIP (not found): $name" >&2
    continue
  fi

  base="${name%.*}"

  for i in "${!FPS_LIST[@]}"; do
    fps="${FPS_LIST[$i]}"
    out_dir="${OUT_LIST[$i]}"
    dst="$out_dir/${base}_${fps}fps.mp4"

    if [ -f "$dst" ] && [ "$FORCE" -ne 1 ]; then
      echo "EXISTS (use -f to overwrite): ${base}_${fps}fps.mp4"
      continue
    fi

    echo ">> $name -> ${fps}fps"

    # -r + -fps_mode cfr : resample to a true constant frame rate, dropping or
    #     duplicating frames as needed. This is what makes
    #     time_seconds = frame_index / fps exact, even for VFR sources.
    # ffmpeg auto-applies any rotation metadata on decode; forcing a filter
    #     pass (scale=iw:ih) bakes that rotation into the actual pixels, and
    #     -metadata:s:v rotate=0 clears any residual flag so OpenCV can't
    #     re-apply or ignore it. setsar=1 forces square pixels.
    # -crf 18 / preset medium : visually near-lossless, reasonable file size.
    ffmpeg -y -hide_banner -loglevel error \
      -i "$src" \
      -vf "scale=iw:ih,setsar=1" \
      -r "$fps" -fps_mode cfr \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
      -metadata:s:v rotate=0 \
      "${AUDIO_ARGS[@]}" \
      -movflags +faststart \
      "$dst"

    # Confirm the output really is CFR and carries no rotation flag.
    probe_video "$dst"
  done
done

echo
echo "Done. Outputs in:"
echo "  $OUT_30"
echo "  $OUT_60"
