#!/usr/bin/env bash
# Lossless DICOM anonymization (mask top-left box) with parallelism
# Layout:
#   INPUT:  /root/<session_date>/<session_id>/.../*.dcm
#   OUTPUT: /out/<session_date>/<session_id>/{DICOM_ANON,ANON_QC,ANON_MOV}
#
# Handles (UNCOMPRESSED only): RGB (planar 0/1), YBR_FULL, YBR_FULL_422,
# YBR_PARTIAL_422, YBR_PARTIAL_420, MONOCHROME1/2 (8/16), PALETTE COLOR.
#
# Requirements in container: bash, coreutils, findutils, dcmdump (dcmtk),
# ffmpeg, python3 + pydicom.

set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""
NUM_THREADS=4
OVERWRITE=0
KEEP_RAW="${KEEP_RAW:-0}"   # set to 1 to keep *.raw intermediates
# Blur single-frame PNGs if there are at least this many black rows from the top
# These have been shown to contain PHI outside of the standard area we are checking.
BLUR_TOP_BLACK_ROWS="${BLUR_TOP_BLACK_ROWS:-5}"

usage() {
  cat <<USAGE
Usage: $0 -i <input_dir> -o <output_dir> [-n <num_threads>] [--overwrite-existing]

Notes:
  * Expects input files under: <input_dir>/<session_date>/<session_id>/.../*.dcm
  * Outputs:
      <output_dir>/<session_date>/<session_id>/DICOM_ANON/*.dcm
      <output_dir>/<session_date>/<session_id>/ANON_MOV/*.mov
      <output_dir>/<session_date>/<session_id>/ANON_IMG/*.png
      <output_dir>/<session_date>/<session_id>/ANON_QC/*.png
  * Only uncompressed transfer syntaxes are edited losslessly. Compressed are skipped.
USAGE
  exit 1
}

# ---------- Black-bar detector (decodes first frame to RGB ONLY FOR ANALYSIS) ----------
detect_black_rows_rgb24_from_dicom() {
  # args: dcm_path rows cols (rows/cols are not strictly needed now)
  local dcm="$1"

  python3 - "$dcm" <<'PY'
import sys, numpy as np
try:
    import pydicom
except Exception:
    print(0); sys.exit(0)

dcm = sys.argv[1]
try:
    ds = pydicom.dcmread(dcm, force=True, stop_before_pixels=False)
    a  = ds.pixel_array  # uses handlers for compressed & uncompressed
except Exception:
    print(0); sys.exit(0)

# Normalize to (F,R,C,3) uint8 **without** manual YCbCr math.
if a.ndim == 2:
    a = a[None, ...]                 # (1,R,C)
if a.ndim == 3:
    if a.shape[-1] == 3:             # (R,C,3)
        a = a[None, ...]             # (1,R,C,3)
    else:                            # (F,R,C) mono
        a = a[..., None]             # (F,R,C,1)

# (F,R,C,S)
if a.shape[-1] == 1:
    a = a.astype(np.uint8)
    a = np.repeat(a, 3, axis=-1)     # MONO→RGB
else:
    a = a[..., :3].astype(np.uint8)  # drop alpha if present

# Use luminance (Rec.601) to detect “black” rows
Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
R = Y.shape[1]; C = Y.shape[2]
threshold = 30.0      # Y <= 30 is “black”
pct       = 0.90      # ≥90% pixels dark
need      = 3         # 3 consecutive rows

streak = 0
start  = -1
for y in range(R):
    dark = (Y[0,y] <= threshold).mean()
    if dark >= pct:
        if streak == 0: start = y
        streak += 1
        if streak >= need:
            print(start + streak)
            sys.exit(0)
    else:
        streak = 0
        start  = -1

print(0)
sys.exit(0)
PY
}

# ---------- Top-of-image black-run detector (rows from top only) ----------
detect_top_black_rows_rgb24_from_dicom() {
  # args: dcm_path
  local dcm="$1"
  python3 - "$dcm" <<'PY'
import sys, numpy as np
try:
    import pydicom
except Exception:
    print(0); sys.exit(0)

dcm = sys.argv[1]
try:
    ds = pydicom.dcmread(dcm, force=True, stop_before_pixels=False)
    a  = ds.pixel_array
except Exception:
    print(0); sys.exit(0)

# Normalize to (F,R,C,3) uint8
if a.ndim == 2:
    a = a[None, ...]
if a.ndim == 3:
    if a.shape[-1] == 3:
        a = a[None, ...]
    else:
        a = a[..., None]
if a.shape[-1] == 1:
    a = a.astype(np.uint8)
    a = np.repeat(a, 3, axis=-1)
else:
    a = a[..., :3].astype(np.uint8)

# Luma
Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
R = Y.shape[1]
threshold = 30.0   # black
pct       = 0.90   # ≥90% pixels in a row are "black"

# Count consecutive black rows starting at the very top/Mask box

top_run = 0
for y in range(R):
    if (Y[0,y] <= threshold).mean() >= pct:
        top_run += 1
    else:
        break

print(top_run)
sys.exit(0)
PY
}

# ---------- Per-file worker ----------
process_one() {
  local dcm="$1" inroot="$2" outroot="$3"

  # Preserve full relative directory from input root
  local ddir rel_full
  ddir="$(dirname "$dcm")"
  rel_full="$(python3 - "$inroot" "$ddir" <<'PY'
import os,sys
root=os.path.realpath(sys.argv[1]); d=os.path.realpath(sys.argv[2])
try:
    print(os.path.relpath(d,root))
except Exception:
    print("")
PY
)"
  [[ -z "$rel_full" ]] && rel_full="unknown"

  # Output dirs: append category folders under the original relative path
  local out_base="$outroot/$rel_full"
  local out_dcm_dir="$out_base/DICOM_ANON"
  local out_qc_dir="$out_base/ANON_QC"
  local out_mov_dir="$out_base/ANON_MOV"
  local out_img_dir="$out_base/ANON_IMG"
  mkdir -p "$out_dcm_dir" "$out_qc_dir" "$out_mov_dir" "$out_img_dir"


  # Pull minimal metadata via dcmdump (fast & robust)
  tag() {
    local file="$1" key="$2"
    dcmdump -q +P "$key" "$file" 2>/dev/null | awk '
      {
        sub(/.*\) [A-Z][A-Z] ?/, "", $0);
        sub(/#.*/ , "", $0);
        gsub(/^[ \t]+|[ \t]+$/, "", $0);
        if ($0 ~ /^\[.*\]$/) { sub(/^\[/,""); sub(/\]$/ ,"") }
        sub(/^=[ \t]*/, "", $0);
        print $0; exit
      }'
  }

  normalize_tsuid() {
    case "$1" in
      LittleEndianImplicit|ImplicitVRLittleEndian) echo 1.2.840.10008.1.2 ;;
      LittleEndianExplicit|ExplicitVRLittleEndian) echo 1.2.840.10008.1.2.1 ;;
      BigEndianExplicit|ExplicitVRBigEndian)       echo 1.2.840.10008.1.2.2 ;;
      *) echo "$1" ;;
    esac
  }

  local rows cols bits pi frames planar tsuid
  rows="$(tag "$dcm" 0028,0010)"; cols="$(tag "$dcm" 0028,0011)"
  bits="$(tag "$dcm" 0028,0100)"; pi="$(tag "$dcm" 0028,0004)"
  frames="$(tag "$dcm" 0028,0008)"; planar="$(tag "$dcm" 0028,0006)"
  tsuid_raw="$(tag "$dcm" 0002,0010)"; tsuid="$(normalize_tsuid "$tsuid_raw")"
  [[ -z "${frames:-}" ]] && frames=1
  [[ -z "${planar:-}" ]] && planar=0

  echo -e "\n🧩 [$rel_full] $(basename "$dcm")"
  echo "  Rows=$rows Cols=$cols Bits=$bits PI=$pi Frames=$frames Planar=$planar TSUID=$tsuid"

  # Endianness from TS
  local endian="LE"
  [[ "$tsuid" == "1.2.840.10008.1.2.2" ]] && endian="BE"

  # Allow uncompressed and JPEG Baseline (decode once, no extra loss)
  case "$tsuid" in
    1.2.840.10008.1.2|1.2.840.10008.1.2.1|1.2.840.10008.1.2.2)
      codec="uncompressed"
      ;;
    1.2.840.10008.1.2.4.50|JPEGBaseline)
      codec="jpeg-baseline"
      ;;
    *)
      echo "  ⚠️  SKIP (compressed TSUID=$tsuid not supported yet)."
      return 0
      ;;
  esac
  echo "  Pixel codec: $codec"

  # Compute mask box (detect top black rows by decoding 1 frame ONLY for analysis)
  local box_height box_width
  #box_height="$(detect_black_rows_rgb24_from_dicom "$dcm" "$rows" "$cols" || echo 0)"
  #box_height="$(detect_black_rows_rgb24_from_dicom "$dcm" "$rows" "$cols")"
  # strip any stray non-digits or newlines, default to 0
  box_height="$(detect_black_rows_rgb24_from_dicom "$dcm")" 
  box_height="${box_height//$'\n'/}"
  box_height="${box_height//[^0-9]/}"
  : "${box_height:=0}"
  [[ -z "$box_height" ]] && box_height=0
  # Width heuristic (your existing logic)
  if (( box_height < 40 )); then box_width=$(( cols * 2 / 3 )); else box_width=$(( cols / 3 )); fi
  echo "  Mask box: height=$box_height width=$box_width"

  # For single-frame PNGs, decide if we should blur the whole image
  local threshold="${BLUR_TOP_BLACK_ROWS:-5}"
  top_black="$(detect_top_black_rows_rgb24_from_dicom "$dcm")"
  top_black="${top_black//$'\n'/}"; top_black="${top_black//[^0-9]/}"; : "${top_black:=0}"
  
  phi_blur=0
  if [[ "${frames:-1}" -le 1 ]]; then
    if [[ "$top_black" -ge "$BLUR_TOP_BLACK_ROWS" ]]; then
      phi_blur=1
    fi
  fi
  echo "  PHI risk (single-frame): phi_blur=$phi_blur (top_black_rows=$top_black, threshold_rows=$BLUR_TOP_BLACK_ROWS)"
  
  local base; base="$(basename "${dcm%.*}")"
  local out_dcm="$out_dcm_dir/${base}_anon.dcm"
  local out_mov="$out_mov_dir/${base}_anon_raw.mov"
  local out_png="$out_qc_dir/${base}_anon_mid.png"
  local out_img="$out_img_dir/${base}_anon.png"

  # Skip already-processed unless --overwrite-existing
  if [[ "$OVERWRITE" -eq 0 ]]; then
    if [[ "${frames:-1}" -le 1 ]]; then
      if [[ -s "$out_dcm" && -s "$out_img" ]]; then
        echo "  ⏭  Skip (exists: DCM & IMG)"
        return 0
      fi
    else
      if [[ -s "$out_dcm" && -s "$out_mov" && -s "$out_png" ]]; then
        echo "  ⏭  Skip (exists: DCM & MOV & PNG)"
        return 0
      fi
    fi
  fi

  # Try to read FrameTime (0018,1063) in ms → fps; default 30
  ft_ms_raw="$(tag "$dcm" 0018,1063 || true)"
  # strip non-numeric (keeps digits and dot)
  ft_ms="${ft_ms_raw//[^0-9.]}"
  if [[ -n "${ft_ms:-}" && "$ft_ms" != "." ]]; then
    fps="$(python3 - "$ft_ms" <<'PY'
import sys
try:
    ft = float(sys.argv[1])
    print(max(1, min(240, round(1000.0/ft))))  # clamp
except Exception:
    print(30)
PY
)"
  else
    fps=30
  fi
  echo "  FPS=$fps (from FrameTime=${ft_ms:-N/A})"

  # Path for rgb24 raw stream (all frames)
  raw_rgb="$out_mov.rgb24.raw"
  raw_gray16="$out_img.gray16le.raw"

  cleanup_raws() {
    [[ "$KEEP_RAW" -eq 1 ]] && return 0
    local deleted=0
    if [[ -n "${raw_rgb:-}"    && -e "$raw_rgb"    ]]; then rm -f -- "$raw_rgb";    deleted=1; fi
    if [[ -n "${raw_gray16:-}" && -e "$raw_gray16" ]]; then rm -f -- "$raw_gray16"; deleted=1; fi
    [[ $deleted -eq 1 ]] && echo "🧹 cleaned intermediate raw files"
  }
  
  trap cleanup_raws RETURN

  # Call Python to LOSSLESSLY mask native Pixel Data inside DICOM
  ROWS="$rows" COLS="$cols" BITS="$bits" FRAMES="$frames" \
  PI="$pi" TSUID="$tsuid" PLANAR="$planar" ENDIAN="$endian" \
  BOXH="$box_height" BOXW="$box_width" \
  SRC_DCM="$dcm" OUT_DCM="$out_dcm" \
  RAW_OUT="$raw_rgb" \
  RAW_GRAY16_OUT="$raw_gray16" \
  python3 - <<'PY'
import os, sys
import numpy as np
import pydicom
from pydicom.tag import Tag
from pydicom.uid import UID, ExplicitVRLittleEndian

# --------- Env ----------
src     = os.environ["SRC_DCM"]
outp    = os.environ["OUT_DCM"]
raw_out = os.environ.get("RAW_OUT")
raw_gray16_out = os.environ.get("RAW_GRAY16_OUT")
boxh    = int(os.environ.get("BOXH", "0") or 0)
boxw    = int(os.environ.get("BOXW", "0") or 0)

# --------- Read dataset ----------
ds = pydicom.dcmread(src, force=True, stop_before_pixels=False)

# Geometry
rows  = int(getattr(ds, "Rows", 0) or 0)
cols  = int(getattr(ds, "Columns", 0) or 0)
frames_attr = getattr(ds, "NumberOfFrames", 1)
try:
    frames = int(str(frames_attr))
except Exception:
    frames = 1
if rows <= 0 or cols <= 0:
    print("ERROR: Invalid geometry", file=sys.stderr); sys.exit(3)

# Transfer syntax: compressed?
tsuid = str(getattr(ds.file_meta, "TransferSyntaxUID", "") or "")
UNCOMP = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}
compressed = tsuid not in UNCOMP

# Photometric & other tags
pi  = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
bits = int(getattr(ds, "BitsAllocated", getattr(ds, "BitsStored", 8)) or 8)
planar = int(getattr(ds, "PlanarConfiguration", 0) or 0)

# --------- Build arr_rgb for MOV (works for compressed & uncompressed) ----------
# We decode via handlers to numpy and normalize to uint8 RGB (F,R,C,3).
def to_rgb_uint8(arr, pi_hint):
    import numpy as _np
    # Ensure frames dimension
    if arr.ndim == 2:                 # (R,C)
        arr = arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[-1] == 3:        # (R,C,3)
            arr = arr[None, ...]
        else:                          # (F,R,C) mono
            arr = arr[..., None]
    # Now (F,R,C,S)
    if arr.shape[-1] == 1:            # MONO → RGB
        arr = arr.astype(_np.uint8)
        arr = _np.repeat(arr, 3, axis=-1)
    else:
        arr = arr[..., :3].astype(_np.uint8)  # drop alpha if present
    return arr

arr_rgb = None
try:
    arr_dec = ds.pixel_array  # decodes compressed (needs pylibjpeg/gdcm) and uncompressed
    arr_rgb = to_rgb_uint8(arr_dec, pi)
    frames = arr_rgb.shape[0]  # trust decoded frames count
except Exception as e:
    print(f"NOTE: could not build arr_rgb for MOV ({e}); MOV may be skipped.", file=sys.stderr)

# Clamp ROI within image
boxw = max(0, min(boxw, cols))
boxh = max(0, min(boxh, rows))

# If we have an RGB decode, mirror the mask for the MOV output
if arr_rgb is not None and boxw > 0 and boxh > 0:
    arr_rgb[:, 0:boxh, 0:boxw, :] = 0

# --------- UNCOMPRESSED path: keep your PI-specific byte masking & TS ---------
if not compressed:
    # Prepare raw bytearray for native masking (your existing logic)
    pb = ds.PixelData
    pix = bytearray(pb if isinstance(pb, (bytes, bytearray)) else bytes(pb))

    # Helpers
    def put16LE(buf, off, v): buf[off]=v&0xFF; buf[off+1]=(v>>8)&0xFF
    def put16BE(buf, off, v): buf[off]=(v>>8)&0xFF; buf[off+1]=v&0xFF
    big = (UID(tsuid) == UID("1.2.840.10008.1.2.2"))
    put16 = put16BE if big else put16LE

    # Per-frame size by PI
    def frame_bytes(pi_s):
        if pi_s in ("RGB","YBR_FULL"):
            if bits != 8: raise ValueError
            return rows*cols*3
        if pi_s in ("YBR_FULL_422","YBR_PARTIAL_422"):
            if bits != 8: raise ValueError
            return rows*cols*2
        if pi_s == "YBR_PARTIAL_420":
            if bits != 8: raise ValueError
            return rows*cols + (rows//2)*(cols//2)*2
        if pi_s in ("MONOCHROME1","MONOCHROME2"):
            return rows*cols*(1 if bits==8 else 2)
        if pi_s == "PALETTE COLOR":
            if bits != 8: raise ValueError
            return rows*cols
        raise ValueError(f"Unsupported PI={pi_s} Bits={bits}")

    try:
        pfb = frame_bytes(pi)
    except Exception as e:
        print(f"ERROR: Unsupported PI={pi} Bits={bits}", file=sys.stderr); sys.exit(2)

    total = len(pix)
    if frames * pfb != total:
        frames = total // pfb

    # If no mask requested, still produce MOV raw (from arr_rgb) and pass-through write
    if boxw == 0 or boxh == 0:
        if arr_rgb is not None and raw_out:
            try:
                with open(raw_out, "wb") as fh:
                    fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
                print(f"Wrote rgb24 raw: {raw_out}")
            except Exception as e:
                print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)
        pydicom.dcmwrite(outp, ds, write_like_original=True)
        print(f"Wrote DICOM (pass-through): {outp}")
        sys.exit(0)

    # Constants for YBR/mono masking
    YBLACK_FULL    = 0
    YBLACK_PARTIAL = 16
    MID            = 128
    maxmono        = (1<<bits) - 1

    # Byte-level mask (your existing cases)
    for f in range(frames):
        base = f*pfb
        if pi == "RGB":
            if int(planar) == 0:
                row_span = cols*3
                for y in range(boxh):
                    off = base + y*row_span
                    for i in range(off, off+boxw*3, 3):
                        pix[i]=0; pix[i+1]=0; pix[i+2]=0
            else:
                plane_sz = rows*cols
                R0 = base; G0 = base + plane_sz; B0 = base + 2*plane_sz
                for y in range(boxh):
                    for x in range(boxw):
                        p = y*cols + x
                        pix[R0+p] = 0; pix[G0+p] = 0; pix[B0+p] = 0

        elif pi == "YBR_FULL":
            row_span = cols*3
            for y in range(boxh):
                off = base + y*row_span
                for i in range(off, off+boxw*3, 3):
                    pix[i]   = YBLACK_FULL
                    pix[i+1] = MID
                    pix[i+2] = MID

        elif pi in ("YBR_FULL_422","YBR_PARTIAL_422"):
            row_span = cols*2
            yblack = YBLACK_PARTIAL if pi.endswith("_PARTIAL_422") else YBLACK_FULL
            for y in range(boxh):
                off = base + y*row_span
                for x in range(0, cols, 2):
                    m = off + (x//2)*4   # Y0 Cb Y1 Cr
                    in0 = x < boxw
                    in1 = (x+1) < boxw
                    if in0: pix[m+0] = yblack
                    if in1: pix[m+2] = yblack
                    if in0 or in1:
                        pix[m+1] = MID
                        pix[m+3] = MID

        elif pi == "YBR_PARTIAL_420":
            Ysz = rows*cols; Csz = (rows//2)*(cols//2)
            Y0 = base; Cb0 = base+Ysz; Cr0 = base+Ysz+Csz
            yblack = YBLACK_PARTIAL
            for y in range(boxh):
                off = Y0 + y*cols
                for i in range(off, off+boxw):
                    pix[i] = yblack
            cw = cols//2; ch = rows//2
            cw_mask = max(0, min(cw, (boxw+1)//2))
            ch_mask = max(0, min(ch, (boxh+1)//2))
            for cy in range(ch_mask):
                row_cb = Cb0 + cy*cw
                row_cr = Cr0 + cy*cw
                for cx in range(cw_mask):
                    pix[row_cb+cx] = MID
                    pix[row_cr+cx] = MID

        elif pi in ("MONOCHROME2","MONOCHROME1"):
            bps = 1 if bits==8 else 2
            row_span = cols*bps
            black = 0 if pi=="MONOCHROME2" else maxmono
            for y in range(boxh):
                off = base + y*row_span
                for x in range(boxw):
                    p = off + x*bps
                    if bps==1:
                        pix[p] = black & 0xFF
                    else:
                        put16(pix, p, black)

        elif pi == "PALETTE COLOR":
            for y in range(boxh):
                off = base + y*cols
                for i in range(off, off+boxw):
                    pix[i] = 0

        else:
            print(f"ERROR: PI={pi} not supported in masker", file=sys.stderr); sys.exit(2)

    # Write back WITHOUT altering transfer syntax/metadata
    ds[Tag(0x7fe0,0x0010)] = pydicom.dataelem.DataElement(Tag(0x7fe0,0x0010), "OW", bytes(pix))
    pydicom.dcmwrite(outp, ds, write_like_original=True)
    print(f"Wrote DICOM: {outp}")

    # Emit rgb24 raw for MOV from arr_rgb (masked earlier to match)
    if arr_rgb is not None and raw_out:
        try:
            with open(raw_out, "wb") as fh:
                fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
            print(f"Wrote rgb24 raw: {raw_out}")
        except Exception as e:
            print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)

    # If MONO 16-bit, also emit gray16 raw for single-frame PNG fidelity
    if raw_gray16_out and pi in ("MONOCHROME1","MONOCHROME2") and bits == 16:
        try:
            # After masking, pix contains the native bytes; extract first (or all) frame(s)
            total_pixels = rows * cols
            frame_bytes = total_pixels * 2
            # For single-frame PNG we only need the first frame; but writing all is fine.
            with open(raw_gray16_out, "wb") as fh:
                fh.write(pix[:frame_bytes])
            print(f"Wrote gray16 raw: {raw_gray16_out}")
        except Exception as e:
            print(f"ERROR: failed to write gray16 raw: {e}", file=sys.stderr)

    sys.exit(0)

# --------- COMPRESSED path (e.g., JPEG Baseline): decode → mask → write uncompressed RGB ----------
# If no mask requested, still produce MOV raw and write pass-through DICOM
if boxw == 0 or boxh == 0:
    if arr_rgb is not None and raw_out:
        try:
            with open(raw_out, "wb") as fh:
                fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
            print(f"Wrote rgb24 raw: {raw_out}")
        except Exception as e:
            print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)
    pydicom.dcmwrite(outp, ds, write_like_original=True)
    print(f"Wrote DICOM (pass-through): {outp}")
    sys.exit(0)

# Mask already applied to arr_rgb above; now write as uncompressed RGB
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
ds.PhotometricInterpretation = "RGB"
ds.SamplesPerPixel = 3
ds.PlanarConfiguration = 0
ds.BitsAllocated = 8
ds.BitsStored = 8
ds.HighBit = 7
if frames > 1:
    ds.NumberOfFrames = str(frames)

ds[Tag(0x7fe0,0x0010)] = pydicom.dataelem.DataElement(
    Tag(0x7fe0,0x0010), "OW", arr_rgb.astype(np.uint8).tobytes(order="C")
)
pydicom.dcmwrite(outp, ds, write_like_original=False)
print(f"Wrote DICOM (decoded→masked→uncompressed): {outp}")

# Emit rgb24 raw for MOV
if arr_rgb is not None and raw_out:
    try:
        with open(raw_out, "wb") as fh:
            fh.write(arr_rgb.reshape(-1,3).tobytes(order="C"))
        print(f"Wrote rgb24 raw: {raw_out}")
    except Exception as e:
        print(f"ERROR: failed to write rgb24 raw: {e}", file=sys.stderr)
PY

  # Build outputs depending on frame count
  if [[ "${frames:-1}" -le 1 ]]; then
    # Single-frame: either blur the entire image (PHI risk) or write lossless PNG.
    if [[ "$phi_blur" -eq 1 ]]; then
      # Prefer rgb24 raw if present; if not, fall back to gray16->rgb24 in ffmpeg.
      if [[ -s "$raw_rgb" ]]; then
        echo "  ffmpeg: BLUR PNG from rgb24 raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" \
          -i "$raw_rgb" \
          -vf "gblur=sigma=10" -frames:v 1 -pix_fmt rgb24 "$out_img"
      elif [[ -s "$raw_gray16" ]]; then
        echo "  ffmpeg: BLUR PNG from gray16le raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format gray16le -video_size "${cols}x${rows}" \
          -i "$raw_gray16" \
          -vf "format=rgb24,gblur=sigma=10" -frames:v 1 -pix_fmt rgb24 "$out_img"
      else
        echo "  ⚠️  No raw produced; skipping image"
      fi
    else
      # Not PHI-risk: write lossless (preserve 16-bit mono when available)
      if [[ -s "$raw_gray16" ]]; then
        echo "  ffmpeg: making 16-bit PNG from gray16le raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format gray16le -video_size "${cols}x${rows}" \
          -i "$raw_gray16" \
          -frames:v 1 -pix_fmt gray16le "$out_img"
      elif [[ -s "$raw_rgb" ]]; then
        echo "  ffmpeg: making PNG from rgb24 raw (${cols}x${rows})"
        ffmpeg -hide_banner -loglevel error -nostdin -y \
          -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" \
          -i "$raw_rgb" \
          -frames:v 1 -pix_fmt rgb24 "$out_img"
      else
        echo "  ⚠️  No raw produced; skipping image"
      fi
    fi
   
    # For single-frame, we skip MOV and QC (png is the deliverable)
    echo "  ✅ Wrote:"
    echo "     - $out_dcm"
    [[ -s "$out_img" ]] && echo "     - $out_img"
  else
    # Multi-frame → MOV + QC as before
    if [[ -s "$raw_rgb" ]]; then
      echo "  ffmpeg: making MOV from rgb24 raw (${cols}x${rows} @ ${fps}fps)"
      ffmpeg -hide_banner -loglevel error -nostdin -y \
        -f rawvideo -pixel_format rgb24 -video_size "${cols}x${rows}" -r "$fps" \
        -i "$raw_rgb" \
        -c:v rawvideo -pix_fmt rgb24 "$out_mov"
    else
      echo "  ⚠️  No rgb24 raw produced; skipping MOV"
    fi

    # Middle frame for QC
    local mid=0
    if [[ -n "$frames" && "$frames" =~ ^[0-9]+$ && "$frames" -gt 0 ]]; then
      mid=$(( (frames - 1) / 2 ))
    fi
    ffmpeg -hide_banner -loglevel error -nostdin -y \
      -i "$out_mov" \
      -vf "select=eq(n\,${mid})" -frames:v 1 "$out_png"
  
    echo "  ✅ Wrote:"
    echo "     - $out_dcm"
    [[ -s "$out_mov" ]] && echo "     - $out_mov"
    [[ -s "$out_png" ]] && echo "     - $out_png"
  fi

}

# ---------- Worker re-entry ----------
if [[ "${1:-}" == "--process-one" ]]; then
  shift
  process_one "$@"
  exit 0
fi

# ---------- CLI ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir)  INPUT_DIR="$2"; shift 2;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2;;
    -n|--num-threads) NUM_THREADS="$2"; shift 2;;
    --overwrite-existing|-O) OVERWRITE=1; shift 1;;
    -h|--help) usage;;
    *) echo "Unknown argument: $1" >&2; usage;;
  esac
done

[[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]] && { echo "❌ --input-dir and --output-dir are required."; usage; }
[[ ! -d "$INPUT_DIR" ]] && { echo "❌ Input directory not found: $INPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

echo "🔎 Input : $INPUT_DIR"
echo "💾 Output: $OUTPUT_DIR"
echo "🚀 Threads: $NUM_THREADS"

SCRIPT_PATH="$(readlink -f "$0")"

# ---------- Parallel dispatcher (streaming; no SIGPIPE aborts) ----------
export PYTHONUNBUFFERED=1
export BLUR_TOP_BLACK_ROWS
export -f process_one detect_black_rows_rgb24_from_dicom detect_top_black_rows_rgb24_from_dicom

# Save & disable pipefail only for this pipeline
_prev_pipefail="$(set -o | awk '/pipefail/{print $3}')"
set +o pipefail
# If available, make the last pipeline command run in current shell:
shopt -s lastpipe 2>/dev/null || true

# FOR TESTING
#| grep -zE 'dvzhsi|dwjadv|ebxizz' \
find "$INPUT_DIR" -type f -name '*.dcm' -print0 \
| xargs -0 -r -n 1 -P "$NUM_THREADS" -I{} bash -c '
  file="$1"; inroot="$2"; outroot="$3"
  echo ">>> START  $file"
  if process_one "$file" "$inroot" "$outroot"; then
    echo "<<< END OK  $file"
    exit 0
  else
    rc=$?
    echo "<<< END FAIL[$rc]  $file"
    exit "$rc"
  fi
' _ {} "$INPUT_DIR" "$OUTPUT_DIR"

# Restore previous pipefail state
[[ "$_prev_pipefail" = "on" ]] && set -o pipefail


