import re
import argparse
import math
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import shutil
import numpy as np
import imageio.v3 as iio
import cv2
from typing import Optional
from pathlib import Path

try:
    from simulator_interface.airsim_tools import fetch_image_from_simulator
except ModuleNotFoundError:
    fetch_image_from_simulator = None

class ImageEnhancementUtils:
    def enhance_image(self, image_path: str, enhancement_type: Optional[dict[str, str]]) -> Optional[str]:
        """Apply enhancement to the image and return path to enhanced image."""
        if enhancement_type is None:
            return image_path

        if enhancement_type.get("method") == "resize":
            target_size = list(map(int, enhancement_type.get("target_size").split(",")[0:2]))
            assert isinstance(target_size, list) and len(target_size) == 2, "Invalid target size for resizing"
            output_path = self._get_output_path(image_path, "_resize")
            success = self._resize_image(image_path, output_path, target_size)
            assert success, "Image resizing failed"
            return output_path

        if enhancement_type.get("method") == "road_guide_overlay":
            return self.draw_road_guide_overlay(image_path)

        if enhancement_type.get("method") == "resize_then_guide_overlay":
            target_size = list(map(int, enhancement_type.get("target_size").split(",")[0:2]))
            assert isinstance(target_size, list) and len(target_size) == 2, "Invalid target size for resizing"
            resize_path = self._get_output_path(image_path, "_resize")
            success = self._resize_image(image_path, resize_path, target_size)
            assert success, "Image resizing failed"
            return self.draw_road_guide_overlay(resize_path)

        if enhancement_type.get("method") == "upward_arrow":
            return self.draw_upward_arrow(image_path)

        if enhancement_type.get("method") == "resize_then_upward_arrow":
            target_size = list(map(int, enhancement_type.get("target_size").split(",")[0:2]))
            assert isinstance(target_size, list) and len(target_size) == 2, "Invalid target size for resizing"
            resize_path = self._get_output_path(image_path, "_resize")
            success = self._resize_image(image_path, resize_path, target_size)
            assert success, "Image resizing failed"
            return self.draw_upward_arrow(resize_path)

        if enhancement_type.get("method") == "upward_arrow_with_text":
            return self.draw_upward_arrow_with_text(image_path)

        if enhancement_type.get("method") == "resize_then_upward_arrow_with_text":
            target_size = list(map(int, enhancement_type.get("target_size").split(",")[0:2]))
            assert isinstance(target_size, list) and len(target_size) == 2, "Invalid target size for resizing"
            resize_path = self._get_output_path(image_path, "_resize")
            success = self._resize_image(image_path, resize_path, target_size)
            assert success, "Image resizing failed"
            return self.draw_upward_arrow_with_text(resize_path)

        if enhancement_type.get("method") == "three_arrows_with_text":
            return self.draw_three_arrows_with_text(image_path)

        if enhancement_type.get("method") == "resize_then_three_arrows_with_text":
            target_size = list(map(int, enhancement_type.get("target_size").split(",")[0:2]))
            assert isinstance(target_size, list) and len(target_size) == 2, "Invalid target size for resizing"
            resize_path = self._get_output_path(image_path, "_resize")
            success = self._resize_image(image_path, resize_path, target_size)
            assert success, "Image resizing failed"
            return self.draw_three_arrows_with_text(resize_path)

        if enhancement_type.get("method") == "some_other_method":
            output_path = self._get_output_path(image_path, "_enhanced")
            try:
                shutil.copy2(image_path, output_path)
                return output_path
            except Exception as e:
                print(f"⚠ Failed to copy image {image_path} to {output_path}: {e}")
                return None

        print(f"⚠ Unknown enhancement type: {enhancement_type}")
        return None

    def _resize_image(self, image_path: str, output_path: str, target_size=(256, 256)) -> bool:
        """Resize image to target size and save to output path."""
        try:
            img = Image.open(image_path)
            img_resized = img.resize(target_size)
            img_resized.save(output_path)
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ Failed to resize image {image_path}: {e}")
            return False

    @staticmethod
    def _get_output_path(image_path: str, suffix: str) -> str:
        img_path = Path(image_path)
        return str(img_path.with_name(f"{img_path.stem}{suffix}{img_path.suffix}"))

    def draw_road_guide_overlay(self, image_path: str) -> Optional[str]:
        """
        Analyzes the image using OpenCV to find the Main Road and draws a
        Guidance Arrow pointing along the road's curvature.
        Returns the duplicated image path on success.
        """
        # Load image
        try:
            image_pil = Image.open(image_path)
        except Exception as e:
            print(f"⚠ Failed to open image {image_path}: {e}")
            return None

        # Duplicate image path to avoid overwriting original
        try:
            img_path = Path(image_path)
            output_path = img_path.with_name(f"{img_path.stem}_overlay{img_path.suffix}")
        except Exception as e:
            print(f"⚠ Failed to create output path for {image_path}: {e}")
            return None

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # 1. Preprocessing (Blur to remove noise/texture)
        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 2. Masking: Isolate the Road
        lower_grey = np.array([0, 0, 50])      # Dark grey
        upper_grey = np.array([180, 50, 220])  # Light grey

        mask = cv2.inRange(hsv, lower_grey, upper_grey)

        # Morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 3. Find the "Main Road" Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            try:
                image_pil.save(output_path)
                return str(output_path)
            except Exception as e:
                print(f"⚠ Failed to save image {output_path}: {e}")
                return None

        largest_contour = max(contours, key=cv2.contourArea)

        # 4. Calculate Direction
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = w // 2, h // 2

        # Define Points
        start_point = (w // 2, h - 50)  # Bottom Center
        target_y = min(cY, h // 2)
        end_point = (cX, target_y)

        # 5. Determine Text Label based on Angle
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = math.degrees(math.atan2(dy, dx))  # -90 is straight up
        deviation = angle + 90

        action_text = "GUIDE: FWD"
        if deviation > 20:
            action_text = "GUIDE: TURN RIGHT"
        elif deviation < -20:
            action_text = "GUIDE: TURN LEFT"

        # 6. Draw Overlay
        arrow_color = (255, 0, 0)     # Blue for Guidance
        text_color = (255, 255, 255)
        text_outline = (0, 0, 0)

        cv2.arrowedLine(img_cv, start_point, end_point, arrow_color, 10, tipLength=0.3)

        cv2.putText(img_cv, action_text, (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_outline, 6)
        cv2.putText(img_cv, action_text, (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        try:
            output_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            output_pil.save(output_path)
            return str(output_path)
        except Exception as e:
            print(f"⚠ Failed to save image {output_path}: {e}")
            return None

    def draw_upward_arrow(self, image_path: str) -> Optional[str]:
        """Draw a simple upward arrow at the top center of the image."""
        try:
            image_pil = Image.open(image_path)
        except Exception as e:
            print(f"⚠ Failed to open image {image_path}: {e}")
            return None

        try:
            output_path = self._get_output_path(image_path, "_up_arrow")
        except Exception as e:
            print(f"⚠ Failed to create output path for {image_path}: {e}")
            return None

        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        start_point = (w // 2, 40)
        end_point = (w // 2, 5)
        arrow_color = (255, 0, 0)

        cv2.arrowedLine(img_cv, start_point, end_point, arrow_color, 6, tipLength=0.4)

        try:
            output_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            output_pil.save(output_path)
            return str(output_path)
        except Exception as e:
            print(f"⚠ Failed to save image {output_path}: {e}")
            return None

    def draw_upward_arrow_with_text(self, image_path: str) -> Optional[str]:
        """Draw a simple upward arrow and the text 'Forward' at the top center."""
        try:
            image_pil = Image.open(image_path)
        except Exception as e:
            print(f"⚠ Failed to open image {image_path}: {e}")
            return None

        try:
            output_path = self._get_output_path(image_path, "_up_arrow_text")
        except Exception as e:
            print(f"⚠ Failed to create output path for {image_path}: {e}")
            return None

        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        start_point = (w // 2, 50)
        end_point = (w // 2, 10)
        arrow_color = (255, 0, 0)
        text_color = (255, 255, 255)
        text_outline = (0, 0, 0)

        cv2.arrowedLine(img_cv, start_point, end_point, arrow_color, 6, tipLength=0.4)

        text = "Forward"
        text_pos = (w // 2 - 60, 85)
        cv2.putText(img_cv, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_outline, 4)
        cv2.putText(img_cv, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        try:
            output_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            output_pil.save(output_path)
            return str(output_path)
        except Exception as e:
            print(f"⚠ Failed to save image {output_path}: {e}")
            return None

    def draw_three_arrows_with_text(self, image_path: str) -> Optional[str]:
        """Draw forward, left, and right arrows with text labels near the top of the image."""
        try:
            image_pil = Image.open(image_path)
        except Exception as e:
            print(f"⚠ Failed to open image {image_path}: {e}")
            return None

        try:
            output_path = self._get_output_path(image_path, "_three_arrows_text")
        except Exception as e:
            print(f"⚠ Failed to create output path for {image_path}: {e}")
            return None

        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        arrow_color = (255, 0, 0)
        text_color = (255, 255, 255)
        text_outline = (0, 0, 0)

        # Forward arrow (center, up)
        fwd_start = (w // 2, 50)
        fwd_end = (w // 2, 10)
        cv2.arrowedLine(img_cv, fwd_start, fwd_end, arrow_color, 6, tipLength=0.4)
        fwd_text_pos = (w // 2 - 60, 85)
        cv2.putText(img_cv, "Forward", fwd_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_outline, 4)
        cv2.putText(img_cv, "Forward", fwd_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        # Left arrow (top-left, pointing left)
        left_start = (w // 2 - 120, 50)
        left_end = (w // 2 - 190, 50)
        cv2.arrowedLine(img_cv, left_start, left_end, arrow_color, 6, tipLength=0.4)
        left_text_pos = (w // 2 - 220, 85)
        cv2.putText(img_cv, "Left", left_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_outline, 4)
        cv2.putText(img_cv, "Left", left_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        # Right arrow (top-right, pointing right)
        right_start = (w // 2 + 120, 50)
        right_end = (w // 2 + 190, 50)
        cv2.arrowedLine(img_cv, right_start, right_end, arrow_color, 6, tipLength=0.4)
        right_text_pos = (w // 2 + 130, 85)
        cv2.putText(img_cv, "Right", right_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_outline, 4)
        cv2.putText(img_cv, "Right", right_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        try:
            output_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            output_pil.save(output_path)
            return str(output_path)
        except Exception as e:
            print(f"⚠ Failed to save image {output_path}: {e}")
            return None

        


def fetch_image(state, img_enhancement_type: Optional[dict[str, str]] = None) -> str:
    """Fetch current image from AirSim based on drone pose in state."""
    if fetch_image_from_simulator is None:
        raise ModuleNotFoundError(
            "simulator_interface is not available. Run from project root with proper PYTHONPATH, "
            "or use this module's local image enhancement utilities directly."
        )

    drone_pose = state["drone_pose"]
    current_x = drone_pose.get("x", 0.0)
    current_y = drone_pose.get("y", 0.0)
    current_z = drone_pose.get("z", 0.0)
    current_yaw = drone_pose.get("yaw", 0.0)
    payload = {
        "x": current_x,
        "y": current_y,
        "z": current_z,
        "yaw": current_yaw,
    }

    result = fetch_image_from_simulator(**payload)
    stepped_img_path = result.get("image_path")
    collision_info = result.get("collision_info")
    if not collision_info and result.get("collided"):
        collision_info = ["Unknown"]
    if img_enhancement_type is None:
        return stepped_img_path, collision_info
    else:
        enhancer = ImageEnhancementUtils()
        enhanced_path = enhancer.enhance_image(image_path=stepped_img_path, enhancement_type=img_enhancement_type)
        return enhanced_path, collision_info
    
def stitch_images_to_video(
    image_folder,
    image_files,
    output_path=None,
    fps=10,
    codec='libx264',
    quality=20,
    output_format="webp",
    annotate_step_action=True,
):
    """Stitch image sequence to WebP (default) or MP4.

    Args:
        image_folder: Folder containing image files.
        image_files: Ordered list of image filenames.
        output_path: Destination path. If None, returns bytes.
        fps: Playback FPS.
        codec: Video codec used for MP4 output.
        quality: Quality setting (WebP quality or imageio quality).
        output_format: "webp" (default) or "mp4".

    Returns:
        If output_path is provided: bool success.
        If output_path is None: bytes content (or None on failure).
    """
    if not image_files:
        return False if output_path else None

    fmt = (output_format or "webp").strip().lower()
    if fmt not in {"webp", "mp4"}:
        raise ValueError(
            f"Unsupported output_format '{output_format}'. Use 'webp' or 'mp4'."
        )

    ordered_files = _order_frame_filenames(image_files)
    ordered_files = _dedupe_consecutive_identical_files(image_folder, ordered_files)

    step_action_map = (
        _load_step_action_map(image_folder)
        if annotate_step_action
        else {}
    )

    first_img = Image.open(os.path.join(image_folder, ordered_files[0])).convert("RGB")
    size = first_img.size
    frames_pil = [_annotate_frame(first_img, ordered_files[0], step_action_map, frame_index=1)]
    for idx, fname in enumerate(ordered_files[1:], start=2):
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB").resize(size)
        img = _annotate_frame(img, fname, step_action_map, frame_index=idx)
        frames_pil.append(img)

    if fmt == "webp":
        duration_ms = int(1000 / fps) if fps > 0 else 100

        if output_path:
            frames_pil[0].save(
                output_path,
                format="WEBP",
                save_all=True,
                append_images=frames_pil[1:],
                duration=duration_ms,
                loop=0,
                quality=quality,
                method=6,
                lossless=True,
            )
            return True

        with tempfile.NamedTemporaryFile(delete=False, suffix='.webp') as tmpfile:
            webp_path = tmpfile.name
        frames_pil[0].save(
            webp_path,
            format="WEBP",
            save_all=True,
            append_images=frames_pil[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=6,
            lossless=True,
        )
        with open(webp_path, 'rb') as f:
            media_bytes = f.read()
        os.remove(webp_path)
        return media_bytes

    # MP4 path
    frames_np = [np.array(img) for img in frames_pil]
    mp4_quality = max(1, min(10, int(quality)))
    if output_path:
        iio.imwrite(output_path, frames_np, fps=fps, codec=codec, quality=mp4_quality)
        return True

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        video_path = tmpfile.name
    iio.imwrite(video_path, frames_np, fps=fps, codec=codec, quality=mp4_quality)
    with open(video_path, 'rb') as f:
        media_bytes = f.read()
    os.remove(video_path)
    return media_bytes


def _extract_step_idx(filename: str):
    match = re.search(r"(?:^|_)step_(\d+)(?:\.|_|$)", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _extract_seg_parent_step_idx(filename: str):
    match = re.search(r"airsim_latest_(\d+)_seg_\d+(?:\.|_|$)", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _extract_seg_idx(filename: str):
    match = re.search(r"(?:^|_)seg_(\d+)(?:\.|_|$)", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _extract_annotation_step_idx(filename: str):
    step_idx = _extract_step_idx(filename)
    if step_idx is not None:
        return step_idx
    seg_parent = _extract_seg_parent_step_idx(filename)
    if seg_parent is not None:
        return seg_parent - 1  # Use source step, not destination; keep t=0 until step 1 appears
    return None


def _frame_sequence_key(filename: str):
    """Sort frames as: step0, segs-to-step1, step1, segs-to-step2, step2, ..."""
    step_idx = _extract_step_idx(filename)
    if step_idx is not None:
        return (step_idx, 1, 0, filename)

    seg_parent = _extract_seg_parent_step_idx(filename)
    if seg_parent is not None:
        seg_idx = _extract_seg_idx(filename) or 0
        return (seg_parent, 0, seg_idx, filename)

    return (10**9, 2, 0, filename)


def _order_frame_filenames(image_files: list[str]) -> list[str]:
    return sorted(image_files, key=_frame_sequence_key)


def _dedupe_consecutive_identical_files(image_folder: str, ordered_files: list[str]) -> list[str]:
    """Remove back-to-back identical frame files by content hash.

    This helps when intermediate capture produces repeated PNGs (same pixels),
    which can make playback appear to jump timesteps unexpectedly.
    """
    if not ordered_files:
        return ordered_files

    deduped = []
    prev_hash = None
    for fname in ordered_files:
        fpath = os.path.join(image_folder, fname)
        try:
            with open(fpath, "rb") as f:
                current_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            current_hash = None

        if current_hash is not None and current_hash == prev_hash:
            continue

        deduped.append(fname)
        if current_hash is not None:
            prev_hash = current_hash

    return deduped


def _load_step_action_map(image_folder: str) -> dict:
    """Load step->action mapping from sibling debug_outputs json if present."""
    try:
        base_dir = Path(image_folder).parent
        debug_json = base_dir / "debug_outputs" / "step_by_step_intermediate_results.json"
        if not debug_json.exists():
            return {}
        with open(debug_json, "r") as f:
            data = json.load(f) or {}
        mapping = {}
        for k, v in data.items():
            try:
                step = int(k)
            except Exception:
                continue
            action = v.get("current_action")
            if isinstance(action, list):
                action = action[0] if action else None
            if not action:
                action = v.get("parsed_action")
            if action is not None:
                mapping[step] = str(action)
        return mapping
    except Exception:
        return {}


def _annotate_frame(
    img: Image.Image,
    filename: str,
    step_action_map: dict,
    frame_index: int | None = None,
) -> Image.Image:
    """Draw timestep and action overlay on a frame."""
    step_idx = _extract_annotation_step_idx(filename)
    if step_idx is None:
        return img

    action_text = step_action_map.get(step_idx, "N/A")
    if frame_index is None:
        overlay_text = f"t={step_idx} | action={action_text}"
    else:
        overlay_text = f"f={frame_index:03d} | t={step_idx} | action={action_text}"

    draw = ImageDraw.Draw(img)
    try:
        # Use a much larger font for readability (~4x bigger than default).
        # Scale with image height and keep a practical minimum.
        font_size = max(40, int(img.height * 0.04))
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = None

    x = 16
    y = 14
    if font is not None:
        bbox = draw.textbbox((x, y), overlay_text, font=font)
    else:
        bbox = draw.textbbox((x, y), overlay_text)

    pad = 12
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0),
    )

    if font is not None:
        draw.text((x, y), overlay_text, fill=(255, 255, 255), font=font)
    else:
        draw.text((x, y), overlay_text, fill=(255, 255, 255))
    return img


def _normalize_name_token(value: str) -> str:
    token = (value or "unknown").strip().lower()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9._]+", "", token)
    token = token.replace("_preview", "")
    if token.startswith("gemini_"):
        token = "gemini" + token[len("gemini_"):]
    return token or "unknown"


def _to_metric_token(value, default: str = "na") -> str:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return f"{num:.2f}"
    except Exception:
        return default


def _default_webp_name_from_image_dir(image_dir: str) -> str:
    img_dir = Path(image_dir)
    exp_dir = img_dir.parent

    mission_name = "unknown_mission"
    model_name = "unknown_model"
    try:
        model_name = _normalize_name_token(exp_dir.parents[1].name)
        mission_name = _normalize_name_token(exp_dir.parents[2].name)
    except Exception:
        pass

    mp = "na"
    sr5 = "na"
    sr10 = "na"
    eval_path = exp_dir / "evaluation_result.json"
    if eval_path.exists():
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f) or {}
            mp = _to_metric_token(eval_data.get("mission_progress"))
            sr5 = _to_metric_token(eval_data.get("sr5"))
            sr10 = _to_metric_token(eval_data.get("sr10"))
        except Exception:
            pass

    return (
        f"{mission_name}_{model_name}_"
        f"mp_{mp}_sr5_{sr5}_sr10_{sr10}.webp"
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create animated WebP from an image directory.",
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        type=str,
        help="Directory containing input frames (.png/.jpg/.jpeg)",
    )
    parser.add_argument(
        "--image_dir",
        dest="image_dir_flag",
        type=str,
        default=None,
        help="Directory containing input frames (.png/.jpg/.jpeg)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WebP path (default: <image_dir>/mission_video.webp)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Playback fps (default: 1)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=20,
        help="WebP quality (0-100, default: 20)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable timestep/action overlay text on frames.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir_flag or args.image_dir
    if not image_dir:
        parser.error("image_dir is required (positional or --image_dir)")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not image_files:
        raise ValueError(f"No image frames found in: {image_dir}")

    output_path = args.output or os.path.join(
        image_dir,
        _default_webp_name_from_image_dir(image_dir),
    )
    ok = stitch_images_to_video(
        image_folder=image_dir,
        image_files=image_files,
        output_path=output_path,
        fps=max(1, args.fps),
        quality=max(0, min(100, args.quality)),
        output_format="webp",
        annotate_step_action=not args.no_annotate,
    )
    if ok:
        print(f"✅ WebP saved: {output_path}")
    else:
        print("❌ Failed to create WebP")
