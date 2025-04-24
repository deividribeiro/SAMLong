import argparse
import os
import re
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from typing import List, Tuple, Optional
from moviepy.editor import ImageSequenceClip

from sam2.build_sam import build_sam2_video_predictor


class SAM2LongProcessor:
    SAM_DIR = "/users/5/ribei056/software/python/sam2"
    SAM2LONG_DIR = "//users/5/ribei056/software/python/SAM2Long/sam2"
    def __init__(self):
        self.frame_rate_render = 6
        self.visualization_step = 15
        self.unique_id = None
        self.outdir = None
        self.start_frame = 0
        self.inference_state = None
        self.predictor = None
        self.first_frame_path = None
        self.tracking_points = []
        self.trackings_input_label = []
        self.video_frames_dir = None
        self.scanned_frames = []
        self.frame_names = []
        self.available_frames = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, checkpoint="tiny"):
        """Load the SAM2 model based on the specified checkpoint size"""
        if checkpoint == "tiny":
            sam2_checkpoint = f"{self.SAM_DIR}/checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = f"{self.SAM2LONG_DIR}/configs/sam2.1/sam2.1_hiera_t.yaml"
        elif checkpoint == "small":
            sam2_checkpoint = f"{self.SAM_DIR}/checkpoints/sam2.1_hiera_small.pt"
            model_cfg = f"{self.SAM2LONG_DIR}/configs/sam2.1/sam2.1_hiera_s.yaml"
        elif checkpoint == "base-plus":
            sam2_checkpoint = f"{self.SAM_DIR}/checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = f"{self.SAM2LONG_DIR}/configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            raise ValueError(f"Invalid checkpoint: {checkpoint}")

        print(f"Loading checkpoint: {checkpoint}, ({sam2_checkpoint}, {model_cfg})")
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        print("Model loaded successfully")
        return self.predictor

    def preprocess_video(self, video_path, outdir, max_duration=60):
        """
        Extract frames from a video file.
        Args:
            video_path: Path to the video file
            outdir: Output directory path
            max_duration: Maximum duration to process in seconds
        Returns:
            Path to the first frame
        """
        try:
            # Generate a unique ID based on the video filename
            self.unique_id = os.path.splitext(os.path.basename(video_path))[0]

            # Create output directory
            self.outdir = outdir
            extracted_frames_output_dir = f'{self.outdir}/frames_{self.unique_id}'

            # Create output directories if they don't exist
            os.makedirs(extracted_frames_output_dir, exist_ok=True)
            os.makedirs(f"{self.outdir}/frames_output_images", exist_ok=True)

            # Check if frames already exist
            existing_frames = [
                p for p in os.listdir(extracted_frames_output_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ]

            # If frames already exist, skip extraction
            if existing_frames:
                print(f"Found {len(existing_frames)} existing frames, skipping extraction")
                self.scanned_frames = existing_frames
                self.scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
                self.video_frames_dir = extracted_frames_output_dir
                self.first_frame_path = os.path.join(extracted_frames_output_dir, self.scanned_frames[0])
                print(f"Using existing frames from: {extracted_frames_output_dir}")
                print(f"First frame at: {self.first_frame_path}")
                return self.first_frame_path

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error: Could not open video file {video_path}")

            try:
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    print(f"Warning: Invalid FPS ({fps}), using default value of 30")
                    fps = 30

                max_frames = int(fps * max_duration)

                # Extract frames
                frame_number = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame_number >= max_frames:
                        break

                    if frame_number % self.frame_rate_render == 0:  # Save every nth frame
                        frame_filename = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')
                        # Check if frame file already exists
                        if not os.path.exists(frame_filename):
                            cv2.imwrite(frame_filename, frame)

                    # Store first frame path
                    if frame_number == 0:
                        self.first_frame_path = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')

                    frame_number += 1

            except Exception as e:
                print(f"Error during frame extraction: {str(e)}")
                raise
            finally:
                # Release video resources
                cap.release()

            # Scan all JPEG frames
            self.scanned_frames = [
                p for p in os.listdir(extracted_frames_output_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ]

            if not self.scanned_frames:
                raise ValueError(f"No frames were extracted from the video {video_path}")

            self.scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
            self.video_frames_dir = extracted_frames_output_dir

            print(f"Processed {len(self.scanned_frames)} frames from video")
            print(f"First frame saved at: {self.first_frame_path}")

            # Return the path to access the first frame
            return self.first_frame_path

        except FileNotFoundError:
            print(f"Error: Video file not found: {video_path}")
            raise
        except PermissionError:
            print(f"Error: Permission denied when accessing {video_path} or {outdir}")
            raise
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            raise

    def add_point(self, x, y, start_frame_idx, point_type="include"):
        """
        Add a tracking point
        """
        self.tracking_points.append([x, y])
        if start_frame_idx < 0:
            raise ValueError("Start frame index must be >= 0")
        self.start_frame = start_frame_idx

        # Add label (1 for include, 0 for exclude)
        if point_type == "include":
            self.trackings_input_label.append(1)
        elif point_type == "exclude":
            self.trackings_input_label.append(0)
        else:
            raise ValueError("Point type must be 'include' or 'exclude'")

        print(f"Added {point_type} point at ({x}, {y}) starting at frame {self.start_frame}")
        return self.tracking_points, self.trackings_input_label

    def clear_points(self):
        """Clear all tracking points"""
        self.tracking_points = []
        self.trackings_input_label = []
        print("All points cleared")
        return self.tracking_points, self.trackings_input_label

    def get_mask(self, checkpoint="tiny"):
        """
        Process the first frame with SAM2 to get the initial mask

        Args:
            checkpoint: Model size ("tiny", "small", "base-plus")

        Returns:
            Path to the output image with mask
        """
        if not self.predictor:
            self.load_model(checkpoint)

        if not self.tracking_points:
            raise ValueError("No tracking points added. Use add_point() first.")

        # Init inference state if not already done
        if self.inference_state is None:
            self.inference_state = self.predictor.init_state(video_path=self.video_frames_dir)
            self.inference_state['num_pathway'] = 3
            self.inference_state['iou_thre'] = 0.3
            self.inference_state['uncertainty'] = 2
            self.inference_state['device'] = self.device

        # Process the points
        ann_frame_idx = self.start_frame # First frame
        ann_obj_id = 1  # Object ID

        points = np.array(self.tracking_points, dtype=np.float32)
        labels = np.array(self.trackings_input_label, np.int32)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.title(f"Initial Mask (frame {ann_frame_idx})")
        plt.imshow(Image.open(os.path.join(self.video_frames_dir, self.scanned_frames[ann_frame_idx])))
        self._show_points(points, labels, plt.gca())
        self._show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        # Save output
        output_filename = f"{self.outdir}/output_first_frame.jpg"
        plt.savefig(output_filename, format='jpg')
        plt.close()

        self.frame_names = self.scanned_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Initial mask generated and saved to {output_filename}")
        return output_filename

    def propagate(self, vis_frame_type="check", checkpoint="tiny"):
        """
        Propagate mask to all frames

        Args:
            vis_frame_type: "check" (every 15 frames) or "render" (all frames)
            checkpoint: Model size

        Returns:
            If vis_frame_type is "check": List of output image paths
            If vis_frame_type is "render": Path to the output video
        """
        if not self.inference_state:
            raise ValueError("No inference state. Call get_mask() first.")

        # Use bfloat16 for efficiency if possible
        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Make sure the right model is loaded
        if self.predictor is None:
            self.load_model(checkpoint)

        # Ensure device is set correctly
        if torch.cuda.is_available():
            self.inference_state["device"] = 'cuda'
        else:
            self.inference_state["device"] = 'cpu'

        # Clear output directory
        frames_output_dir = f"{self.outdir}/frames_output_images"
        for f in os.listdir(frames_output_dir):
            os.remove(os.path.join(frames_output_dir, f))

        # Run propagation
        print(f"Starting mask propagation across {self.inference_state['num_frames']} frames...")
        start_time = datetime.now()
        out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=0,
            reverse=False
        )
        propagation_time = datetime.now() - start_time
        print(f"Mask propagation completed in {propagation_time.total_seconds():.2f} seconds")

        # Store results
        print("Processing mask results...")
        video_segments = {}
        for frame_idx in range(0, self.inference_state['num_frames']):
            video_segments[frame_idx] = {
                out_obj_ids[0]: (out_mask_logits[frame_idx] > 0.0).cpu().numpy()
            }

        # Determine visualization stride
        vis_frame_stride = self.visualization_step if vis_frame_type == "check" else 1

        # Calculate total frames to process
        total_frames = len(range(0, len(self.frame_names), vis_frame_stride))
        print(f"Rendering {total_frames} frames with masks...")

        # Render frames with masks in batches
        jpeg_images = []
        batch_size = 100  # Adjust based on your memory constraints
        processed_frames = 0

        for start_idx in range(0, len(self.frame_names), batch_size):
            batch_frames = []
            end_idx = min(start_idx + batch_size, len(self.frame_names))
            batch_start_time = datetime.now()

            print(f"Processing batch {start_idx//batch_size + 1}/{(len(self.frame_names) + batch_size - 1)//batch_size}...")

            for out_frame_idx in range(start_idx, end_idx, vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")

                # Open and process image
                img = Image.open(os.path.join(self.video_frames_dir, self.frame_names[out_frame_idx]))
                plt.imshow(img)

                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    self._show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

                # Save frame
                output_filename = os.path.join(frames_output_dir, f"frame_{out_frame_idx}.jpg")
                plt.savefig(output_filename, format='jpg')
                plt.close()
                img.close()  # Explicitly close PIL image

                batch_frames.append(output_filename)
                processed_frames += 1

                # Print progress every 10 frames or at specific percentages
                if processed_frames % 10 == 0 or processed_frames == total_frames:
                    progress_percent = (processed_frames / total_frames) * 100
                    print(f"Progress: {processed_frames}/{total_frames} frames ({progress_percent:.1f}%)")

            jpeg_images.extend(batch_frames)

            batch_time = datetime.now() - batch_start_time
            frames_per_second = len(batch_frames) / batch_time.total_seconds() if batch_time.total_seconds() > 0 else 0
            print(f"Batch completed in {batch_time.total_seconds():.2f} seconds ({frames_per_second:.2f} frames/second)")

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = datetime.now() - start_time
        print(f"Total processing time: {total_time.total_seconds():.2f} seconds")

        if vis_frame_type == "check":
            print(f"Generated {len(jpeg_images)} sample frames with masks")
            return jpeg_images
        elif vis_frame_type == "render":
            # Create video from frames
            print("Creating output video...")
            video_start_time = datetime.now()

            cap = cv2.VideoCapture(os.path.join(self.video_frames_dir, self.frame_names[0]))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Instead of creating ImageSequenceClip with all images at once
            # Use a more memory-efficient approach
            output_path = f"{self.outdir}/output_video.mp4"

            # Option 1: Create clip with lazy loading
            clip = ImageSequenceClip(jpeg_images, fps=original_fps // self.frame_rate_render,
                                     load_images=False)  # Lazy loading
            clip.write_videofile(output_path, codec='libx264', threads=4,
                                 logger=None)

            # Option 2: Use ffmpeg directly
            # import subprocess
            # cmd = f"ffmpeg -framerate {original_fps // self.frame_rate_render} -i {frames_output_dir}/frame_%d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}"
            # subprocess.call(cmd, shell=True)
            video_time = datetime.now() - video_start_time
            print(f"Video creation completed in {video_time.total_seconds():.2f} seconds")
            print(f"Generated output video: {output_path}")
            return output_path
        else:
            raise ValueError("Invalid vis_frame_type. Use 'check' or 'render'.")

    def _show_mask(self, mask, ax, obj_id=None, random_color=False):
        """Helper function to display mask on pyplot axis"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _show_points(self, coords, labels, ax, marker_size=200):
        """Helper function to display points on pyplot axis"""
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]

        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                       s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                       s=marker_size, edgecolor='white', linewidth=1.25)

    def cleanup_temp_files(self):
        """
        Clean up temporary files created during video generation.
        Removes all temporary frame images from frames_output_images directory.
        """
        frames_output_dir = f"{self.outdir}/frames_output_images"
        if os.path.exists(frames_output_dir):
            file_count = 0
            for f in os.listdir(frames_output_dir):
                file_path = os.path.join(frames_output_dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        file_count += 1
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")

            print(f"Cleaned up {file_count} temporary files from {frames_output_dir}")
        else:
            print(f"Temporary directory {frames_output_dir} not found")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SAM2Long Video Segmenter")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--points", type=str, help="Points as 'x1,y1:1;x2,y2:0' where 1=include, 0=exclude")
    parser.add_argument("--frame", type=int, default=0, help="Frame to start process")
    parser.add_argument("--checkpoint", type=str, default="tiny",
                        choices=["tiny", "small", "base-plus"],
                        help="Model checkpoint to use")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")

    return parser.parse_args()


def main():
    """Main function for command line usage"""
    args = parse_args()

    # Validate video file exists and is a valid video file
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")

    # Check file extension (simple validation)
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    if not any(args.video.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Invalid video file format. Supported formats: {', '.join(valid_extensions)}")

    # Initialize processor
    processor = SAM2LongProcessor()

    # Process video
    processor.preprocess_video(args.video, args.outdir)

    # Parse and add points
    if args.points:
        # Validate points format using regex
        point_pattern = r'^\d+,\d+:[01](?:;\d+,\d+:[01])*$'
        if not re.match(point_pattern, args.points):
            raise ValueError("Invalid points format. Expected format: 'x1,y1:1;x2,y2:0' where x,y are integers and "
                             "the value after colon is either 0 or 1 (1=include, 0=exclude)")

        points = args.points.split(';')
        for point in points:
            try:
                coords, label = point.split(':')
                if label not in ['0', '1']:
                    raise ValueError(f"Invalid label value: {label}. Must be either 0 or 1")

                try:
                    x, y = map(int, coords.split(','))
                except ValueError:
                    raise ValueError(f"Invalid coordinates: {coords}. Must be integers")

                point_type = "include" if label == "1" else "exclude"
                processor.add_point(x, y, args.frame, point_type)
            except Exception as e:
                raise ValueError(f"Error parsing point '{point}': {str(e)}")

    # Generate mask
    processor.get_mask(args.checkpoint)

    # Propagate to all frames
    sample_frames = processor.propagate("check", args.checkpoint)
    video_output = processor.propagate("render", args.checkpoint)


if __name__ == "__main__":
    main()