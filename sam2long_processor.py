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

# Import SAM2 predictor - you'll need to install the SAM2Long package
from sam2.build_sam import build_sam2_video_predictor


class SAM2LongProcessor:
    def __init__(self):
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

        # Create output directories if they don't exist
        os.makedirs("frames_output_images", exist_ok=True)

    def load_model(self, checkpoint="tiny"):
        """Load the SAM2 model based on the specified checkpoint size"""
        if checkpoint == "tiny":
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif checkpoint == "small":
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif checkpoint == "base-plus":
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            raise ValueError(f"Invalid checkpoint: {checkpoint}")

        print(f"Loading checkpoint: {checkpoint}")
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        print("Model loaded successfully")
        return self.predictor

    def preprocess_video(self, video_path, max_duration=60):
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file
            max_duration: Maximum duration to process in seconds

        Returns:
            Path to the first frame
        """
        # Generate a unique ID based on current date and time
        unique_id = datetime.now().strftime('%Y%m%d%H%M%S')

        # Create output directory
        extracted_frames_output_dir = f'frames_{unique_id}'
        os.makedirs(extracted_frames_output_dir, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        max_frames = int(fps * max_duration)

        # Extract frames
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_number >= max_frames:
                break

            if frame_number % 6 == 0:  # Save every 6th frame
                frame_filename = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')
                cv2.imwrite(frame_filename, frame)

            # Store first frame path
            if frame_number == 0:
                self.first_frame_path = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')

            frame_number += 1

        # Release video
        cap.release()

        # Scan all JPEG frames
        self.scanned_frames = [
            p for p in os.listdir(extracted_frames_output_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        self.scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))

        self.video_frames_dir = extracted_frames_output_dir

        print(f"Processed {len(self.scanned_frames)} frames from video")
        print(f"First frame saved at: {self.first_frame_path}")

        # Return the path to access the first frame
        return self.first_frame_path

    def add_point(self, x, y, point_type="include"):
        """
        Add a tracking point

        Args:
            x, y: Coordinates of the point
            point_type: "include" or "exclude"
        """
        self.tracking_points.append([x, y])

        # Add label (1 for include, 0 for exclude)
        if point_type == "include":
            self.trackings_input_label.append(1)
        elif point_type == "exclude":
            self.trackings_input_label.append(0)
        else:
            raise ValueError("Point type must be 'include' or 'exclude'")

        print(f"Added {point_type} point at ({x}, {y})")
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
        ann_frame_idx = 0  # First frame
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
        output_filename = "output_first_frame.jpg"
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
        frames_output_dir = "frames_output_images"
        for f in os.listdir(frames_output_dir):
            os.remove(os.path.join(frames_output_dir, f))

        # Run propagation
        out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=0,
            reverse=False
        )

        # Store results
        video_segments = {}
        for frame_idx in range(0, self.inference_state['num_frames']):
            video_segments[frame_idx] = {
                out_obj_ids[0]: (out_mask_logits[frame_idx] > 0.0).cpu().numpy()
            }

        # Determine visualization stride
        vis_frame_stride = 15 if vis_frame_type == "check" else 1

        # Render frames with masks
        jpeg_images = []
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(self.video_frames_dir, self.frame_names[out_frame_idx])))

            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                self._show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

            # Save frame
            output_filename = os.path.join(frames_output_dir, f"frame_{out_frame_idx}.jpg")
            plt.savefig(output_filename, format='jpg')
            plt.close()

            jpeg_images.append(output_filename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if vis_frame_type == "check":
            print(f"Generated {len(jpeg_images)} sample frames with masks")
            return jpeg_images
        elif vis_frame_type == "render":
            # Create video from frames
            cap = cv2.VideoCapture(os.path.join(self.video_frames_dir, self.frame_names[0]))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Create clip
            clip = ImageSequenceClip(jpeg_images, fps=original_fps // 6)

            # Write output
            output_path = "output_video.mp4"
            clip.write_videofile(output_path, codec='libx264')

            print(f"Generated output video: {output_path}")
            return output_path

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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SAM2Long Video Segmenter")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--points", type=str, help="Points as 'x1,y1:1;x2,y2:0' where 1=include, 0=exclude")
    parser.add_argument("--checkpoint", type=str, default="tiny",
                        choices=["tiny", "small", "base-plus"],
                        help="Model checkpoint to use")
    parser.add_argument("--output", type=str, default="check",
                        choices=["check", "render"],
                        help="Output type (check=sample frames, render=video)")

    return parser.parse_args()


def main():
    """Main function for command line usage"""
    args = parse_args()

    # Initialize processor
    processor = SAM2LongProcessor()

    # Process video
    processor.preprocess_video(args.video)

    # Parse and add points
    if args.points:
        points = args.points.split(';')
        for point in points:
            coords, label = point.split(':')
            x, y = map(int, coords.split(','))
            point_type = "include" if label == "1" else "exclude"
            processor.add_point(x, y, point_type)

    # Generate mask
    processor.get_mask(args.checkpoint)

    # Propagate to all frames
    result = processor.propagate(args.output, args.checkpoint)

    # Print result
    if args.output == "check":
        print(f"Generated {len(result)} frames with masks")
    else:
        print(f"Generated video at {result}")


if __name__ == "__main__":
    main()