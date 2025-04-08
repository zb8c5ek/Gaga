__author__ = 'Xuan-Li CHEN'
"""
Xuan-Li Chen
Domain: Computer Vision, Machine Learning
Email: chen_alphonse_xuanli(at)outlook.com
"""
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def display_images(render_path, semantic_path, fp_output=None):
    # 读取渲染图像和语义图像
    render_img = cv2.imread(render_path)
    semantic_img = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)

    # 将语义图像叠加到渲染图像上
    overlay_img = cv2.addWeighted(render_img, 0.7, semantic_img, 0.3, 0)

    # 将BGR图像转换为RGB图像以便matplotlib显示
    render_img_rgb = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
    semantic_img_rgb = cv2.cvtColor(semantic_img, cv2.COLOR_BGR2RGB)
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    # 创建一个图形窗口
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示渲染图像
    axes[0].imshow(render_img_rgb)
    axes[0].set_title('Rendered Image')
    axes[0].axis('off')

    # 显示语义图像
    axes[1].imshow(semantic_img_rgb)
    axes[1].set_title('Semantic Image')
    axes[1].axis('off')

    # 显示叠加图像
    axes[2].imshow(overlay_img_rgb)
    axes[2].set_title('Overlay Image')
    axes[2].axis('off')

    # 显示图形
    plt.tight_layout()
    if fp_output is None:
        plt.show()
    else:
        plt.savefig(fp_output, dpi=300)
    plt.close(fig)

def sample_single():
    render_path = "/e_disk/ParkSide/down480p/train/ours_10000/renders/frameGH010466_0002.png"
    semantic_path = "/e_disk/ParkSide/down480p/train/ours_10000/objects_pred/frameGH010466_0002.png"
    display_images(render_path, semantic_path)

def main_multi(dn_render, dn_semantic, dp_output=None):
    """
    Fuses rendered images with semantic images and saves the output.
    Args:
        dn_render (str): Directory containing rendered images
        dn_semantic (str): Directory containing semantic images
        dp_output (str): Directory to save the output images

    """
    dp_render = Path(dn_render).resolve()
    dp_semantic = Path(dn_semantic).resolve()
    assert dp_render.is_dir(), f"Render directory {dp_render} does not exist"
    assert dp_semantic.is_dir(), f"Semantic directory {dp_semantic} does not exist"

    fps_images_render = list(dp_render.glob('*.png')) + list(dp_render.glob('*.jpg'))
    fps_images_semantic = list(dp_semantic.glob('*.png')) + list(dp_semantic.glob('*.jpg'))
    assert len(fps_images_render) == len(fps_images_semantic), "Number of images in render and semantic directories must match"
    fps_images_render.sort()
    fps_images_semantic.sort()
    if dp_output is None:
        dp_output = dp_render.parent / "fuse_viz_output"
    dp_output.mkdir(parents=True, exist_ok=True)

    for image_pair in tqdm(zip(fps_images_render, fps_images_semantic), total=len(fps_images_render), desc="Fusing Viz images"):
        render_path = image_pair[0]
        semantic_path = image_pair[1]
        output_path = dp_output / f"{render_path.stem}_overlay.png"
        display_images(render_path, semantic_path, fp_output=output_path)


def images_to_video(image_folder, output_path=None, fps=12):
    """
    Convert a folder of images to a video file.

    Args:
        image_folder (str): Path to the folder containing images
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
    """
    # Get list of images
    images = sorted([img for img in Path(image_folder).glob("*.png")])
    if not images:
        print("No images found in the specified folder")
        return

    # Read first image to get dimensions
    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if output_path is None:
        output_path = Path(image_folder).parent / ("video_%s.mp4" % (Path(image_folder).name))
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Add images to video
    for image_path in images:
        frame = cv2.imread(str(image_path))
        video.write(frame)

    # Release resources
    video.release()
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    from fire import Fire
    Fire(
        {"single": sample_single,
         "multi": main_multi,
         "2video": images_to_video,
         }
    )
