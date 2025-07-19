import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataloader import PairedDataset


def load_depth(depth_path):
    # CSV로 저장된 depth map 파일 읽기 (예: np.loadtxt 사용)
    depth = np.loadtxt(depth_path, delimiter=",")
    return depth  # numpy array

def compute_average_depth_map(paired_dataset):
    """
    paired_dataset: PairedDataset 클래스의 인스턴스.
    self.file_list에는 (img_path, depth_path, audio_path, mic_info)가 저장되어 있습니다.
    """
    sum_depth = None
    count = 0

    for item in paired_dataset.file_list:
        # item[1]는 depth_path 입니다.
        depth_path = item[1]
        depth = load_depth(depth_path)  # numpy array로 불러옴
        # 첫 번째 파일의 shape으로 초기화
        if sum_depth is None:
            sum_depth = np.zeros_like(depth, dtype=np.float64)
        # shape이 일치하는지 확인
        if depth.shape != sum_depth.shape:
            raise ValueError(f"Depth shape mismatch in file {depth_path}")
        sum_depth += depth
        count += 1

    if count == 0:
        raise ValueError("No depth maps found in dataset.")
    avg_depth = sum_depth / count
    return avg_depth

# 사용 예시:
if __name__ == "__main__":
    dataset_root = "/home/byulharang/small_DT/Val"
    depth_root = "/home/byulharang/small_DT/depth/Val"
    paired_dataset = PairedDataset(dataset_root, depth_root)

    avg_depth_map = compute_average_depth_map(paired_dataset)
    print("Average depth map shape:", avg_depth_map.shape)

    # 결과를 CSV로 저장
    np.savetxt("average_depth_map.csv", avg_depth_map, delimiter=",", fmt="%.6f")
    print(avg_depth_map.min(), avg_depth_map.max())
    print("CSV 파일 saved: average_depth_map.csv")

    # 결과를 PNG 이미지로 저장 (정규화 과정 포함)
    depth_min = np.min(avg_depth_map)
    depth_max = np.max(avg_depth_map)
    norm_depth = (avg_depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    norm_depth = (norm_depth * 255).astype(np.uint8)
    avg_depth_img = Image.fromarray(norm_depth)
    avg_depth_img.save("average_depth_map.png")
    print("PNG 파일 saved: average_depth_map.png")

