import cv2
import yaml
import numpy as np


# ==============================
# 1. 读取 YAML 文件中的 3D 坐标
# ==============================
def load_points_from_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    object_pts = []
    for key, value in data.items():
        if isinstance(value, dict) and "x" in value:
            object_pts.append([value["x"], value["y"], value["z"]])

    return np.array(object_pts, dtype=np.float32)


# ==============================
# 2. 从图片中鼠标取像素点
# ==============================
imagePoints = []


def mouse_callback(event, x, y, flags, param):
    global imagePoints
    if event == cv2.EVENT_LBUTTONDOWN:
        imagePoints.append([x, y])
        print(f"Point {len(imagePoints)}: {x}, {y}")

        # 在图像上标记点
        img = param
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(len(imagePoints)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ==============================
# 3. 主流程
# ==============================
def main():
    global imagePoints

    # --- 相机内参（你提供的） ---
    fx = 4479.131275040303
    fy = 4462.034355490826
    cx = 1973.930688628363
    cy = 1641.077482676097

    cameraMatrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    distCoeffs = np.zeros((5, 1))  # 若有畸变可替换

    # --- 读取 3D 赛场坐标 ---
    objectPoints = load_points_from_yaml("3Dcord.yaml")
    print("Loaded 3D points from YAML:\n", objectPoints)
    print(f"需要点击 {len(objectPoints)} 个点")

    # --- 显示图像并取点 ---
    img = cv2.imread("rador.jpg")  # ← 换成你的雷达图像路径
    if img is None:
        print("❌ 无法读取图像，请检查路径")
        return

    display_img = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback, display_img)

    print("\n请按顺序点击图像中的点，与 YAML 中 3D 点顺序对应")
    print("点击完成后按 'q' 键继续，按 'r' 键重新开始")

    while True:
        cv2.imshow("image", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 检查是否收集了足够多的点
            if len(imagePoints) == len(objectPoints):
                break
            else:
                print(f"❌ 还需要点击 {len(objectPoints) - len(imagePoints)} 个点")
        elif key == ord('r'):
            # 重新开始
            imagePoints = []
            display_img = img.copy()
            print("已重置，请重新点击")
        elif key == 27:  # ESC键
            print("用户取消操作")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    imagePoints_np = np.array(imagePoints, dtype=np.float32)

    if len(imagePoints_np) != len(objectPoints):
        print("❌ 点数量不一致！")
        return

    print(f"\n收集到的 2D 点:\n{imagePoints_np}")

    # ==============================
    # 4. solvePnP 求解外参
    # ==============================
    retval, rvec, tvec = cv2.solvePnP(
        objectPoints,
        imagePoints_np,
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not retval:
        print("❌ solvePnP 失败！")
        return

    # --- 旋转向量转旋转矩阵 ---
    R, _ = cv2.Rodrigues(rvec)

    print("\n" + "=" * 50)
    print("solvePnP 结果")
    print("=" * 50)
    print("Rotation Vector (rvec):")
    print(rvec.flatten())
    print("\nRotation Matrix (R):")
    print(R)
    print("\nTranslation Vector (tvec):")
    print(tvec.flatten())

    # --- 外参矩阵 ---
    extrinsic = np.hstack((R, tvec))
    print("\nExtrinsic Matrix [R|t]:")
    print(extrinsic)

    # --- 相机在世界坐标系的位置 ---
    camera_position = -R.T @ tvec
    print("\nCamera Position in World Frame:")
    print(camera_position.flatten())

    # --- 重投影误差检验 ---
    projected_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
    projected_points = projected_points.reshape(-1, 2)

    reprojection_error = np.mean(np.linalg.norm(imagePoints_np - projected_points, axis=1))
    print(f"\n重投影误差: {reprojection_error:.4f} pixels")


if __name__ == "__main__":
    main()