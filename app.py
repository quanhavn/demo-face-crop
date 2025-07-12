from flask import Flask, request, send_file, render_template
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import mediapipe as mp
from rembg import remove
from base64 import b64encode
from PIL import ImageDraw
import math

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def detect_face_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]

def get_head_chin_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    landmark_points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]
    forehead = landmark_points[10]  # gần trán
    chin = landmark_points[152]     # gần cằm
    return forehead, chin

def get_face_center_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    landmark_points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]
    # Sử dụng contour ngoài cùng của khuôn mặt (các điểm 127-356)
    contour_indices = list(range(127, 357))
    contour_points = [landmark_points[i] for i in contour_indices]
    x_center = int(sum([p[0] for p in contour_points]) / len(contour_points))
    y_center = int(sum([p[1] for p in contour_points]) / len(contour_points))
    head = landmark_points[10]
    chin = landmark_points[152]
    return head, chin, contour_points, x_center, y_center

def apply_background(image_pil, color=(255, 255, 255)):
    no_bg = remove(image_pil)
    bg = Image.new("RGBA", no_bg.size, color + (255,))
    composed = Image.alpha_composite(bg, no_bg)
    return composed.convert("RGB")

def crop_by_ratio(image, head, chin, ratio=(10, 39, 11), output_size=(472, 709), x_center=None):
    top, face, bottom = ratio
    total = top + face + bottom
    out_w, out_h = output_size
    face_height = chin[1] - head[1]
    scale = (face / total) * out_h / face_height
    top_px = (top / total) * out_h / scale
    bottom_px = (bottom / total) * out_h / scale
    crop_top = int(head[1] - top_px)
    crop_bottom = int(chin[1] + bottom_px)
    crop_height = crop_bottom - crop_top
    # Sử dụng x_center nếu có, không thì lấy trung điểm head-chin
    if x_center is None:
        x_center = int((head[0] + chin[0]) / 2)
    crop_width = int(out_w / out_h * crop_height)
    crop_left = x_center - crop_width // 2
    crop_right = x_center + crop_width // 2
    # Đảm bảo crop không bị cắt sát biên
    if crop_left < 0:
        crop_right += -crop_left
        crop_left = 0
    if crop_right > image.width:
        diff = crop_right - image.width
        crop_left -= diff
        crop_right = image.width
        if crop_left < 0:
            crop_left = 0
    if crop_right - crop_left < crop_width:
        if crop_left > 0:
            crop_left -= 1
        elif crop_right < image.width:
            crop_right += 1
    crop_img = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_img = crop_img.resize((out_w, out_h), Image.LANCZOS)
    return crop_img

def rotate_point(x, y, cx, cy, angle_rad):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_new = cos_a * (x - cx) - sin_a * (y - cy) + cx
    y_new = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return int(x_new), int(y_new)

def align_face(image, head, chin):
    dx = chin[0] - head[0]
    dy = chin[1] - head[1]
    angle = math.degrees(math.atan2(dx, dy))
    img_center = (image.width // 2, image.height // 2)
    rotated_img = image.rotate(-angle, resample=Image.BICUBIC, center=img_center, expand=True)
    angle_rad = math.radians(angle)
    head_rot = rotate_point(head[0], head[1], *img_center, -angle_rad)
    chin_rot = rotate_point(chin[0], chin[1], *img_center, -angle_rad)
    return rotated_img, head_rot, chin_rot

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-passport', methods=['POST'])
def process_passport():
    file = request.files['image']
    ratio_str = request.form.get('ratio', '10:39:11')
    bg_color_str = request.form.get('background_color', '255,255,255')
    output_size_str = request.form.get('output_size', '')
    output_ratio_str = request.form.get('output_ratio', '2:3')
    output_height_str = request.form.get('output_height', '709')
    show_bbox = request.form.get('show_bbox', None)

    image_pil = Image.open(file).convert("RGB")
    image = np.array(image_pil)

    landmarks = detect_face_landmarks(image)
    if landmarks is None:
        return {"error": "Face not detected"}, 400

    # Lấy head, chin, contour_points, x_center, y_center
    head, chin, contour_points, x_center, y_center = get_face_center_landmarks(landmarks, image.shape)
    bg_color = tuple(map(int, bg_color_str.strip().split(',')))
    ratio = tuple(map(int, ratio_str.strip().split(':')))

    # Xử lý output_size
    if output_size_str:
        output_size = tuple(map(int, output_size_str.strip().split('x')))
    elif output_ratio_str and output_height_str:
        ratio_w, ratio_h = map(int, output_ratio_str.strip().split(':'))
        out_h = int(output_height_str)
        out_w = int(out_h * ratio_w / ratio_h)
        output_size = (out_w, out_h)
    else:
        output_size = (472, 709)  # mặc định 40x60mm

    image_with_bg = apply_background(image_pil, color=bg_color)
    # Xoay ảnh cho thẳng khuôn mặt
    image_with_bg, head, chin = align_face(image_with_bg, head, chin)
    # Sau khi xoay, cũng cần xoay lại toàn bộ contour để lấy trung tâm mới
    img_center = (image_with_bg.width // 2, image_with_bg.height // 2)
    dx = chin[0] - head[0]
    dy = chin[1] - head[1]
    angle = math.degrees(math.atan2(dx, dy))
    angle_rad = math.radians(angle)
    def rotate_point(x, y):
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        x_new = cos_a * (x - img_center[0]) - sin_a * (y - img_center[1]) + img_center[0]
        y_new = sin_a * (x - img_center[0]) + cos_a * (y - img_center[1]) + img_center[1]
        return int(x_new), int(y_new)
    contour_points_rot = [rotate_point(*pt) for pt in contour_points]
    x_center = int(sum([p[0] for p in contour_points_rot]) / len(contour_points_rot))
    # Truyền x_center vào crop_by_ratio
    final_crop = crop_by_ratio(image_with_bg, head=head, chin=chin, ratio=ratio, output_size=output_size, x_center=x_center)

    buf = BytesIO()
    final_crop.save(buf, format='JPEG')
    buf.seek(0)

    if show_bbox is not None:
        bbox_img = image_with_bg.copy()
        draw = ImageDraw.Draw(bbox_img)
        # Vẽ bounding box crop lên ảnh
        # Tính lại crop_top, crop_bottom, crop_width như trong crop_by_ratio
        top, face, bottom = ratio
        total = top + face + bottom
        out_w, out_h = output_size
        face_height = chin[1] - head[1]
        scale = (face / total) * out_h / face_height
        top_px = (top / total) * out_h / scale
        bottom_px = (bottom / total) * out_h / scale
        crop_top = int(head[1] - top_px)
        crop_bottom = int(chin[1] + bottom_px)
        crop_height = crop_bottom - crop_top
        crop_width = int(out_w / out_h * crop_height)
        crop_left = max(0, x_center - crop_width // 2)
        crop_right = min(bbox_img.width, x_center + crop_width // 2)
        if crop_right - crop_left < crop_width:
            if crop_left > 0:
                crop_left -= 1
            elif crop_right < bbox_img.width:
                crop_right += 1
        draw.rectangle([crop_left, crop_top, crop_right, crop_bottom], outline=(0,255,0), width=3)
        bbox_buf = BytesIO()
        bbox_img.save(bbox_buf, format='JPEG')
        bbox_buf.seek(0)
        bbox_b64 = 'data:image/jpeg;base64,' + b64encode(bbox_buf.read()).decode('utf-8')
        result_b64 = 'data:image/jpeg;base64,' + b64encode(buf.read()).decode('utf-8')
        return {"bbox_image": bbox_b64, "result_image": result_b64}
    else:
        return send_file(buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)

