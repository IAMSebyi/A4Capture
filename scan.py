import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageTk
from datetime import datetime
import tkinter as tk
import numpy as np
import os

# Constants
PAGE_WIDTH, PAGE_HEIGHT = A4
GAUSSIAN_BLUR_KERNEL = (5, 5)
DOC_NAME_DEFAULT = "Documents/" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + ".pdf"

# Globals
document = []
scan_requested = 0
scans_completed = 0
doc_name = DOC_NAME_DEFAULT


class CameraError(Exception):
    pass


def compare_points(point):
    return point[0] + point[1]


def get_document_points(image):
    # Preprocess image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, GAUSSIAN_BLUR_KERNEL, 0)
    _, image_thres = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the image
    contours, _ = cv2.findContours(image_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find document contour (largest 4 point contour in the contours list)
    max_area = 0
    document_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approximate = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
        if area > max_area and len(approximate) == 4:
            document_contour = approximate
            max_area = area

    # Check if the document contour was found
    if document_contour is None:
        return np.zeros((4, 2))

    # Ensure documentPoints is the correct shape for getPerspectiveTransform
    document_points = sorted(document_contour.reshape(4, 2).astype("float32"), key=compare_points)
    document_points = np.array(document_points, "float32")

    return document_points


def get_image_scan(image, document_points):
    # Compute document contour shape
    height = max(document_points[1][1] - document_points[0][1], document_points[2][1] - document_points[0][1])
    width = max(document_points[1][0] - document_points[0][0], document_points[2][0] - document_points[0][0])

    # Check for page tab (2 A4 pages)
    tab = height / width < 28 / 21

    # Define destination points for perspective transform
    if tab:
        dst_points = np.array([(0, 0), (0, int(PAGE_HEIGHT)), (int(PAGE_WIDTH) * 2, 0),
                               (int(PAGE_WIDTH) * 2, int(PAGE_HEIGHT))], dtype="float32")
        dst_size = (int(PAGE_WIDTH) * 2, int(PAGE_HEIGHT))
    else:
        dst_points = np.array([(0, 0), (int(PAGE_WIDTH), 0), (0, int(PAGE_HEIGHT)),
                               (int(PAGE_WIDTH), int(PAGE_HEIGHT))], dtype="float32")
        dst_size = (int(PAGE_WIDTH), int(PAGE_HEIGHT))

    # Get perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(document_points, dst_points)
    image_scan = cv2.warpPerspective(image, perspective_matrix, dst_size)

    return [image_scan, tab]


def scan_click():
    global scan_requested
    scan_requested += 1


def update():
    global doc_name, document, scan_requested, scans_completed

    doc_name = "Documents/" + doc_name_entry.get() + ".pdf" if doc_name_entry.get() else DOC_NAME_DEFAULT

    ret, frame = camera.read()
    if not ret:
        raise CameraError("Web camera stopped working")
    copy = frame.copy()
    points = get_document_points(frame)
    show_points = points.reshape(4, 2).astype("int32")
    if points is not None:
        cv2.rectangle(copy, tuple(show_points[0]), tuple(show_points[3]), (0, 255, 0), 5)

        if scan_requested > scans_completed:
            scans_completed += 1
            scan, tab = get_image_scan(frame, points)
            if tab:
                # Crop the tab (2 pages) scan in two
                height, width, _ = scan.shape
                document.append(scan[:, 0:int(width / 2)])
                document.append(scan[:, int(width / 2):])
            else:
                document.append(scan)

    copy = cv2.cvtColor(cv2.resize(copy, (800, 450)), cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(image=Image.fromarray(copy))
    image_label.config(image=image)
    image_label.image = image

    root.after(1, update)


# Initialize Camera
try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise CameraError("Cannot open web camera")
except CameraError as e:
    print(e)
    exit(1)

# GUI Setup
root = tk.Tk()
root.title("A4Capture")
root.iconbitmap('A4.ico')

doc_name_label = tk.Label(root, text="Enter document name:")
doc_name_label.grid(row=0, column=0)

doc_name_entry = tk.Entry(root, width=90)
doc_name_entry.grid(row=0, column=1)

quit_button = tk.Button(root, text="Quit", command=root.quit)
quit_button.grid(row=0, column=2)

image_label = tk.Label(root)
image_label.grid(row=1, column=1)

scan_button = tk.Button(root, text="Scan", command=scan_click, padx=400)
scan_button.grid(row=2, column=1)

root.after(1, update)
root.mainloop()

# Stop recording
camera.release()
cv2.destroyAllWindows()

# Create a PDF canvas
c = canvas.Canvas(doc_name, A4)

is_temp = False
for page in document:
    is_temp = True
    cv2.imwrite("temp.jpg", page)
    c.drawImage(ImageReader("temp.jpg"), 0, 0, PAGE_WIDTH, PAGE_HEIGHT)
    c.showPage()

if is_temp:
    os.remove("temp.jpg")
c.save()
