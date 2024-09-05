import sys

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


class CameraError(Exception):
    pass


class DocumentScanner:
    def __init__(self, camera_index=0):
        self.document = []
        self.scan_requested = 0
        self.scans_completed = 0
        self.doc_path = ""
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise CameraError("Cannot open web camera")

    @staticmethod
    def compare_points(point):
        return point[0] + point[1]

    @staticmethod
    def get_generic_doc_path():
        return "Documents/" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + ".pdf"

    def get_document_points(self, image):
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
        document_points = sorted(document_contour.reshape(4, 2).astype("float32"), key=self.compare_points)
        document_points = np.array(document_points, "float32")

        return document_points

    @staticmethod
    def get_image_scan(image, document_points):
        # Compute document contour shape
        height = max(document_points[1][1] - document_points[0][1], document_points[2][1] - document_points[0][1])
        width = max(document_points[1][0] - document_points[0][0], document_points[2][0] - document_points[0][0])

        # Check for page tab (2 A4 pages)
        tab = height / width < 28 / 21

        # Define destination points for perspective transform
        if tab:
            dst_points = np.array([(0, 0), (0, int(PAGE_HEIGHT) * 4), (int(PAGE_WIDTH) * 4 * 2, 0),
                                   (int(PAGE_WIDTH) * 4 * 2, int(PAGE_HEIGHT) * 4)], dtype="float32")
            dst_size = (int(PAGE_WIDTH) * 4 * 2, int(PAGE_HEIGHT) * 4)
        else:
            dst_points = np.array([(0, 0), (int(PAGE_WIDTH) * 4, 0), (0, int(PAGE_HEIGHT) * 4),
                                   (int(PAGE_WIDTH) * 4, int(PAGE_HEIGHT) * 4)], dtype="float32")
            dst_size = (int(PAGE_WIDTH) * 4, int(PAGE_HEIGHT) * 4)

        # Get perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(document_points, dst_points)
        image_scan = cv2.warpPerspective(image, perspective_matrix, dst_size)

        return [image_scan, tab]

    def save_document(self, doc_path):
        # Create a PDF canvas
        c = canvas.Canvas(doc_path, A4)
        is_temp = False
        for page in self.document:
            is_temp = True
            cv2.imwrite("temp.jpg", page)
            c.drawImage(ImageReader("temp.jpg"), 0, 0, PAGE_WIDTH, PAGE_HEIGHT)
            c.showPage()

        if is_temp:
            os.remove("temp.jpg")
        c.save()

    def release_camera(self):
        self.camera.release()
        cv2.destroyAllWindows()


class DocumentScannerApp:
    def __init__(self, root, scanner):
        self.root = root
        self.scanner = scanner
        self.root.title("A4Capture")
        self.root.iconbitmap('A4.ico')

        self.doc_name_label = tk.Label(root, text="Enter document name:")
        self.doc_name_label.grid(row=0, column=0)

        self.doc_name_entry = tk.Entry(root, width=90)
        self.doc_name_entry.grid(row=0, column=1)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit, padx=50)
        self.quit_button.grid(row=0, column=2)

        self.image_label = tk.Label(root)
        self.image_label.grid(row=1, column=1)

        self.scan_button = tk.Button(root, text="Scan", command=self.scan_click, padx=385)
        self.scan_button.grid(row=2, column=1)

        self.save_button = tk.Button(root, text="Save", command=self.save_click, padx=50)
        self.save_button.grid(row=2, column=2)

        self.root.after(1, self.update)

    def scan_click(self):
        self.scanner.scan_requested += 1

    def save_click(self):
        doc_name = self.doc_name_entry.get()
        doc_path = f"Documents/{doc_name}.pdf" if doc_name else self.scanner.get_generic_doc_path()
        self.scanner.save_document(doc_path)

    def update(self):
        ret, frame = self.scanner.camera.read()
        if not ret:
            raise CameraError("Web camera stopped working")
        copy = frame.copy()
        points = self.scanner.get_document_points(frame)
        if points is not None:
            show_points = points.reshape(4, 2).astype("int32")
            cv2.rectangle(copy, tuple(show_points[0]), tuple(show_points[3]), (0, 255, 0), 5)

            if self.scanner.scan_requested > self.scanner.scans_completed:
                self.scanner.scans_completed += 1
                scan, tab = self.scanner.get_image_scan(frame, points)
                if tab:
                    # Crop the tab (2 pages) scan in two
                    height, width, _ = scan.shape
                    self.scanner.document.append(scan[:, 0:int(width / 2)])
                    self.scanner.document.append(scan[:, int(width / 2):])
                else:
                    self.scanner.document.append(scan)

        copy = cv2.cvtColor(cv2.resize(copy, (800, 450)), cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(image=Image.fromarray(copy))
        self.image_label.config(image=image)
        self.image_label.image = image

        self.root.after(30, self.update)


def main():
    try:
        scanner = DocumentScanner()
    except CameraError as e:
        print(e)
        return

    root = tk.Tk()
    app = DocumentScannerApp(root, scanner)
    root.mainloop()
    scanner.release_camera()


if __name__ == "__main__":
    main()
