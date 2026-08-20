import os
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

APP_TITLE = "Image Processing Studio"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Output.jpeg")

KERNELS = {
    "Mean": np.ones((3, 3), dtype=np.float32) / 9.0,
    "Gaussian": np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.float32
    ) / 16.0,
    "Sobel X": np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=np.float32
    ),
    "Sobel Y": np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]], dtype=np.float32
    ),
    "Laplacian": np.array(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=np.float32
    ),
}


class ImageProcessingStudio(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1250x760")
        self.minsize(1000, 650)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.original = None
        self.processed = None
        self.image_path = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.build_header()
        self.build_sidebar()
        self.build_viewer()
        self.build_status()

    def build_header(self):
        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="🖼️  IMAGE PROCESSING STUDIO",
            font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=22, pady=16)

        ctk.CTkLabel(
            header,
            text="OpenCV • NumPy • Pillow • CustomTkinter",
            text_color="gray70",
            font=ctk.CTkFont(size=13)
        ).grid(row=0, column=1, sticky="e", padx=22)

    def build_sidebar(self):
        side = ctk.CTkFrame(self, width=285, corner_radius=0)
        side.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        side.grid_propagate(False)

        ctk.CTkLabel(
            side, text="CONTROLS",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(padx=18, pady=(18, 12), anchor="w")

        ctk.CTkButton(
            side, text="📂  Open Image", command=self.open_image, height=40
        ).pack(fill="x", padx=18, pady=5)

        self.filter_var = ctk.StringVar(value="Original")
        ctk.CTkLabel(side, text="Filter / Operation").pack(
            padx=18, pady=(18, 4), anchor="w"
        )

        self.filter_menu = ctk.CTkOptionMenu(
            side,
            variable=self.filter_var,
            values=["Original", "Grayscale", "Mean", "Gaussian", "Sobel X",
                    "Sobel Y", "Sobel Magnitude", "Laplacian"],
            command=self.on_filter_change
        )
        self.filter_menu.pack(fill="x", padx=18, pady=5)

        ctk.CTkButton(
            side, text="⚙  Apply Filter", command=self.apply_filter, height=40
        ).pack(fill="x", padx=18, pady=5)

        ctk.CTkButton(
            side, text="💾  Save Output.jpeg", command=self.save_output, height=40
        ).pack(fill="x", padx=18, pady=5)

        ctk.CTkButton(
            side, text="↩  Reset", command=self.reset_image, height=38
        ).pack(fill="x", padx=18, pady=5)

        ctk.CTkLabel(
            side, text="Analysis",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=18, pady=(24, 8), anchor="w")

        ctk.CTkButton(
            side, text="🔢  Show Pixel Matrix",
            command=self.show_pixel_matrix
        ).pack(fill="x", padx=18, pady=4)

        ctk.CTkButton(
            side, text="🔳  Show Kernel",
            command=self.show_kernel
        ).pack(fill="x", padx=18, pady=4)

        ctk.CTkLabel(
            side,
            text="Tip: Open an image → choose a filter → Apply → Save Output.jpeg",
            wraplength=235,
            justify="left",
            text_color="gray65"
        ).pack(padx=18, pady=(25, 10), anchor="w")

    def build_viewer(self):
        viewer = ctk.CTkFrame(self)
        viewer.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        viewer.grid_columnconfigure((0, 1), weight=1)
        viewer.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            viewer, text="ORIGINAL",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=10)

        ctk.CTkLabel(
            viewer, text="PROCESSED OUTPUT",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=1, pady=10)

        self.original_label = ctk.CTkLabel(
            viewer, text="No image loaded", fg_color=("gray15", "gray15"),
            corner_radius=10
        )
        self.original_label.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=10
        )

        self.processed_label = ctk.CTkLabel(
            viewer, text="Output will appear here", fg_color=("gray15", "gray15"),
            corner_radius=10
        )
        self.processed_label.grid(
            row=1, column=1, sticky="nsew", padx=10, pady=10
        )

        self.file_label = ctk.CTkLabel(
            viewer, text="File: —", text_color="gray65"
        )
        self.file_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))

    def build_status(self):
        self.status = ctk.CTkLabel(
            self,
            text="Ready — open an image to begin.",
            anchor="w",
            text_color="gray70"
        )
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 8))

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not read the selected image.")
            return

        self.image_path = path
        self.original = image
        self.processed = image.copy()
        self.filter_var.set("Original")

        self.file_label.configure(text=f"File: {os.path.basename(path)}")
        self.display_images()
        self.status.configure(
            text=f"Loaded: {os.path.basename(path)}  |  Size: {image.shape[1]} × {image.shape[0]}"
        )

    def on_filter_change(self, _value):
        # Selecting a filter does not change the image until Apply is pressed.
        pass

    def apply_filter(self):
        if self.original is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return

        name = self.filter_var.get()
        img = self.original.copy()

        try:
            if name == "Original":
                result = img

            elif name == "Grayscale":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            elif name == "Mean":
                result = cv2.blur(img, (3, 3))

            elif name == "Gaussian":
                result = cv2.GaussianBlur(img, (5, 5), 0)

            elif name == "Sobel X":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                result = cv2.convertScaleAbs(result)

            elif name == "Sobel Y":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                result = cv2.convertScaleAbs(result)

            elif name == "Sobel Magnitude":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32))
                result = cv2.normalize(
                    magnitude, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

            elif name == "Laplacian":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = cv2.Laplacian(gray, cv2.CV_64F)
                result = cv2.convertScaleAbs(result)

            else:
                result = img

            self.processed = result
            self.display_images()
            self.auto_save_output()

            shape = result.shape
            self.status.configure(
                text=f"Applied: {name}  |  Output shape: {shape}  |  Saved: {OUTPUT_FILE}"
            )

        except Exception as exc:
            messagebox.showerror("Processing Error", str(exc))

    def auto_save_output(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if self.processed is not None:
            cv2.imwrite(OUTPUT_FILE, self.processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def save_output(self):
        if self.processed is None:
            messagebox.showwarning("No output", "Process an image first.")
            return

        self.auto_save_output()
        messagebox.showinfo(
            "Saved",
            f"Output image saved successfully:\n{os.path.abspath(OUTPUT_FILE)}"
        )
        self.status.configure(text=f"Output saved: {OUTPUT_FILE}")

    def reset_image(self):
        if self.original is None:
            return
        self.processed = self.original.copy()
        self.filter_var.set("Original")
        self.display_images()
        self.auto_save_output()
        self.status.configure(text="Reset to original image.")

    def display_images(self):
        self.show_cv_image(self.original, self.original_label)
        self.show_cv_image(self.processed, self.processed_label)

    def show_cv_image(self, image, label):
        if image is None:
            return

        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        # Fit image inside the viewer without distorting it.
        max_w, max_h = 500, 500
        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        label.configure(image=photo, text="")
        label.image = photo

    def show_pixel_matrix(self):
        if self.processed is None:
            messagebox.showwarning("No image", "Please open/process an image first.")
            return

        if len(self.processed.shape) == 2:
            matrix = self.processed[:10, :10]
            text = "First 10 × 10 grayscale pixel values:\n\n"
            text += np.array2string(matrix, separator=" ")
        else:
            rgb = cv2.cvtColor(self.processed, cv2.COLOR_BGR2RGB)
            matrix = rgb[:5, :5]
            text = "First 5 × 5 RGB pixel values:\n\n"
            text += np.array2string(matrix, separator=" ")

        self.show_text_window("Pixel Matrix Analysis", text)

    def show_kernel(self):
        name = self.filter_var.get()

        if name not in KERNELS:
            messagebox.showinfo(
                "Kernel",
                "Select Mean, Gaussian, Sobel X, Sobel Y, or Laplacian first."
            )
            return

        kernel = KERNELS[name]
        text = f"{name} kernel:\n\n{np.array2string(kernel, precision=3, suppress_small=True)}"
        self.show_text_window("Kernel Visualization", text)

    def show_text_window(self, title, text):
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("620x430")
        win.grab_set()

        textbox = ctk.CTkTextbox(win, font=("Consolas", 15))
        textbox.pack(fill="both", expand=True, padx=15, pady=15)
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app = ImageProcessingStudio()
    app.mainloop()
