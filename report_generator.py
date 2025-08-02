import os
import datetime
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, "Model Evaluation Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

    def add_section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_text_color(0)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def add_text_block(self, text):
        self.set_font("Courier", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_image_centered(self, image_path, w=150):
        if os.path.exists(image_path):
            self.image(image_path, x=(210 - w) / 2, w=w)
            self.ln(5)

# === PDF Generation Function ===
def generate_model_report_pdf(report_path='outputs/demo_data_classification_report.txt',
                              cm_path='outputs/demo_data_confusion_matrix.png',
                              roc_path='outputs/demo_data_roc_multiclass.png',
                              output_pdf='outputs/demo_data_model_report.pdf'):

    print("[INFO] Generating PDF report...")

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Section 1: Report Summary
    pdf.add_section_title("1. Classification Report")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_text = f.read()
        pdf.add_text_block(report_text)
    else:
        pdf.add_text_block("No classification report found.")

    # Section 2: Confusion Matrix
    pdf.add_section_title("2. Confusion Matrix (Normalized)")
    pdf.add_image_centered(cm_path)

    # Section 3: ROC Curve
    pdf.add_section_title("3. Multiclass ROC Curve")
    pdf.add_image_centered(roc_path)

    # Section 4: Meta Info
    pdf.add_section_title("4. Meta Information")
    pdf.add_text_block("Model: XGBoost with SMOTE\nDataset: CIC-IDS-2017 (Cleaned)\nTest Size: 25%\nFormat: Auto-generated")

    # Save
    pdf.output(output_pdf)
    print(f"[INFO] PDF report generated at: {output_pdf}")

