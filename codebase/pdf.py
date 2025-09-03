from fpdf import FPDF
import markdown2
from bs4 import BeautifulSoup
import re



class PDF(FPDF):
    def header(self):
        """self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Executive Summary', ln=True, align='C')
        self.ln(5)"""
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def project_header(self, project_name):
        self.set_fill_color(61, 118, 46)  # #3D762E green background
        self.set_text_color(255, 255, 255)  # white text
        self.set_font('Arial', 'B', 16)
        # Full width cell with background color
        self.cell(0, 12, project_name, ln=True, fill=True)
        # Reset colors back to normal after header
        self.set_text_color(0, 0, 0)  # black

    def markdown_to_pdf(self, markdown_text):
        html = markdown2.markdown(markdown_text)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(recursive=False):
            if tag.name == 'h1':
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, tag.text, ln=True)
                self.ln(2)
            elif tag.name == 'h2':
                self.set_font('Arial', 'B', 14)
                self.cell(0, 9, tag.text, ln=True)
                self.ln(2)
            elif tag.name == 'p':
                self.set_font('Arial', '', 12)
                self.multi_cell(0, 7, tag.text)
                self.ln(2)
            elif tag.name in ['ul', 'ol']:
                items = tag.find_all('li', recursive=False)
                for idx, item in enumerate(items, 1):
                    bullet = '-' if tag.name == 'ul' else f'{idx}.'
                    self.set_font('Arial', '', 12)
                    self.cell(8)
                    self.multi_cell(0, 7, f'{bullet} {item.text}')
                self.ln(2)


def clean_markdown_codeblock(text):
    # remove the outer ```markdown ```
    match = re.search(r'```markdown(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text