from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re

def convert_md_to_pdf(input_md, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    local_styles = getSampleStyleSheet()
    
    # Custom Styles
    style_h1 = ParagraphStyle('H1', parent=local_styles['Heading1'], fontSize=18, spaceAfter=12, textColor=colors.darkblue)
    style_h2 = ParagraphStyle('H2', parent=local_styles['Heading2'], fontSize=14, spaceAfter=10, textColor=colors.midnightblue)
    style_h3 = ParagraphStyle('H3', parent=local_styles['Heading3'], fontSize=12, spaceAfter=8, textColor=colors.teal)
    style_body = ParagraphStyle('Body', parent=local_styles['Normal'], fontSize=10, spaceAfter=6, leading=14)
    style_code = ParagraphStyle('Code', parent=local_styles['Code'], fontSize=8, backColor=colors.lightgrey, borderPadding=5, spaceAfter=10)

    story = []

    with open(input_md, 'r') as f:
        lines = f.readlines()

    in_code_block = False
    code_block_content = []

    for line in lines:
        line = line.rstrip()
        
        # Code Block Handling
        if line.startswith("```"):
            if in_code_block:
                # End block
                full_code = "\n".join(code_block_content)
                story.append(Preformatted(full_code, style_code))
                story.append(Spacer(1, 12))
                in_code_block = False
                code_block_content = []
            else:
                # Start block
                in_code_block = True
            continue
        
        if in_code_block:
            code_block_content.append(line)
            continue

        # Normal Markdown Parsing
        if line.startswith("# "):
            story.append(Paragraph(line[2:], style_h1))
            story.append(Spacer(1, 6))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], style_h2))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], style_h3))
        elif line.startswith("---"):
            story.append(Spacer(1, 20))
        elif len(line.strip()) > 0:
            # Escape XML special chars first
            text = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Regex for Bold **text** -> <b>text</b>
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            
            # Regex for Inline Code `text` -> <font face="Courier" backColor="lightgrey">text</font>
            # Note: ReportLab doesn't support 'backColor' in font tag easily, simplified to font-face
            text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
            
            story.append(Paragraph(text, style_body))
        
    doc.build(story)
    print(f"PDF Generated: {output_pdf}")

if __name__ == "__main__":
    src = "/Users/vedangavaghade/.gemini/antigravity/brain/8e024dcb-1b90-43ee-a104-33d585362a2a/project_documentation.md"
    dst = "/Users/vedangavaghade/.gemini/antigravity/brain/8e024dcb-1b90-43ee-a104-33d585362a2a/DeepDebris_Project_Documentation.pdf"
    convert_md_to_pdf(src, dst)
