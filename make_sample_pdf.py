"""(Optional) Regenerates data/sample_academic_regulations.pdf. Needs: pip install reportlab"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent / "data" / "sample_academic_regulations.pdf"
styles = getSampleStyleSheet()

PAGE_1 = [
    ("h", "Sample University - Academic Regulations (fictional data for testing)"),
    ("h2", "Attendance Rules"),
    ("p", "A student must have a minimum attendance of 75 percent in each course to be allowed to write "
          "the semester examination. Attendance is calculated separately for every course, counting "
          "lectures, tutorials and laboratory sessions."),
    ("h2", "Condonation of Shortage"),
    ("p", "A student whose attendance is between 65 and 74 percent may apply for condonation of shortage "
          "with a valid reason such as illness, supported by a medical certificate. Condonation is "
          "allowed only once in a programme and a fee is charged."),
    ("h2", "Examination Eligibility"),
    ("p", "To be eligible for the semester examination a student must satisfy the attendance rule, clear "
          "all dues and complete the internal assessment of the course."),
]
PAGE_2 = [
    ("h2", "Internal Assessment"),
    ("p", "Each course has 40 marks of internal assessment and 60 marks of semester examination. The "
          "internal marks are given for mid-semester tests, assignments and laboratory records."),
    ("h2", "Passing Marks"),
    ("p", "A student must score at least 40 percent of the total marks in a course to pass it. A student "
          "who fails may write the supplementary examination in the next semester."),
    ("h2", "Grading"),
    ("p", "Grades are awarded on a 10-point scale. A score of 90 percent or more earns the grade O "
          "(outstanding) and 80 to 89 percent earns the grade A+."),
]


def build() -> None:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm)
    story = []
    for part, page in enumerate((PAGE_1, PAGE_2)):
        for kind, text in page:
            style = {"h": styles["Title"], "h2": styles["Heading2"], "p": styles["BodyText"]}[kind]
            story += [Paragraph(text, style), Spacer(1, 8)]
        if part == 0:
            story.append(PageBreak())
    doc.build(story)
    print("Created", OUT)


if __name__ == "__main__":
    build()
