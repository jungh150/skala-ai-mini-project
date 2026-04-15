"""
Formatting Node: 검증된 초안을 PDF로 변환
- ReportLab 사용
- 목차 구조 자동 생성
- LLM 판단 불필요 (변환 작업만 수행)
"""
import os
import re
from datetime import datetime
from typing import List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from agents.state import AgentState

OUTPUT_DIR = "outputs"

# 색상 정의
COLOR_PRIMARY = HexColor("#1E3A5F")      # 네이비
COLOR_SECONDARY = HexColor("#2E86AB")    # 블루
COLOR_ACCENT = HexColor("#E84855")       # 레드
COLOR_LIGHT = HexColor("#F5F5F5")        # 연회색
COLOR_BORDER = HexColor("#CCCCCC")       # 테두리


def _register_fonts():
    """한글 폰트 등록 (시스템 폰트 사용)"""
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/Windows/Fonts/malgun.ttf",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("Korean", path))
                pdfmetrics.registerFont(TTFont("Korean-Bold", path))
                return "Korean"
            except Exception:
                continue
    
    return "Helvetica"


def _get_styles(font_name: str) -> dict:
    """스타일 정의"""
    base = getSampleStyleSheet()
    
    styles = {
        "title": ParagraphStyle(
            "CustomTitle",
            fontName=font_name + "-Bold" if font_name == "Korean" else font_name,
            fontSize=22,
            textColor=COLOR_PRIMARY,
            spaceAfter=12,
            spaceBefore=6,
            leading=28,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName=font_name,
            fontSize=11,
            textColor=HexColor("#666666"),
            spaceAfter=20,
            leading=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName=font_name,
            fontSize=14,
            textColor=COLOR_PRIMARY,
            spaceBefore=18,
            spaceAfter=8,
            borderPad=6,
            leading=20,
            leftIndent=0,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName=font_name,
            fontSize=12,
            textColor=COLOR_SECONDARY,
            spaceBefore=12,
            spaceAfter=6,
            leading=18,
            leftIndent=10,
        ),
        "h3": ParagraphStyle(
            "H3",
            fontName=font_name,
            fontSize=11,
            textColor=HexColor("#333333"),
            spaceBefore=8,
            spaceAfter=4,
            leading=16,
            leftIndent=20,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=font_name,
            fontSize=10,
            textColor=black,
            spaceAfter=6,
            leading=16,
            leftIndent=10,
            rightIndent=10,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=font_name,
            fontSize=10,
            textColor=black,
            spaceAfter=4,
            leading=15,
            leftIndent=20,
            bulletIndent=10,
        ),
        "summary_box": ParagraphStyle(
            "SummaryBox",
            fontName=font_name,
            fontSize=10,
            textColor=HexColor("#1E3A5F"),
            leading=16,
            leftIndent=10,
        ),
        "reference": ParagraphStyle(
            "Reference",
            fontName=font_name,
            fontSize=9,
            textColor=HexColor("#555555"),
            spaceAfter=4,
            leading=13,
            leftIndent=10,
        ),
        "trl_note": ParagraphStyle(
            "TRLNote",
            fontName=font_name,
            fontSize=9,
            textColor=HexColor("#8B0000"),
            leading=13,
            leftIndent=10,
            spaceBefore=4,
        ),
    }
    return styles


def _parse_draft(draft: str) -> List[Tuple[str, str]]:
    """
    마크다운 초안을 (type, content) 튜플 리스트로 파싱
    type: title, h1, h2, h3, body, bullet, hr, trl_note
    """
    elements = []
    lines = draft.split("\n")
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if stripped.startswith("# "):
            elements.append(("h1", stripped[2:]))
        elif stripped.startswith("## "):
            elements.append(("h2", stripped[3:]))
        elif stripped.startswith("### "):
            elements.append(("h3", stripped[4:]))
        elif stripped.startswith("---"):
            elements.append(("hr", ""))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            elements.append(("bullet", stripped[2:]))
        elif stripped.startswith("※") or "TRL 4~6" in stripped and "추정" in stripped:
            elements.append(("trl_note", stripped))
        else:
            elements.append(("body", stripped))
    
    return elements


def _make_cover_page(styles: dict, font_name: str) -> List:
    """표지 생성"""
    elements = []
    elements.append(Spacer(1, 3*cm))
    
    # 상단 색 박스
    header_data = [["SK Hynix R&D 기술 전략 분석 보고서"]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, -1), white),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 18),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING", (0, 0), (-1, -1), 15),
        ("RIGHTPADDING", (0, 0), (-1, -1), 15),
        ("ROUNDEDCORNERS", [5, 5, 5, 5]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 1*cm))
    
    # 부제목
    elements.append(Paragraph(
        "HBM4 · PIM · CXL 경쟁사 기술 성숙도 및 위협 수준 분석",
        ParagraphStyle("cover_sub", fontName=font_name, fontSize=13,
                      textColor=COLOR_SECONDARY, spaceAfter=8, alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 2*cm))
    
    # 메타 정보 테이블
    meta_data = [
        ["분석 대상", "HBM4, PIM (Processing-In-Memory), CXL (Compute Express Link)"],
        ["분석 경쟁사", "Samsung Electronics, Micron Technology"],
        ["분석 기준", "TRL (Technology Readiness Level) 9단계"],
        ["작성일", datetime.now().strftime("%Y년 %m월 %d일")],
        ["보고서 유형", "R&D 전략 참고용 (내부 배포)"],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 13*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), COLOR_LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 2*cm))
    
    # 주의사항 - Paragraph 사용으로 자동 줄바꿈
    notice_style = ParagraphStyle(
        "Notice",
        fontName=font_name,
        fontSize=9,
        textColor=HexColor("#856404"),
        leading=14,
        leftIndent=0,
        rightIndent=0,
    )
    notice_para = Paragraph(
        "⚠ 주의: TRL 4~6 구간의 추정값은 특허 출원 패턴, 학회 발표 빈도, 채용공고 키워드 등 간접 지표 기반이며 실제와 다를 수 있습니다.",
        notice_style
    )
    notice_data = [[notice_para]]
    notice_table = Table(notice_data, colWidths=[15*cm])
    notice_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FFF3CD")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("BOX", (0, 0), (-1, -1), 1, HexColor("#FFECB5")),
    ]))
    elements.append(notice_table)
    elements.append(PageBreak())
    return elements


def generate_pdf(draft: str, output_path: str) -> str:
    """보고서 초안을 PDF로 변환"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    font_name = _register_fonts()
    styles = _get_styles(font_name)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    
    story = []
    
    # 표지
    story.extend(_make_cover_page(styles, font_name))
    
    # 본문 파싱
    parsed = _parse_draft(draft)
    
    in_summary = False
    summary_content = []
    
    for elem_type, content in parsed:
        if not content and elem_type != "hr":
            continue
        
        # 안전한 문자 처리
        safe_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        if elem_type == "h1":
            if "SUMMARY" in content.upper():
                in_summary = True
                story.append(Paragraph(f"■ {safe_content}", styles["h1"]))
                story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY))
            else:
                in_summary = False
                # Summary 박스 닫기
                if summary_content:
                    summary_text = "<br/>".join(summary_content)
                    box_data = [[Paragraph(summary_text, styles["summary_box"])]]
                    box_table = Table(box_data, colWidths=[15*cm])
                    box_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#EBF4FF")),
                        ("BORDER", (0, 0), (-1, -1), 1, COLOR_SECONDARY),
                        ("TOPPADDING", (0, 0), (-1, -1), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("LEFTPADDING", (0, 0), (-1, -1), 15),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                    ]))
                    story.append(box_table)
                    story.append(Spacer(1, 0.3*cm))
                    summary_content = []
                
                story.append(Spacer(1, 0.3*cm))
                story.append(Paragraph(f"■ {safe_content}", styles["h1"]))
                story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY))
        
        elif elem_type == "h2":
            if in_summary:
                summary_content.append(f"<b>{safe_content}</b>")
            else:
                story.append(Paragraph(f"▶ {safe_content}", styles["h2"]))
        
        elif elem_type == "h3":
            if in_summary:
                summary_content.append(safe_content)
            else:
                story.append(Paragraph(f"◆ {safe_content}", styles["h3"]))
        
        elif elem_type == "bullet":
            if in_summary:
                summary_content.append(f"• {safe_content}")
            else:
                story.append(Paragraph(f"• {safe_content}", styles["bullet"]))
        
        elif elem_type == "trl_note":
            story.append(Paragraph(safe_content if safe_content.startswith("※") else f"※ {safe_content}", styles["trl_note"]))
        
        elif elem_type == "hr":
            story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER))
            story.append(Spacer(1, 0.2*cm))
        
        elif elem_type == "body":
            if in_summary:
                summary_content.append(safe_content)
            else:
                # 표 형식 감지 (| 포함)
                if "|" in content and content.count("|") >= 3:
                    _add_table_from_markdown(story, content, font_name)
                else:
                    story.append(Paragraph(safe_content, styles["body"]))
    
    # 마지막 summary 박스 닫기
    if summary_content:
        summary_text = "<br/>".join(summary_content)
        box_data = [[Paragraph(summary_text, styles["summary_box"])]]
        box_table = Table(box_data, colWidths=[15*cm])
        box_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#EBF4FF")),
            ("BORDER", (0, 0), (-1, -1), 1, COLOR_SECONDARY),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 15),
            ("RIGHTPADDING", (0, 0), (-1, -1), 15),
        ]))
        story.append(box_table)
    
    # 페이지 번호 추가 함수
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 9)
        canvas.setFillColor(HexColor("#999999"))
        page_num = canvas.getPageNumber()
        canvas.drawRightString(
            A4[0] - 2*cm, 1.2*cm,
            f"SK Hynix R&D 기술 전략 분석 보고서  |  {page_num}"
        )
        canvas.drawString(
            2*cm, 1.2*cm,
            "CONFIDENTIAL - 내부 배포용"
        )
        canvas.restoreState()
    
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output_path


def _add_table_from_markdown(story: List, line: str, font_name: str):
    """마크다운 테이블을 ReportLab 테이블로 변환"""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if not cells:
        return
    
    # 구분선 행 건너뜀
    if all(c.replace("-", "").replace(":", "").strip() == "" for c in cells):
        return
    
    safe_cells = [c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for c in cells]
    
    col_width = 15*cm / len(safe_cells)
    table_data = [[Paragraph(c, ParagraphStyle(
        "TableCell", fontName=font_name, fontSize=9, leading=13
    )) for c in safe_cells]]
    
    t = Table(table_data, colWidths=[col_width] * len(safe_cells))
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)


def run_formatting_node(state: AgentState) -> AgentState:
    """
    Formatting Node 실행.

    【설계서 준수】
    설계서 명시: formatting_node → supervisor → END
    따라서 PDF 생성 완료 후 next를 "end"로 직접 설정하지 않고
    final_report_path에 경로를 저장한 뒤 supervisor로 복귀한다.
    supervisor가 final_report_path를 확인하고 최종 END를 결정한다.
    """
    print("\n[Formatting Node] PDF 생성 시작...")

    draft = state.get("draft", "")
    if not draft:
        print("[Formatting Node] 오류: 초안 없음")
        return state

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai-mini_output_2반_조정윤신정화김도연권서현_{timestamp}.pdf"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        result_path = generate_pdf(draft, output_path)
        print(f"[Formatting Node] PDF 생성 완료: {result_path}")
        # supervisor로 복귀 — supervisor가 final_report_path 확인 후 END 결정
        state["final_report_path"] = result_path
    except Exception as e:
        print(f"[Formatting Node] PDF 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        # 텍스트 백업 저장 후 supervisor로 복귀
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(draft)
        print(f"[Formatting Node] 텍스트 백업 저장: {txt_path}")
        state["final_report_path"] = txt_path  # 백업 경로라도 저장하여 종료 조건 충족

    return state
