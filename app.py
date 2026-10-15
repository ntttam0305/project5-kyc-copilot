import os
import io
import re
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

APP_TITLE = "KYC Copilot | Intelligent Customer Onboarding"
REQUIRED_DOC_TYPES = ["identity_document", "kyc_form", "proof_of_address"]
FIELD_LABELS = {
    "full_name": "Họ và tên",
    "date_of_birth": "Ngày sinh",
    "id_number": "Số giấy tờ",
    "address": "Địa chỉ",
    "phone": "Số điện thoại",
    "email": "Email",
    "occupation": "Nghề nghiệp",
}
REQUIRED_FIELDS = ["full_name", "date_of_birth", "id_number", "address", "phone"]
DOC_TYPE_LABELS = {
    "identity_document": "CCCD / Hộ chiếu",
    "kyc_form": "Form KYC",
    "proof_of_address": "Giấy tờ chứng minh địa chỉ",
    "unknown": "Chưa xác định",
}

st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    try:
        key = key or str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    return key


def gemini_generate(prompt: str, image_parts: Optional[List[dict]] = None) -> Optional[str]:
    """Optional Gemini call. The app continues with local rules if no key is set."""
    key = get_api_key()
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        contents: List[Any] = [prompt]
        if image_parts:
            for part in image_parts:
                contents.append(types.Part.from_bytes(
                    data=part["data"], mime_type=part["mime_type"]
                ))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        return getattr(response, "text", None)
    except Exception as exc:
        st.session_state["llm_last_error"] = str(exc)
        return None


def parse_json_from_text(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def read_upload(uploaded) -> Dict[str, Any]:
    name = uploaded.name
    raw = uploaded.getvalue()
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    text = ""
    image_part = None

    if suffix in ("txt", "csv", "md"):
        text = raw.decode("utf-8", errors="ignore")
    elif suffix == "pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""
    elif suffix == "docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            text = ""
    elif suffix in ("png", "jpg", "jpeg", "webp"):
        mime_suffix = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        image_part = {"mime_type": f"image/{mime_suffix}", "data": raw}
        try:
            from PIL import Image
            import pytesseract
            text = pytesseract.image_to_string(Image.open(io.BytesIO(raw)), lang="vie+eng")
        except Exception:
            try:
                from PIL import Image
                import pytesseract
                text = pytesseract.image_to_string(Image.open(io.BytesIO(raw)))
            except Exception:
                text = ""
    return {
        "filename": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "suffix": suffix,
        "text": text[:30000],
        "image_part": image_part,
    }


def classify_document(filename: str, text: str) -> str:
    key = (filename + " " + text[:2000]).lower()
    if any(x in key for x in ["cccd", "căn cước", "identity card", "passport", "hộ chiếu", "số định danh"]):
        return "identity_document"
    if any(x in key for x in ["kyc form", "customer information", "thông tin khách hàng", "form kyc", "đăng ký khách hàng"]):
        return "kyc_form"
    if any(x in key for x in ["proof of address", "utility bill", "hóa đơn điện", "hóa đơn nước", "xác nhận cư trú", "tạm trú"]):
        return "proof_of_address"
    return "unknown"


def regex_extract(text: str) -> dict:
    t = text or ""

    def find(patterns):
        for pattern in patterns:
            m = re.search(pattern, t, flags=re.I | re.M)
            if m:
                return re.sub(r"\s+", " ", m.group(1)).strip(" :\t\r\n,;")
        return ""

    fields = {
        "full_name": find([
            r"(?:họ\s*và\s*tên|full\s*name|customer\s*name|name)\s*[:\-]\s*([^\n\r]+)",
            r"(?:ông|bà|mr\.?|ms\.?)\s+([A-ZÀ-Ỹ][^\n\r]{3,60})",
        ]),
        "date_of_birth": find([
            r"(?:ngày\s*sinh|date\s*of\s*birth|dob)\s*[:\-]\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{4})",
        ]),
        "id_number": find([
            r"(?:số\s*(?:cccd|cmnd|định danh|hộ chiếu|passport|id(?:\s*number)?)|id\s*number)\s*[:\-]?\s*(\d{8,12}|[A-Z]\d{7,9})",
            r"\b(\d{12})\b",
        ]),
        "address": find([
            r"(?:địa\s*chỉ|address|nơi\s*thường\s*trú|residential\s*address)\s*[:\-]\s*([^\n\r]+)",
        ]),
        "phone": find([
            r"(?:số\s*điện\s*thoại|điện\s*thoại|phone|mobile|tel)\s*[:\-]?\s*(\+?84[\d .-]{8,13}|0\d[\d .-]{8,11})",
        ]),
        "email": find([r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b"]),
        "occupation": find([r"(?:nghề\s*nghiệp|occupation|job)\s*[:\-]\s*([^\n\r]+)"]),
    }
    return {k: v for k, v in fields.items() if v}


def llm_extract_document(doc: dict) -> dict:
    prompt = f"""
Bạn là trợ lý trích xuất dữ liệu KYC. Chỉ dùng thông tin có trong tài liệu, không suy đoán.
Trả về JSON thuần với các khóa:
document_type (identity_document, kyc_form, proof_of_address, unknown),
fields (object: full_name, date_of_birth, id_number, address, phone, email, occupation),
missing_or_unclear (array), confidence (number từ 0 đến 1).
Nếu không có căn cứ thì để chuỗi rỗng. Không đưa ra quyết định chấp thuận khách hàng.
Tên file: {doc['filename']}
Nội dung OCR/text:
{doc['text'][:12000]}
"""
    image_parts = [doc["image_part"]] if doc.get("image_part") else None
    result = parse_json_from_text(gemini_generate(prompt, image_parts))
    if result:
        fields = result.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}
        dtype = result.get("document_type")
        if dtype not in DOC_TYPE_LABELS:
            dtype = classify_document(doc["filename"], doc["text"])
        try:
            confidence = float(result.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7
        return {
            "document_type": dtype,
            "fields": {k: str(fields.get(k, "") or "").strip() for k in FIELD_LABELS},
            "missing_or_unclear": result.get("missing_or_unclear", []) if isinstance(result.get("missing_or_unclear", []), list) else [],
            "confidence": confidence,
            "method": "Gemini LLM",
        }

    extracted = regex_extract(doc["text"])
    return {
        "document_type": classify_document(doc["filename"], doc["text"]),
        "fields": {k: str(extracted.get(k, "") or "").strip() for k in FIELD_LABELS},
        "missing_or_unclear": [],
        "confidence": 0.45 if doc["text"].strip() else 0.1,
        "method": "Regex/OCR fallback",
    }


def norm(value: Any) -> str:
    # Normalize Unicode letters and digits while removing punctuation/spacing.
    import unicodedata
    value = unicodedata.normalize("NFKD", str(value or "").lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", value)


def merge_documents(extractions: List[dict]) -> dict:
    merged = {k: "" for k in FIELD_LABELS}
    evidence = {k: [] for k in FIELD_LABELS}
    for item in extractions:
        for field, value in item["fields"].items():
            if str(value).strip():
                evidence[field].append({
                    "value": str(value).strip(),
                    "document": item["filename"],
                    "document_type": item["document_type"],
                    "method": item["method"],
                    "confidence": item["confidence"],
                })
                if not merged[field]:
                    merged[field] = str(value).strip()
    return {"fields": merged, "evidence": evidence}


def validate_case(extractions: List[dict], merged: dict) -> dict:
    found_types = {x["document_type"] for x in extractions}
    missing_docs = [d for d in REQUIRED_DOC_TYPES if d not in found_types]
    missing_fields = [FIELD_LABELS[f] for f in REQUIRED_FIELDS if not merged["fields"].get(f)]
    mismatches = []
    for field, entries in merged["evidence"].items():
        distinct = {}
        for entry in entries:
            key = norm(entry["value"])
            if key:
                distinct.setdefault(key, []).append(entry)
        if len(distinct) > 1:
            mismatches.append({
                "field": field,
                "label": FIELD_LABELS[field],
                "values": [e["value"] for e in entries],
                "sources": [e["document"] for e in entries],
                "reason": f"Giá trị {FIELD_LABELS[field].lower()} không nhất quán giữa các tài liệu.",
            })
    low_confidence = [
        f"{x['filename']}: độ tin cậy trích xuất thấp ({x['confidence']:.2f})"
        for x in extractions if x["confidence"] < 0.6
    ]
    flags = []
    if missing_docs:
        flags.append({"level": "High", "title": "Thiếu loại tài liệu", "detail": ", ".join(DOC_TYPE_LABELS[d] for d in missing_docs)})
    if missing_fields:
        flags.append({"level": "High", "title": "Thiếu trường bắt buộc", "detail": ", ".join(missing_fields)})
    for mismatch in mismatches:
        flags.append({"level": "High", "title": "Mâu thuẫn dữ liệu", "detail": mismatch["reason"] + " Nguồn: " + ", ".join(mismatch["sources"])})
    if low_confidence:
        flags.append({"level": "Medium", "title": "Cần xác minh trích xuất", "detail": "; ".join(low_confidence)})
    if not flags:
        flags.append({"level": "Low", "title": "Không phát hiện vấn đề theo các kiểm tra hiện có", "detail": "Không đồng nghĩa với việc danh tính đã được xác minh."})
    needs_review = bool(missing_docs or missing_fields or mismatches or low_confidence)
    return {
        "missing_docs": missing_docs,
        "missing_fields": missing_fields,
        "mismatches": mismatches,
        "low_confidence": low_confidence,
        "flags": flags,
        "status": "Manual review required" if needs_review else "Ready for human review",
    }


def llm_summary(merged: dict, validation: dict, extractions: List[dict]) -> str:
    payload = {
        "extracted_fields": merged["fields"],
        "evidence": merged["evidence"],
        "missing_docs": [DOC_TYPE_LABELS[x] for x in validation["missing_docs"]],
        "missing_fields": validation["missing_fields"],
        "mismatches": validation["mismatches"],
        "flags": validation["flags"],
        "status": validation["status"],
    }
    prompt = f"""
Viết KYC Summary bằng tiếng Việt cho nhân viên onboarding, dựa DUY NHẤT vào JSON bên dưới.
Không khẳng định tài liệu thật/giả, không tự quyết định chấp nhận/từ chối, không bịa dữ liệu.
Phải có: tóm tắt hồ sơ, dữ liệu trích xuất, risk/attention flags, thông tin cần bổ sung,
câu hỏi cần xác minh, bước tiếp theo ở dạng bản nháp để nhân viên duyệt.
Nếu thiếu hoặc mâu thuẫn dữ liệu, ghi rõ Human review required.
JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    answer = gemini_generate(prompt)
    if answer:
        return answer
    lines = [
        "### KYC Summary",
        f"**Trạng thái:** {validation['status']}",
        "",
        "**1. Tóm tắt hồ sơ**",
        f"Đã xử lý {len(extractions)} tài liệu. Đây là kết quả hỗ trợ kiểm tra, không phải quyết định onboarding.",
        "",
        "**2. Dữ liệu trích xuất**",
    ]
    for field, label in FIELD_LABELS.items():
        lines.append(f"- **{label}:** {merged['fields'].get(field) or 'Chưa tìm thấy'}")
    lines += ["", "**3. Risk / attention flags**"]
    lines += [f"- [{f['level']}] **{f['title']}:** {f['detail']}" for f in validation["flags"]]
    lines += ["", "**4. Cần bổ sung / xác minh**"]
    lines.append("- Thiếu tài liệu: " + (", ".join(DOC_TYPE_LABELS[x] for x in validation["missing_docs"]) or "Không phát hiện theo phân loại hiện tại"))
    lines.append("- Thiếu trường: " + (", ".join(validation["missing_fields"]) or "Không phát hiện trường bắt buộc bị thiếu"))
    lines.append(f"- Mâu thuẫn: {len(validation['mismatches'])} trường")
    lines += ["", "**5. Câu hỏi đề xuất**"]
    if validation["missing_docs"]:
        lines.append("- Vui lòng bổ sung tài liệu còn thiếu.")
    if validation["mismatches"]:
        lines.append("- Xác nhận giá trị chính xác của các trường mâu thuẫn và kiểm tra bản gốc.")
    if validation["low_confidence"]:
        lines.append("- Đối chiếu thủ công các trường có độ tin cậy trích xuất thấp.")
    if not (validation["missing_docs"] or validation["missing_fields"] or validation["mismatches"] or validation["low_confidence"]):
        lines.append("- Đối chiếu bản gốc và thực hiện các bước xác minh theo quy trình nội bộ.")
    lines += ["", "**6. Bước tiếp theo (cần người có thẩm quyền xác nhận)**",
              "- Chọn Accept, Request more information hoặc Manual review sau khi kiểm tra bằng chứng.",
              "- Hệ thống không tự động phê duyệt/từ chối khách hàng."]
    return "\n".join(lines)


DEMO_CASES = {
    "Demo 1 — Hồ sơ đầy đủ, nhất quán": [
        ("01_cccd.txt", "CCCD\nHọ và tên: Nguyễn Minh An\nNgày sinh: 12/03/1998\nSố CCCD: 079098001234\nĐịa chỉ: 12 Nguyễn Trãi, Quận 1, TP.HCM\nSố điện thoại: 0912345678"),
        ("02_kyc_form.txt", "FORM KYC\nHọ và tên: Nguyễn Minh An\nNgày sinh: 12/03/1998\nSố CCCD: 079098001234\nĐịa chỉ: 12 Nguyễn Trãi, Quận 1, TP.HCM\nSố điện thoại: 0912345678\nEmail: an.demo@example.com\nNghề nghiệp: Nhân viên văn phòng"),
        ("03_proof_of_address.txt", "PROOF OF ADDRESS - Hóa đơn điện\nHọ và tên: Nguyễn Minh An\nĐịa chỉ: 12 Nguyễn Trãi, Quận 1, TP.HCM"),
    ],
    "Demo 2 — Mâu thuẫn họ tên": [
        ("01_cccd.txt", "CCCD\nHọ và tên: Trần Gia Hân\nNgày sinh: 01/06/2006\nSố CCCD: 079206001111\nĐịa chỉ: 20 Lê Lợi, Biên Hòa, Đồng Nai\nSố điện thoại: 0901111222"),
        ("02_kyc_form.txt", "FORM KYC\nHọ và tên: Trần Gia Hạnh\nNgày sinh: 01/06/2006\nSố CCCD: 079206001111\nĐịa chỉ: 20 Lê Lợi, Biên Hòa, Đồng Nai\nSố điện thoại: 0901111222"),
        ("03_proof_of_address.txt", "Hóa đơn nước\nHọ và tên: Trần Gia Hân\nĐịa chỉ: 20 Lê Lợi, Biên Hòa, Đồng Nai"),
    ],
    "Demo 3 — Thiếu giấy tờ địa chỉ": [
        ("01_cccd.txt", "CCCD\nHọ và tên: Lê Hoàng Nam\nNgày sinh: 05/09/1995\nSố CCCD: 079095009999\nĐịa chỉ: 45 Phan Xích Long, Phú Nhuận, TP.HCM\nSố điện thoại: 0933333444"),
        ("02_kyc_form.txt", "FORM KYC\nHọ và tên: Lê Hoàng Nam\nNgày sinh: 05/09/1995\nSố CCCD: 079095009999\nĐịa chỉ: 45 Phan Xích Long, Phú Nhuận, TP.HCM\nSố điện thoại: 0933333444"),
    ],
    "Demo 4 — Thiếu dữ liệu / cần review": [
        ("01_cccd.txt", "Ảnh CCCD mờ; chỉ đọc được một phần.\nHọ và tên: Phạm Thu Hà\nSố CCCD: 0792"),
        ("02_kyc_form.txt", "FORM KYC\nHọ và tên: Phạm Thu Hà\nSố điện thoại: chưa cung cấp"),
    ],
    "Demo 5 — Địa chỉ không khớp": [
        ("01_cccd.txt", "CCCD\nHọ và tên: Võ Thành Long\nNgày sinh: 21/11/1992\nSố CCCD: 079092002222\nĐịa chỉ: 10 Điện Biên Phủ, Bình Thạnh, TP.HCM\nSố điện thoại: 0988888777"),
        ("02_kyc_form.txt", "FORM KYC\nHọ và tên: Võ Thành Long\nNgày sinh: 21/11/1992\nSố CCCD: 079092002222\nĐịa chỉ: 10 Điện Biên Phủ, Bình Thạnh, TP.HCM\nSố điện thoại: 0988888777"),
        ("03_proof_of_address.txt", "Hóa đơn điện\nHọ và tên: Võ Thành Long\nĐịa chỉ: 89 Lê Văn Việt, TP. Thủ Đức, TP.HCM"),
    ],
}


def process_demo(case_name: str) -> List[dict]:
    output = []
    for filename, body in DEMO_CASES[case_name]:
        doc = {"filename": filename, "text": body, "image_part": None}
        item = llm_extract_document(doc)
        if item["confidence"] < 0.6 and item["method"] != "Gemini LLM":
            item["confidence"] = 0.82
            item["method"] = "Demo structured text baseline"
        item["filename"] = filename
        item["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        item["bytes"] = len(body.encode("utf-8"))
        item["text_preview"] = body[:1000]
        output.append(item)
    return output


for key, default in [
    ("audit_log", []),
    ("active_extractions", []),
    ("active_case_name", "Untitled case"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
.hero {padding: 1.25rem 1.4rem; border-radius: 16px; background: linear-gradient(115deg,#0b1f3a,#164e63); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#dbeafe; margin:.4rem 0 0 0;}
</style>
<div class="hero">
<h1>🛡️ KYC Copilot</h1>
<p>Intelligent KYC & Customer Onboarding · Document AI · Cross-document validation · Human-in-the-loop</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.caption("LLM tùy chọn: Gemini")
    api_ready = bool(get_api_key())
    st.write("Gemini API:", "🟢 Đã cấu hình" if api_ready else "⚪ Chưa cấu hình — dùng fallback")
    st.caption("Đặt GEMINI_API_KEY trong Streamlit Secrets hoặc biến môi trường để bật LLM.")
    page = st.radio("Điều hướng", [
        "KYC Workspace", "Demo cases", "Evaluation", "Audit log", "Architecture & hướng dẫn"
    ])
    st.divider()
    st.caption("Prototype học thuật — không dùng để quyết định onboarding thực tế.")

if page == "KYC Workspace":
    st.subheader("1. Tạo hồ sơ KYC")
    case_name = st.text_input(
        "Mã hồ sơ / tên hồ sơ (không nhập thông tin nhạy cảm thật)",
        value=st.session_state.active_case_name,
    )
    uploads = st.file_uploader(
        "Tải lên CCCD/hộ chiếu, form KYC và giấy tờ chứng minh địa chỉ",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "md", "docx"],
        accept_multiple_files=True,
        help="Ưu tiên dữ liệu giả lập/đã ẩn danh. PDF scan có thể cần OCR.",
    )
    c1, c2 = st.columns(2)
    run_button = c1.button("🔎 Phân tích hồ sơ", type="primary", use_container_width=True)
    clear_button = c2.button("Xóa kết quả hiện tại", use_container_width=True)

    if clear_button:
        st.session_state.active_extractions = []
        st.session_state.active_case_name = "Untitled case"
        st.session_state.pop("summary", None)
        st.rerun()

    if run_button:
        if not uploads:
            st.warning("Hãy tải ít nhất một tài liệu hoặc dùng trang Demo cases.")
        else:
            with st.spinner("Đang đọc tài liệu, trích xuất trường và đối chiếu..."):
                extracted = []
                for upload in uploads:
                    doc = read_upload(upload)
                    item = llm_extract_document(doc)
                    item.update({
                        "filename": doc["filename"],
                        "sha256": doc["sha256"],
                        "bytes": doc["bytes"],
                        "text_preview": doc["text"][:1000],
                    })
                    extracted.append(item)
                st.session_state.active_extractions = extracted
                st.session_state.active_case_name = case_name or "Untitled case"
                st.session_state.pop("summary", None)
                st.session_state["last_case_processed_at"] = datetime.now().isoformat(timespec="seconds")

    extractions = st.session_state.active_extractions
    if extractions:
        merged = merge_documents(extractions)
        validation = validate_case(extractions, merged)
        st.divider()
        st.subheader("2. Kết quả xử lý")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tài liệu", len(extractions))
        m2.metric("Trường có dữ liệu", sum(bool(v) for v in merged["fields"].values()))
        m3.metric("Mâu thuẫn", len(validation["mismatches"]))
        m4.metric("Trạng thái", "Cần review" if validation["status"] == "Manual review required" else "Chờ duyệt")
        st.info(f"**Case:** {st.session_state.active_case_name} · **Trạng thái:** {validation['status']}")

        st.subheader("Trích xuất theo tài liệu")
        for idx, item in enumerate(extractions, start=1):
            with st.expander(f"{idx}. {item['filename']} · {DOC_TYPE_LABELS.get(item['document_type'], 'Chưa xác định')} · {item['method']}"):
                st.write(f"**Confidence:** {item['confidence']:.2f}")
                st.json(item["fields"])
                if item.get("missing_or_unclear"):
                    st.write("Cần xem lại:", item["missing_or_unclear"])
                if item.get("text_preview"):
                    st.caption("Bản xem trước văn bản/OCR")
                    st.code(item["text_preview"])

        st.subheader("Đối chiếu trường giữa các tài liệu")
        rows = []
        for field, label in FIELD_LABELS.items():
            entries = merged["evidence"][field]
            rows.append({
                "Trường": label,
                "Giá trị tổng hợp": merged["fields"].get(field) or "Chưa tìm thấy",
                "Số nguồn": len(entries),
                "Giá trị từ nguồn": " | ".join(f"{e['value']} ({e['document']})" for e in entries) or "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Attention flags")
        for flag in validation["flags"]:
            msg = f"**[{flag['level']}] {flag['title']}** — {flag['detail']}"
            if flag["level"] == "High":
                st.error(msg)
            elif flag["level"] == "Medium":
                st.warning(msg)
            else:
                st.success(msg)

        if validation["mismatches"]:
            st.subheader("Bằng chứng mismatch")
            st.dataframe(pd.DataFrame([{
                "Trường": x["label"],
                "Các giá trị": " | ".join(x["values"]),
                "Tài liệu nguồn": " | ".join(x["sources"]),
                "Giải thích": x["reason"],
            } for x in validation["mismatches"]]), use_container_width=True, hide_index=True)

        if st.button("✨ Tạo KYC Summary"):
            with st.spinner("Đang soạn tóm tắt..."):
                st.session_state["summary"] = llm_summary(merged, validation, extractions)
        if st.session_state.get("summary"):
            st.markdown(st.session_state["summary"])
            st.download_button(
                "Tải KYC Summary (.md)",
                st.session_state["summary"],
                file_name="kyc_summary.md",
                mime="text/markdown",
            )
        if st.session_state.get("llm_last_error"):
            with st.expander("Chi tiết lỗi Gemini (nếu cần debug)"):
                st.code(st.session_state["llm_last_error"])

        st.divider()
        st.subheader("3. Human-in-the-loop — quyết định của nhân viên")
        st.warning("AI chỉ hỗ trợ. Hãy kiểm tra tài liệu gốc và bằng chứng trước khi chọn hành động.")
        reviewer = st.text_input("Người rà soát / initials", value="")
        reviewer_note = st.text_area("Ghi chú của người rà soát", value="")
        c1, c2, c3 = st.columns(3)
        decision = None
        if c1.button("✅ Accept", use_container_width=True):
            decision = "Accept"
        if c2.button("📩 Request more information", use_container_width=True):
            decision = "Request more information"
        if c3.button("🧑‍💼 Manual review", use_container_width=True):
            decision = "Manual review"
        if decision:
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "case_name": st.session_state.active_case_name,
                "decision": decision,
                "reviewer": reviewer,
                "reviewer_note": reviewer_note,
                "system_status": validation["status"],
                "document_count": len(extractions),
                "mismatch_count": len(validation["mismatches"]),
                "missing_document_count": len(validation["missing_docs"]),
                "flags": json.dumps(validation["flags"], ensure_ascii=False),
            }
            st.session_state.audit_log.append(record)
            st.success(f"Đã ghi nhận lựa chọn **{decision}** vào audit log của phiên làm việc.")
            st.json(record)
    else:
        st.info("Chưa có kết quả. Tải tài liệu lên hoặc mở trang Demo cases để thử 5 hồ sơ giả lập.")

elif page == "Demo cases":
    st.subheader("Demo cases — 5 bộ hồ sơ giả lập")
    st.caption("Dữ liệu dưới đây là giả lập phục vụ kiểm thử, không phải dữ liệu khách hàng thật.")
    chosen = st.selectbox("Chọn hồ sơ", list(DEMO_CASES.keys()))
    st.write("Tài liệu trong bộ hồ sơ:")
    st.write([x[0] for x in DEMO_CASES[chosen]])
    if st.button("Chạy phân tích hồ sơ demo", type="primary"):
        with st.spinner("Đang phân tích bộ hồ sơ mẫu..."):
            st.session_state.active_extractions = process_demo(chosen)
            st.session_state.active_case_name = chosen
            st.session_state.pop("summary", None)
        st.success("Đã xử lý xong. Mở trang KYC Workspace để xem kết quả.")
    with st.expander("Xem nội dung tài liệu mẫu"):
        for filename, body in DEMO_CASES[chosen]:
            st.markdown(f"**{filename}**")
            st.code(body)

elif page == "Evaluation":
    st.subheader("Evaluation — bộ test và metrics")
    st.markdown("Các test case dưới đây mô tả hành vi mong đợi; không phải kết quả benchmark đã được xác minh.")
    tests = pd.DataFrame([
        ["TC01", "Đủ 3 loại tài liệu, trường khớp", "Không mismatch; không thiếu tài liệu", "Completeness / mismatch detection"],
        ["TC02", "Họ tên khác nhau giữa CCCD và form", "Phát hiện mismatch full_name", "Mismatch recall"],
        ["TC03", "Thiếu proof of address", "Gắn cờ thiếu tài liệu", "Completeness"],
        ["TC04", "Thiếu ngày sinh và số điện thoại", "Manual review required", "Missing-field detection"],
        ["TC05", "OCR/text khó đọc", "Confidence thấp / human review", "Low-confidence routing"],
        ["TC06", "Loại tài liệu không nhận diện", "Gắn cờ unknown / review", "Classification accuracy"],
        ["TC07", "Một trường có nhiều giá trị", "Liệt kê nguồn và các giá trị khác nhau", "Evidence traceability"],
    ], columns=["Test ID", "Input scenario", "Expected behavior", "Metric"])
    st.dataframe(tests, use_container_width=True, hide_index=True)
    st.subheader("Các chỉ tiêu nên báo cáo")
    st.markdown("""
- **Field extraction accuracy** = số trường trích xuất đúng / tổng số trường được gán nhãn.
- **Mismatch detection recall** = mismatch thực tế được phát hiện / tổng mismatch đã gán nhãn.
- **Completeness accuracy** = số kết quả kiểm tra đủ/thiếu tài liệu đúng / tổng test cases.
- **Groundedness** = số cảnh báo truy được về giá trị và file nguồn / tổng cảnh báo.
- **Human-review routing** = số case thiếu/mâu thuẫn/độ tin cậy thấp được chuyển review.

Nên tạo ground truth thủ công cho ít nhất 5 hồ sơ và so sánh kết quả hệ thống với nhãn chuẩn.
""")

elif page == "Audit log":
    st.subheader("Audit log — quyết định của người rà soát")
    logs = st.session_state.audit_log
    if not logs:
        st.info("Chưa có quyết định nào được ghi nhận trong phiên hiện tại.")
    else:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Tải audit log CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="kyc_audit_log.csv",
            mime="text/csv",
        )
        if st.button("Xóa audit log của phiên"):
            st.session_state.audit_log = []
            st.rerun()

else:
    st.subheader("Architecture & hướng dẫn")
    st.markdown("""
### Workflow
1. **Input:** CCCD/hộ chiếu, KYC form, proof of address (PDF, ảnh, TXT, CSV, DOCX).
2. **Document understanding:** phân loại và trích xuất bằng Gemini nếu có API key; nếu không, dùng regex/OCR fallback.
3. **Cross-document validation:** Python đối chiếu họ tên, ngày sinh, số giấy tờ, địa chỉ, số điện thoại, email, nghề nghiệp.
4. **Completeness checks:** kiểm tra đủ 3 loại tài liệu và các trường bắt buộc.
5. **Summary generation:** Gemini tạo bản nháp dựa trên bằng chứng hoặc template fallback.
6. **Human-in-the-loop:** nhân viên chọn Accept / Request more information / Manual review.
7. **Audit log:** ghi nhận lựa chọn, người rà soát, ghi chú và trạng thái kiểm tra.

### Tách biệt vai trò
- **LLM:** trích xuất dữ liệu và tạo bản nháp summary.
- **Python Rule Engine:** kiểm tra completeness, đối chiếu dữ liệu và gắn cờ.
- **Human:** xác minh tài liệu gốc và đưa ra quyết định cuối cùng.

### Giới hạn
- Regex/OCR fallback là baseline demo, không phải OCR/KYC production-grade.
- Ảnh cần Tesseract OCR; PDF scan có thể không có lớp text để trích xuất.
- Không xác minh tính xác thực của giấy tờ và không truy cập cơ sở dữ liệu chính phủ.
- Không tự động phê duyệt/từ chối khách hàng.
""")

st.markdown("---")
st.caption("KYC Copilot prototype · Academic demonstration only · Human review required before any real-world action.")
