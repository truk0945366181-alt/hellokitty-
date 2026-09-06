"""
app.py — MVP Prototype
=======================
ระบบแจ้งเตือนคนแปลกหน้าหน้าห้องพัก ด้วยกล้องวงจรปิด

รันด้วยคำสั่ง:  streamlit run app.py
"""

import hashlib
import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import face_engine as fe
import event_log as ev
from alert import send_telegram_alert

st.set_page_config(page_title="Stranger Alert MVP", page_icon="🔔", layout="wide")

# ---------- Helpers ----------

def pil_to_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_image):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


if "bot_token" not in st.session_state:
    st.session_state.bot_token = ""
if "chat_id" not in st.session_state:
    st.session_state.chat_id = ""
if "threshold" not in st.session_state:
    st.session_state.threshold = 75
if "auto_send" not in st.session_state:
    st.session_state.auto_send = True
if "last_alert_hash" not in st.session_state:
    st.session_state.last_alert_hash = None

# ---------- Sidebar: settings ----------

with st.sidebar:
    st.header("⚙️ ตั้งค่าการแจ้งเตือน")
    st.caption(
        "ใช้ Telegram Bot แทน LINE Notify เนื่องจาก LINE Notify ถูกยกเลิกบริการไปแล้วตั้งแต่ 31 มี.ค. 2025"
    )
    st.session_state.bot_token = st.text_input(
        "Telegram Bot Token", value=st.session_state.bot_token, type="password"
    )
    st.session_state.chat_id = st.text_input(
        "Telegram Chat ID", value=st.session_state.chat_id
    )

    if st.session_state.bot_token and st.session_state.chat_id:
        st.success("✅ ตั้งค่า Telegram ครบแล้ว พร้อมส่งแจ้งเตือน")
    else:
        st.error("❌ ยังไม่ได้กรอก Bot Token / Chat ID — ระบบจะยังส่งแจ้งเตือนไม่ได้")

    st.session_state.auto_send = st.checkbox(
        "🔔 ส่งแจ้งเตือนอัตโนมัติทันทีที่พบคนแปลกหน้า",
        value=st.session_state.auto_send,
        help="ถ้าปิดไว้ ระบบจะรอให้กดปุ่มยืนยันเองทุกครั้ง (ไม่เหมาะกับการใช้งานจริงที่ต้องการ real-time)",
    )

    with st.expander("วิธีตั้งค่า Telegram Bot"):
        st.markdown(
            """
1. เปิด Telegram หาบัญชี **BotFather** พิมพ์ `/newbot` ทำตามขั้นตอน จะได้ **Bot Token**
2. ส่งข้อความอะไรก็ได้หาบอทของตัวเอง 1 ครั้ง (เพื่อให้บอทรู้จัก chat)
3. เปิดลิงก์ `https://api.telegram.org/bot<TOKEN>/getUpdates` เพื่อดู **chat id**
4. นำค่าทั้งสองมาใส่ในช่องด้านบน
            """
        )
    st.divider()
    st.session_state.threshold = st.slider(
        "ความเข้มงวดในการจดจำใบหน้า (LBPH threshold)",
        min_value=30,
        max_value=120,
        value=st.session_state.threshold,
        help="ค่ายิ่งต่ำ = เข้มงวดมาก (เสี่ยงแจ้งเตือนคนที่รู้จักผิดว่าเป็นคนแปลกหน้า) / ค่ายิ่งสูง = หลวมขึ้น (เสี่ยงพลาดคนแปลกหน้า)",
    )

st.title("🔔 ระบบแจ้งเตือนคนแปลกหน้าหน้าห้องพัก")
st.caption("Design an MVP — Prototype สำหรับทดสอบ Core Experience เท่านั้น ไม่ใช่ระบบ Production")

tab1, tab2, tab3 = st.tabs(
    ["👤 ลงทะเบียนผู้พักอาศัย (Whitelist)", "📷 ทดสอบตรวจจับ (Run Detection)", "📋 ประวัติการแจ้งเตือน (Event Log)"]
)

# ---------- TAB 1: Whitelist registration ----------
with tab1:
    st.subheader("ลงทะเบียนใบหน้าผู้พักอาศัย")
    st.info(
        "⚠️ ข้อควรระวังด้านความเป็นส่วนตัว: ภาพใบหน้าเป็นข้อมูลอ่อนไหว (biometric data) "
        "ในการใช้งานจริงต้องขอความยินยอมจากเจ้าของภาพก่อนทุกครั้ง และควรมีนโยบายการลบข้อมูลที่ชัดเจน "
        "สำหรับ Prototype นี้ แนะนำให้ใช้ภาพจำลอง/ภาพที่ได้รับความยินยอมจากเพื่อนในทีมเท่านั้น"
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        name_input = st.text_input("ชื่อผู้พักอาศัย (หรือรหัสห้อง เช่น 'ห้อง 104')")
        uploaded_imgs = st.file_uploader(
            "อัปโหลดภาพใบหน้า (แนะนำ 3-5 ภาพต่อคน มุมต่าง ๆ)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if st.button("➕ บันทึกใบหน้าเข้า Whitelist", type="primary"):
            if not name_input:
                st.warning("กรุณาระบุชื่อก่อน")
            elif not uploaded_imgs:
                st.warning("กรุณาอัปโหลดภาพอย่างน้อย 1 ภาพ")
            else:
                success_count = 0
                for uf in uploaded_imgs:
                    img = pil_to_bgr(Image.open(uf))
                    ok, msg = fe.register_face(name_input.strip(), img)
                    if ok:
                        success_count += 1
                    else:
                        st.error(f"{uf.name}: {msg}")
                if success_count:
                    st.success(f"บันทึกสำเร็จ {success_count} ภาพสำหรับ '{name_input.strip()}'")
                    ok, msg, n_people, n_faces = fe.train_model()
                    if ok:
                        st.info(f"🔄 เทรนโมเดลใหม่อัตโนมัติแล้ว ({msg})")
                    else:
                        st.warning(msg)

    with col_b:
        st.markdown("**รายชื่อที่ลงทะเบียนแล้ว**")
        people = fe.list_whitelist_people()
        if not people:
            st.write("_ยังไม่มีใครลงทะเบียน_")
        else:
            for name, count in people.items():
                c1, c2 = st.columns([3, 1])
                c1.write(f"👤 {name} — {count} ภาพ")
                if c2.button("ลบ", key=f"del_{name}"):
                    fe.delete_person(name)
                    st.rerun()

# ---------- TAB 2: Detection ----------
with tab2:
    st.subheader("ทดสอบตรวจจับใบหน้าจากภาพ/กล้อง")

    model_exists = __import__("os").path.exists(fe.MODEL_FILE)
    if not model_exists:
        st.warning("⚠️ ยังไม่มีการเทรนโมเดล — ทุกใบหน้าที่พบจะถูกถือว่าเป็น 'คนแปลกหน้า' ทั้งหมด กรุณาลงทะเบียนคนในแท็บที่ 1 ก่อน")

    source = st.radio("เลือกแหล่งภาพ", ["📁 อัปโหลดภาพ", "📸 ถ่ายภาพจากกล้อง (webcam)"], horizontal=True)

    test_image = None
    if source == "📁 อัปโหลดภาพ":
        uf = st.file_uploader("อัปโหลดภาพจำลองสถานการณ์ (เช่น ภาพคนเดินผ่านหน้าห้อง)", type=["jpg", "jpeg", "png"])
        if uf:
            test_image = pil_to_bgr(Image.open(uf))
    else:
        cam_img = st.camera_input("ถ่ายภาพเพื่อทดสอบ")
        if cam_img:
            test_image = pil_to_bgr(Image.open(cam_img))

    if test_image is not None:
        annotated, results, has_unknown = fe.analyze_image(test_image, st.session_state.threshold)
        image_hash = hashlib.md5(test_image.tobytes()).hexdigest()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(bgr_to_pil(annotated), caption="ผลการตรวจจับ (กรอบเขียว = รู้จัก, กรอบแดง = ไม่รู้จัก)", use_container_width=True)

        with col2:
            st.markdown("**ผลลัพธ์**")
            if not results:
                st.write("ไม่พบใบหน้าในภาพนี้")
            for r in results:
                if r["status"] == "known":
                    st.success(f"✅ รู้จัก: {r['name']} (confidence={r['confidence']:.1f})")
                else:
                    conf_text = f" (confidence={r['confidence']:.1f})" if r["confidence"] is not None else ""
                    st.error(f"🚨 ไม่รู้จัก!{conf_text}")

            if has_unknown:
                st.markdown("---")
                can_send = bool(st.session_state.bot_token and st.session_state.chat_id)
                already_alerted = st.session_state.last_alert_hash == image_hash

                def _do_send():
                    snapshot_path = fe.save_event_snapshot(annotated)
                    message = (
                        "🚨 แจ้งเตือน: พบคนแปลกหน้าบริเวณหน้าห้องพัก\n"
                        f"เวลา: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    sent_ok, send_msg = send_telegram_alert(
                        st.session_state.bot_token, st.session_state.chat_id, message, snapshot_path
                    )
                    for r in results:
                        ev.add_event(
                            status=r["status"],
                            name=r["name"],
                            confidence=r["confidence"],
                            image_path=snapshot_path,
                            alert_sent=sent_ok,
                        )
                    st.session_state.last_alert_hash = image_hash
                    return sent_ok, send_msg

                if not can_send:
                    st.warning(
                        "⚠️ พบคนแปลกหน้า แต่ยังไม่ได้ตั้งค่า Telegram Bot Token / Chat ID "
                        "ในแถบด้านซ้าย จึงยังส่งแจ้งเตือนไม่ได้ (ระบบจะไม่บันทึกเหตุการณ์นี้จนกว่าจะตั้งค่า)"
                    )
                elif st.session_state.auto_send:
                    if already_alerted:
                        st.info("🔔 ส่งแจ้งเตือนสำหรับภาพนี้ไปแล้ว")
                        if st.button("ส่งซ้ำอีกครั้ง"):
                            sent_ok, send_msg = _do_send()
                            (st.success if sent_ok else st.error)(send_msg)
                            st.rerun()
                    else:
                        sent_ok, send_msg = _do_send()
                        if sent_ok:
                            st.success(f"🔔 ส่งแจ้งเตือนอัตโนมัติแล้ว — {send_msg}")
                        else:
                            st.error(
                                f"🚨 พบคนแปลกหน้า แต่ส่งแจ้งเตือนไม่สำเร็จ: {send_msg}\n"
                                "(ตรวจสอบ Bot Token / Chat ID หรือการเชื่อมต่ออินเทอร์เน็ต — "
                                "ระบบยังคงบันทึก log ไว้ตามปกติ)"
                            )
                        st.rerun()
                else:
                    st.markdown("**⚠️ พบคนแปลกหน้า — ส่งแจ้งเตือน?**")
                    if st.button("🔔 ส่งแจ้งเตือนตอนนี้", type="primary"):
                        sent_ok, send_msg = _do_send()
                        if sent_ok:
                            st.success(send_msg)
                        else:
                            st.warning(f"ไม่สามารถส่งแจ้งเตือนได้: {send_msg}\n(ระบบยังคงบันทึก log ไว้ตามปกติ)")
                        st.rerun()
            elif results:
                if st.button("💾 บันทึกลง Log (ไม่ต้องแจ้งเตือน เพราะรู้จักทุกคน)"):
                    snapshot_path = fe.save_event_snapshot(annotated)
                    for r in results:
                        ev.add_event(
                            status=r["status"],
                            name=r["name"],
                            confidence=r["confidence"],
                            image_path=snapshot_path,
                            alert_sent=False,
                        )
                    st.success("บันทึกแล้ว")
                    st.rerun()

# ---------- TAB 3: Event log ----------
with tab3:
    st.subheader("ประวัติการแจ้งเตือนทั้งหมด")
    summary = ev.get_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("เหตุการณ์ทั้งหมด", summary["total"])
    c2.metric("🚨 คนแปลกหน้า", summary["unknown"])
    c3.metric("✅ คนรู้จัก", summary["known"])

    df = ev.get_all_events()
    if df.empty:
        st.write("_ยังไม่มีประวัติเหตุการณ์_")
    else:
        st.dataframe(
            df.rename(
                columns={
                    "timestamp": "เวลา",
                    "status": "สถานะ",
                    "name": "ชื่อ",
                    "confidence": "ค่าความมั่นใจ",
                    "image_path": "ไฟล์ภาพ",
                    "alert_sent": "ส่งแจ้งเตือนแล้ว",
                }
            ),
            use_container_width=True,
        )

        with st.expander("ดูภาพเหตุการณ์ล่าสุด 5 รายการ"):
            for _, row in df.head(5).iterrows():
                try:
                    st.image(row["image_path"], caption=f"{row['timestamp']} — {row['status']}", width=300)
                except Exception:
                    pass
