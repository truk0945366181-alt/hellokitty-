"""
alert.py
========
โมดูลสำหรับส่งการแจ้งเตือนไปยังแอดมิน/เจ้าของหอ

ใช้ Telegram Bot API เป็นช่องทางหลัก เพราะสมัครใช้งานง่าย ฟรี และไม่มีข้อจำกัดทางธุรกิจ
(หมายเหตุ: เดิมทีตั้งใจใช้ LINE Notify แต่บริการนี้ถูกยกเลิกไปแล้วตั้งแต่ 31 มี.ค. 2025
ผู้ที่ต้องการใช้ LINE แทน ต้องเปลี่ยนไปใช้ LINE Messaging API ผ่าน LINE Official Account
ซึ่งซับซ้อนกว่า Telegram Bot พอสมควร)

วิธีตั้งค่า Telegram Bot (ทำครั้งเดียว):
1. เปิดแอป Telegram ค้นหา "BotFather" แล้วพิมพ์ /newbot ทำตามขั้นตอน จะได้ Bot Token
2. เพิ่มบอทเข้าไปคุยด้วย (หรือเข้ากลุ่ม) แล้วส่งข้อความอะไรก็ได้ 1 ครั้ง
3. เปิด https://api.telegram.org/bot<TOKEN>/getUpdates เพื่อดู chat_id ของตัวเอง
4. นำ Token และ Chat ID มาใส่ในหน้าตั้งค่าของแอปนี้
"""

import requests


def send_telegram_alert(bot_token, chat_id, message, image_path=None):
    """
    ส่งข้อความแจ้งเตือน (พร้อมภาพถ้ามี) ไปยัง Telegram
    คืนค่า (สำเร็จหรือไม่, ข้อความผลลัพธ์)
    """
    if not bot_token or not chat_id:
        return False, "ยังไม่ได้ตั้งค่า Bot Token หรือ Chat ID"

    try:
        if image_path:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, "rb") as img_file:
                resp = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": message},
                    files={"photo": img_file},
                    timeout=10,
                )
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp = requests.post(
                url, data={"chat_id": chat_id, "text": message}, timeout=10
            )

        if resp.status_code == 200:
            return True, "ส่งแจ้งเตือนสำเร็จ"
        return False, f"ส่งไม่สำเร็จ (HTTP {resp.status_code}): {resp.text[:200]}"
    except requests.exceptions.RequestException as e:
        return False, f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}"
