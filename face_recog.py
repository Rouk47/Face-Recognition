import face_recognition as face
import numpy as np
import cv2
import os
import pandas as pd

# โหลดรูปภาพอ้างอิงและเข้ารหัสใบหน้า
known_face_encodings = []
known_face_names = []

# กำหนดเส้นทางไปยังโฟลเดอร์ที่เก็บรูปภาพ
directories = ["pita/", "sorayuth/"]

# วนลูปผ่านโฟลเดอร์และไฟล์ภาพ
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # โหลดไฟล์รูปภาพ
            image_path = os.path.join(directory, filename)
            person_name = os.path.splitext(filename)[0]
            print("กำลังโหลดรูปภาพ:", person_name)
            image = face.load_image_file(image_path)

            # เข้ารหัสใบหน้า
            face_encoding = face.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(directory)

# โหลดไฟล์วิดีโอหรือกล้องเว็บแคมเพื่อดำเนินการวิดีโอสตรีม (สามารถเปลี่ยนเป็น 0 เพื่อใช้กล้องเว็บแคม)
video_capture = cv2.VideoCapture("pita.mp4")

# กำหนดตัวแปรต่าง ๆ เพื่อใช้เก็บข้อมูลใบหน้าที่ตรวจพบในแต่ละเฟรม
face_locations = []
face_encodings = []
face_names = []
face_percent = []
process_this_frame = True  # ตัวแปรที่ใช้สลับการประมวลผลเฟรมเพื่อเพิ่มประสิทธิภาพ

# สร้าง List ของ dictionaries เพื่อเก็บข้อมูลใบหน้าที่ตรวจพบ
detected_faces = []

# Main loop to process each frame of the video
while True:
    ret, frame = video_capture.read()  # อ่านเฟรมจากวิดีโอ
    if ret:
        # Reduce the frame size to increase FPS
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # ลดขนาดเฟรมเพื่อเพิ่มประสิทธิภาพในการประมวลผล
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # แปลงรูปแบบสีจาก BGR เป็น RGB

        # Process every other frame to improve performance
        if process_this_frame:
            face_locations = face.face_locations(rgb_small_frame, model="hog")  # ตรวจจับใบหน้าในเฟรมปัจจุบันโดยใช้ HOG
            face_encodings = face.face_encodings(rgb_small_frame, face_locations)  # เข้ารหัสใบหน้าที่ตรวจพบ

            face_names = []  # เก็บชื่อของโฟลเดอร์ที่ตรงกับใบหน้าที่ระบุ
            face_percent = []  # เก็บเปอร์เซ็นต์ความเหมือนต่อหน้าของแต่ละบุคคล

            for face_encoding in face_encodings:
                face_distances = face.face_distance(known_face_encodings, face_encoding)  # คำนวณระยะห่างระหว่างใบหน้าในเฟรมปัจจุบันกับใบหน้าอ้างอิง
                best_match_index = np.argmin(face_distances)  # หาดัชนีของใบหน้าที่มีระยะห่างน้อยที่สุด
                face_percent_value = 1 - face_distances[best_match_index]  # คำนวณเปอร์เซ็นต์ความเหมือน

                # กำหนดชื่อและเปอร์เซ็นต์ความเหมือนต่อหน้าของแต่ละบุคคล
                if face_percent_value >= 0.5:
                    # หาชื่อที่มีรูปภาพที่ตรงกับใบหน้าที่ระบุ
                    name = os.path.basename(os.path.normpath(known_face_names[best_match_index]))
                    percent = round(face_percent_value * 100, 2)
                else:
                    name = "UNKNOWN"
                    percent = 0

                face_names.append(name)
                face_percent.append(percent)

                # เพิ่มข้อมูลใบหน้าที่ตรวจพบลงใน List ของ dictionaries
                detected_faces.append({'Name': name, 'Match_Percent': percent})

        process_this_frame = not process_this_frame  # สลับการประมวลผลเฟรมเพื่อเพิ่มประสิทธิภาพ

        # Display the results
        for (top, right, bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            top *= 4  # ขยายขนาดกรอบบนตามขนาดเฟรมต้นฉบับ
            right *= 4  # ขยายขนาดกรอบขวาตามขนาดเฟรมต้นฉบับ
            bottom *= 4  # ขยายขนาดกรอบล่างตามขนาดเฟรมต้นฉบับ
            left *= 4  # ขยายขนาดกรอบซ้ายตามขนาดเฟรมต้นฉบับ

            if name == "UNKNOWN":
                color = (0, 0, 255)  # สีแดงสำหรับใบหน้าที่ไม่รู้จัก
            else:
                color = (0, 255, 0)  # สีเขียวสำหรับใบหน้าที่รู้จัก

            # วาดกรอบและข้อความบนเฟรมวิดีโอ
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, f"Match: {percent}%", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Video", frame)  # แสดงผลลัพธ์บนเฟรมวิดีโอ
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # หยุดการทำงานเมื่อผู้ใช้กด 'q'

    else:
        break

video_capture.release()  # คืนทรัพยากรวิดีโอ
cv2.destroyAllWindows()  # ปิดหน้าต่างทุกประการเมื่อเสร็จสิ้น

# สร้าง DataFrame จาก List ของ dictionaries ที่เก็บข้อมูลใบหน้าที่ตรวจพบ
df = pd.DataFrame(detected_faces)

# บันทึก DataFrame เป็นไฟล์ Excel
excel_file = 'detected_faces.xlsx'
df.to_excel(excel_file, index=False)
