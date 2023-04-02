# import qrcode
import cv2
from pathlib import Path
import pyzbar.pyzbar as pyzbar # 바코드 인식시 pip install pyzbar입력
import os
import sys
import json
from PIL import Image
import numpy as np
from pyzbar.pyzbar import ZBarSymbol # waring 경고 해결하기 위함


################################################################################################################################ QR 코드 생성
# def make_qrcode(size , border ,error=qrcode.constants.ERROR_CORRECT_L,  **kwards):
#     qr = qrcode.QRCode(version = 1, box_size = size , border = border , error_correction = error) # qrcode 객체 생성 및 size 설정 , 여백 설정(border)
#     data_dict = {}
#     for (key , value) in kwards.items():
#         data_dict[key] = value
#     data_str = json.dumps(data_dict)
#     qr.add_data(data_str) # qr 코드 데이터 생성
#     qr.make(fit = True)  # qr코드 만들기 
#     qrcode_img = qr.make_image() # qr코드를 PIL이미지 형태로 생성
#     save_img(qrcode_img)
#     return qrcode_img

def save_img(img , path = Path(__file__).parent , file_name = 'qr.jpg'): # PIL 이미지 저장
    img.save(str(path / file_name))

################################################################################################################################ QR 코드 detect
def detect_qr(img):
    qr_codes = pyzbar.decode(img ,symbols=[ZBarSymbol.QRCODE]) # 단순히 qr코드만 인식하는경우 (이미지에 PDF417바코드 없는경우 발생하는 경고 해결)
    if qr_codes is not None:
        data_list = []
        rect = []
        idx = 0
        for qr_code in qr_codes:
            data_str = qr_code.data.decode('utf-8')
            data_dict = json.loads(data_str)
            data_dict['idx'] = idx
            idx +=1
            data_list.append(data_dict)
            x,y,w,h = qr_code.rect
            rect.append((x,y,w,h))
    return data_list , rect

################################################################################################################################ QR 코드 detect 동영상
def detect_qr_video(img):
    try:
        qr_codes = pyzbar.decode(img , symbols=[ZBarSymbol.QRCODE]) # 단순히 qr코드만 인식하는경우 (이미지에 PDF417바코드 없는경우 발생하는 경고 해결)
        data_list = []
        idx = 0
        for qr_code in qr_codes:
            data_str = qr_code.data.decode('utf-8')
            data_dict = json.loads(data_str)
            data_dict['idx'] = idx
            idx +=1
            x,y,w,h = qr_code.rect
            data_list.append(data_dict)
            cv2.rectangle(img , (x ,y) , (x+w , y+h) , (255,0,0) , 2)
            print(f'idx = {idx} , weight = {data_dict["weight"]} ,  height = {data_dict["height"]}')
        return img ,data_list 
    except Exception as e:
        print(e)
################################################################################################################################ QR 코드 detect 동영상    
def detect_qr(img):
    qr_codes = pyzbar.decode(img ,symbols=[ZBarSymbol.QRCODE]) # 단순히 qr코드만 인식하는경우 (이미지에 PDF417바코드 없는경우 발생하는 경고 해결)
    if qr_codes is not None:
        data_list = []
        rect = []
        idx = 0
        for qr_code in qr_codes:
            data_str = qr_code.data.decode('utf-8')
            data_dict = json.loads(data_str)
            data_dict['idx'] = idx
            idx +=1
            data_list.append(data_dict)
            x,y,w,h = qr_code.rect
            rect.append((x,y,w,h))
    return data_list , rect   







