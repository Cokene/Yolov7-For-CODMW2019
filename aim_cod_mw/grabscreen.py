# _*_ coding : utf-8 _*_
# @Time : 2022/8/10 21:15
# @Author : shliz
# @File : grabscreen
# @Project : Yolov7ForCODMW2019

import sys
import cv2
import mss
import numpy as np
import win32gui, win32ui, win32con, win32api


def getWindowData():
    windowName = "Wallpaper UI"
    windowNameDesktop = "Wallpaper UI"
    hwnd = win32gui.FindWindow(None, windowName)  # find the window handle by the window name
    hwnd_desktop = win32gui.FindWindow(None, windowNameDesktop)

    if hwnd == 0 and hwnd_desktop == 0:
        print("cannot find the right windows surface,process will be exited!")
        sys.exit(0)
    elif hwnd != 0:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)  # gets the location data of the window
    elif hwnd_desktop != 0:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd_desktop)  # gets the location data of the window

    width = right - left
    height = bottom - top
    return left, top, right, bottom, width, height

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()
    region = getWindowData()
    if region:
        left, top, x2, y2 = region[:4]
        width = x2 - left
        height = y2 - top
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    # or cv2.COLOR_BRGA2BGR
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def mss_grab():
    region = getWindowData()
    monitor = {
        'left': region[0],
        'top': region[1],
        'width': region[4],
        'height': region[5]
    }

    with mss.mss() as m:
        img = m.grab(monitor)
    img = np.asarray(img)

    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


if __name__ == '__main__':
    import time
    t0 = time.time()
    while True:
        img = mss_grab()
        print(np.asarray(img).shape, f' {(time.time() - t0) * 1E3:.1f}ms {1 / (time.time() - t0 + 1E-2):.1f}fps')
        t0 = time.time()
        cv2.imshow('0', img)
        cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):  # press 'q' to quit
            cv2.destroyAllWindows()
            raise StopIteration