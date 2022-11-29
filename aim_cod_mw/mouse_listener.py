from threading import Thread
from pynput.mouse import Listener, Button


# 监听鼠标移动事件
def on_move(x, y):
    # print('鼠标指针移动到的位置 {0}'.format((x, y)))
    pass

# 监听鼠标按键
def on_click(x, y, button, pressed):
    if button == Button.left:
        print('{0}位置{1}'.format('鼠标左键按下' if pressed else '鼠标左键松开', (x, y)))
    elif button == Button.right:
        print('{0}位置{1}'.format('鼠标右键按下' if pressed else '鼠标右键松开', (x, y)))
    elif button == Button.middle:  # 停止监听
        return False

# 滑轮滚动事件
def on_scroll(x, y, dx, dy):
    print(f'Scrolled {x, y}, {dx, dy}')

class Mouse:
    def __init__(self):
        self.lock_mode = False

    def click(self, x, y, button, pressed):
        if pressed and button == button.x1:  # Side mouse button
            self.lock_mode = not self.lock_mode  # negation
            print('lock mode:', 'on' if self.lock_mode else 'off')

def listener(mouse):
    while True:
        # Collect events until released
        with Listener(on_click=mouse.click) as listener:
            listener.join()

import time
mouse = Mouse()
t = Thread(target=listener, args=[mouse], daemon=True)
t.start()
while True:
    if mouse.lock_mode:
        print('success')
    else:
        print('fail')
    time.sleep(1)