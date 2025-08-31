import cv2
import math
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, ListProperty, BooleanProperty, StringProperty
from kivy.uix.camera import Camera


# ------------ Marker Detection Utilities ------------
MIN_RADIUS = 5

# HSV ranges
BLUE_RANGE  = (np.array([100,150,50]), np.array([130,255,255]))   # Blue (left)
RED_RANGE_1 = (np.array([0,120,70]),   np.array([10,255,255]))    # Red (center, low)
RED_RANGE_2 = (np.array([170,120,70]), np.array([180,255,255]))   # Red (center, high)
GREEN_RANGE = (np.array([40,70,70]),   np.array([80,255,255]))    # Green (right)

def find_marker(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    if r >= MIN_RADIUS:
        return (int(x), int(y))
    return None

def angle_between(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return None
    c = np.dot(v1, v2) / (n1 * n2)
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))

def draw_angle_arc(img, center, base_vec, cur_vec, radius, color_bgr, label):
    if base_vec is None or cur_vec is None:
        return
    base_deg = math.degrees(math.atan2(base_vec[1], base_vec[0]))
    cur_deg  = math.degrees(math.atan2(cur_vec[1],  cur_vec[0]))
    start = base_deg % 360
    end   = cur_deg % 360
    diff  = ((end - start + 540) % 360) - 180
    if diff >= 0:
        ang_start, ang_end = start, start + diff
    else:
        ang_start, ang_end = end, end - diff
    cv2.ellipse(img, center, (radius, radius), 0, ang_start, ang_end, color_bgr, 2)
    mid = (ang_start + ang_end) / 2.0
    mr  = math.radians(mid)
    tx = int(center[0] + radius * math.cos(mr))
    ty = int(center[1] + radius * math.sin(mr))
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)


# ------------ Video Widget (Camera → Numpy → Overlay → Texture) ------------
class Processor(Widget):
    status_text = StringProperty("Tap Select ROI, then Set Baseline.")
    roi_box = ListProperty([0, 0, 0, 0])     # x, y, w, h (frame coords)
    selecting = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Kivy Camera widget (uses Android camera API on device)
        self.camera = Camera(play=True, index=0, resolution=(1280, 720))
        self.add_widget(self.camera)

        # overlay label
        self.label = Label(text=self.status_text, size_hint=(1, None), height="28dp",
                           pos_hint={"x": 0, "y": 0}, color=(1,1,0,1))
        self.add_widget(self.label)

        # baseline vectors (ROI coords)
        self.baseline_left = None
        self.baseline_right = None

        # touch helpers
        self._drag_start_frame = None

        Clock.schedule_interval(self.update, 1/30)

    # Map widget touch → frame pixel coords
    def _widget_to_frame(self, xw, yw, frame_w, frame_h):
        ww, wh = self.width, self.height
        # Camera texture fills the widget; Kivy Camera preserves AR but consider it fullscreen here
        # Approximate uniform scaling
        scale_x = frame_w / ww
        scale_y = frame_h / wh
        xf = int(xw * scale_x)
        yf = int((wh - yw) * scale_y)  # invert Y (widget origin bottom-left)
        xf = np.clip(xf, 0, frame_w-1)
        yf = np.clip(yf, 0, frame_h-1)
        return xf, yf

    def on_touch_down(self, touch):
        if not self.selecting:
            return super().on_touch_down(touch)
        tex = self.camera.texture
        if not tex:
            return True
        fw, fh = tex.size
        xf, yf = self._widget_to_frame(touch.x - self.x, touch.y - self.y, fw, fh)
        self._drag_start_frame = (xf, yf)
        return True

    def on_touch_move(self, touch):
        if not self.selecting or self._drag_start_frame is None:
            return super().on_touch_move(touch)
        tex = self.camera.texture
        if not tex:
            return True
        fw, fh = tex.size
        xf, yf = self._widget_to_frame(touch.x - self.x, touch.y - self.y, fw, fh)
        x0, y0 = self._drag_start_frame
        x = min(x0, xf); y = min(y0, yf)
        w = abs(xf - x0); h = abs(yf - y0)
        self.roi_box = [x, y, w, h]
        self.status_text = f"ROI: {w}x{h}"
        self.label.text = self.status_text
        return True

    def on_touch_up(self, touch):
        if not self.selecting:
            return super().on_touch_up(touch)
        self.selecting = False
        x, y, w, h = self.roi_box
        if w < 20 or h < 20:
            self.roi_box = [0,0,0,0]
            self.status_text = "ROI too small. Try again."
        else:
            self.status_text = "ROI set. Press Set Baseline when flat."
        self.label.text = self.status_text
        self._drag_start_frame = None
        return True

    def set_select_roi(self):
        self.selecting = True
        self.status_text = "Drag to select ROI…"
        self.label.text = self.status_text

    def clear_baseline(self):
        self.baseline_left = None
        self.baseline_right = None
        self.status_text = "Baseline cleared."
        self.label.text = self.status_text

    def set_baseline(self, centers):
        blue, red, green = centers["blue"], centers["red"], centers["green"]
        if not (blue and red and green):
            self.status_text = "Need all 3 markers in ROI for baseline."
            self.label.text = self.status_text
            return
        self.baseline_left  = np.array([blue[0]-red[0],  blue[1]-red[1]])
        self.baseline_right = np.array([green[0]-red[0], green[1]-red[1]])
        self.status_text = "Baseline set."
        self.label.text = self.status_text

    def _process(self, frame_bgr):
        """Process a full BGR frame, draw overlays, return processed BGR frame."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        # If ROI chosen, crop
        x, y, rw, rh = self.roi_box
        if rw > 0 and rh > 0:
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            roi = out[y:y+rh, x:x+rw]
        else:
            roi = out

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue  = find_marker(hsv, *BLUE_RANGE)
        green = find_marker(hsv, *GREEN_RANGE)
        red1  = find_marker(hsv, *RED_RANGE_1)
        red2  = find_marker(hsv, *RED_RANGE_2)
        red   = red1 if red1 else red2

        centers = {"blue": blue, "red": red, "green": green}

        # Draw markers
        if blue:  cv2.circle(roi, blue,  8, (255, 0, 0), -1)
        if red:   cv2.circle(roi, red,   8, (0, 0, 255), -1)
        if green: cv2.circle(roi, green, 8, (0,255, 0), -1)

        # Guide lines
        if red and blue:  cv2.line(roi, red, blue,   (255, 0, 0), 2)
        if red and green: cv2.line(roi, red, green,  (0,255, 0), 2)

        # Baseline angles
        if self.baseline_left is not None and self.baseline_right is not None and red:
            if blue is not None:
                v_now = np.array([blue[0]-red[0], blue[1]-red[1]])
                ang = angle_between(self.baseline_left, v_now)
                if ang is not None:
                    draw_angle_arc(roi, red, tuple(self.baseline_left), tuple(v_now),
                                   radius=45, color_bgr=(255,0,0),
                                   label=f"{ang:.1f}°")
            if green is not None:
                v_now = np.array([green[0]-red[0], green[1]-red[1]])
                ang = angle_between(self.baseline_right, v_now)
                if ang is not None:
                    draw_angle_arc(roi, red, tuple(self.baseline_right), tuple(v_now),
                                   radius=65, color_bgr=(0,255,0),
                                   label=f"{ang:.1f}°")

        # Draw ROI rectangle on full frame
        if (rw > 0 and rh > 0):
            cv2.rectangle(out, (x,y), (x+rw, y+rh), (0,255,255), 2)

        return out, centers

    def update(self, dt):
        # Get camera texture → numpy RGBA
        tex = self.camera.texture
        if not tex:
            return
        fw, fh = tex.size
        # texture pixels are in RGBA, bottom-to-top
        buf = tex.pixels  # bytes
        frame_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fh, fw, 4)
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

        processed, centers = self._process(frame_bgr)

        # If we queued a baseline set, try to set when markers visible
        if getattr(self, "_pending_set_baseline", False):
            if centers["blue"] and centers["red"] and centers["green"]:
                self.set_baseline(centers)
                self._pending_set_baseline = False

        # Show processed
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        tex2 = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt='rgb')
        tex2.blit_buffer(rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        # replace camera texture with processed texture
        self.camera.texture = tex2


class Root(BoxLayout):
    proc = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"

        self.proc = Processor(size_hint=(1, 1))
        self.add_widget(self.proc)

        controls = BoxLayout(size_hint=(1, None), height="56dp", spacing=8, padding=8)
        btn_roi = Button(text="Select ROI")
        btn_baseline = Button(text="Set Baseline")
        btn_clear = Button(text="Clear Baseline")
        btn_quit = Button(text="Quit")

        btn_roi.bind(on_release=lambda *_: self.proc.set_select_roi())
        btn_baseline.bind(on_release=lambda *_: setattr(self.proc, "_pending_set_baseline", True))
        btn_clear.bind(on_release=lambda *_: self.proc.clear_baseline())
        btn_quit.bind(on_release=lambda *_: App.get_running_app().stop())

        controls.add_widget(btn_roi)
        controls.add_widget(btn_baseline)
        controls.add_widget(btn_clear)
        controls.add_widget(btn_quit)
        self.add_widget(controls)


class TwistApp(App):
    def build(self):
        return Root()

if __name__ == "__main__":
    TwistApp().run()
