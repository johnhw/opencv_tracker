import cv2
import numpy as np
import sys
import os
import sqlite3
import scipy.optimize
#import sortedcontainers

key_map = {2424832:"left", 2555904:"right", 2490368:"up", 2621440:"down", 8:"bksp", 2162688:"pageup", 2228224:"pagedown"}        

def create_tables(cursor):
    table_exists = cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='videos'").fetchone()[0]
    if not table_exists:                
        cursor.execute("CREATE TABLE tracks (id INTEGER PRIMARY KEY, start_t REAL, end_t REAL, video INT, FOREIGN KEY video REFERENCES videos(id))")
        cursor.execute("CREATE TABLE track_log (id INTEGER PRIMARY KEY, x1 REAL, y1 REAL, x2 REAL, y2 REAL, t REAL, track INT, FOREIGN KEY track REFERENCES tracks(id))")
        cursor.execute("CREATE TABLE tag_track (id INTEGER PRIMARY KEY, tag INT, track INT, FOREIGN KEY tag REFERENCES tags(id), FOREIGN KEY track REFERENCES tracks(id))")
        cursor.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("CREATE TABLE videos (id INTEGER PRIMARY KEY, path TEXT, name TEXT, duration REAL)")
        cursor.execute("CREATE INDEX track_log_id ON track_log(track)")    
    
class Track(object):
    def __init__(self, t, rect):
        self.start_t = start_t
        self.t = start_t
        self.last_t = start_t
        self.track = [(t,rect)]
        self.tags = set({})
        
    def append_track(self, t, rect):
        self.track.add((t,rect))
        self.last_t = max(self.last_t, t)
        
    def write(self, cursor):
        # insert the track metadata        
        id = cursor.execute("SELECT id FROM video WHERE name=?", (self.video_name,)).fetchone()        
        cursor.execute("INSERT INTO tracks(start_t, end_t, video) VALUES (?,?,?)", (self.start_t, self.end_t, id[0]))
        track = cursor.lastrowid       
        # insert the actual track data
        for t,r in self.track:
            cursor.execute("INSERT INTO track_log(t, x1, y1, x2, y2, track) VALUES (?,?,?,?,?,?)", (t, r[0], r[1], r[2], r[3], track))
            
        for tag in self.tags:
            # get the id of the tag entry
            id = cursor.execute("SELECT id FROM tag WHERE name=?", (tag,)).fetchone()
            if id is None:
                cursor.execute("INSERT INTO tag(name) VALUES (?)", (tag,))
                id = cursor.lastrowid    
            else:
                id = id[0]
            # associate with this track
            cursor.execute("INSERT INTO tag_track(tag, track) VALUES (?,?,?)", (id, track))
    
        
        
class TaggedRectangle(object):
    def __init__(self, rect=(0,0,0,0)):
        self.rect = rect
        self.tags = set({})
        self.tag_text = None
        self.is_mask = False
        self.tag_color = (0,0,0)
        self.locked = False
        self.color = (0,255,0)
        self.off_x, self.off_y = 0,0
        self.scale_x, self.scale_y = 1, 1
        
    def prompt_tag(self):        
        self.tag_color = (0,0,255)
        self.tag_text="?"
        
    def add_tag(self, tag):
        if tag in self.tags:
            self.tags.remove(tag)
        else:
            self.tags.add(tag)
        self.tag_color = (0,255,0)
        self.tag_text = ":".join(self.tags)
            
    def translate(self, x, y):
        self.pre_translate(x,y)
        self.commit()
        
    def commit(self):
        self.rect = self.preview() 
        self.off_x, self.off_y = 0,0
        self.scale_x, self.scale_y = 1, 1
        
    def pre_scale(self, sx, sy):
        self.scale_x = sx
        self.scale_y = sy     
        
    def pre_translate(self, x, y):
        self.off_x = x
        self.off_y = y
                
    def inside(self, x, y):
        return rect_inside(x,y,self.rect[0], self.rect[1], self.rect[2], self.rect[3])
        
    
    def preview(self):
        a,b,c,d = self.rect
        w = (c-a) * self.scale_y
        h = (d-b) * self.scale_x
        cx = (c+a)/2.0
        cy = (b+d)/2.0        
        a1 = cx - w/2
        b1 = cy - h/2
        c1 = cx + w/2
        d1 = cy + h/2        
        return (a1+self.off_x, b1+self.off_y, c1+self.off_x, d1+self.off_y)
    
    def draw(self, img):        
        
        c = np.array(self.color)
        if self.locked:
            c = np.array((0,0,255))
        view_rect = self.preview() 
        
        if self.tag_text is not None:
            
            draw_rect(view_rect, img, c/20., thick=-1)
            w,h,d = img.shape        
            x, y = int(view_rect[0]), int(view_rect[1])
            cv2.putText(img, self.tag_text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.tag_color)
        draw_rect(view_rect, img, c, thick=1)
        
    def set_rect(self, a, b, c, d):
        self.rect = (min(a,c),min(b,d),max(a,c),max(b,d))
        
        
# handle rectangle drawing
class RectHandler(object):
    def __init__(self):
        self.dragging = False
        self.finished = True
        self.moving = []
        self.current_rect = TaggedRectangle()
        self.rect_stack = []
        self.ix = 0
        self.iy = 0
        self.sx = 1
        self.sy = 1
        self.tagging = False
        
    def start(self, x, y):
        if self.tagging:
            return 
        self.current_rect = TaggedRectangle()      
        self.moving = [r for r in self.rect_stack if r.inside(x,y)]                
        self.ix = x
        self.iy = y  
        self.sx = 1
        self.sy = 1
        self.dragging = True
        self.finished = False
        
    def end(self, x, y):        
        if self.tagging:
            return 
            
        self.dragging = False
        if self.moving:
            for rect in self.moving:
                rect.commit()
                rect.locked = False
            self.finished = True
        else:
            self.current_rect.prompt_tag()
            self.tagging = True
        
    def kill(self, x, y):
        kill_list = [r for r in self.rect_stack if r.inside(x,y)]
        for k in kill_list:
            self.rect_stack.remove(r)
            
    def abort(self):
        self.dragging = False
        self.finished = True
        self.moving = []
        self.current_rect = TaggedRectangle()
    
    def draw_mask(self, img):
        for r in self.rect_stack:
            if r.is_mask:
                draw_rect(r.rect, img, (0,0,0), -1)
    
    def add_tag(self, v):
        if self.dragging and self.moving:
            for rect in self.moving:
                rect.add_tag(v)
        else:
            self.current_rect.add_tag(v)                
            if v=='<mask>':
                self.current_rect.is_mask = True
                self.current_rect.locked = True
                        
    def set_tags(self):
        if len(self.current_rect.tags)>0:
            self.rect_stack.append(self.current_rect)
        self.current_rect = TaggedRectangle()                  
        self.finished = True
        self.tagging = False
        
    def toggle_lock(self, x, y):
        toggle_list = [r for r in self.rect_stack if r.inside(x,y)]
        if len(toggle_list)>0:
            locked = toggle_list[0].locked            
            for t in toggle_list:
                t.locked = not locked                
        
    def scale(self, s):
        if self.tagging:
            return 
            
        self.sx *= s
        self.sy *= s
        for rect in self.moving:
            rect.pre_scale(self.sx,self.sy)
        
    
    def move(self, x, y):
        if self.tagging:
            return 
            
        if self.dragging:
            if self.moving:
                for rect in self.moving:
                    rect.pre_translate(x-self.ix, y-self.iy)
            else:
                # create a new rectangle
                self.current_rect.set_rect(self.ix, self.iy, x, y)

def rect_inside(x,y,a,b,c,d):
    return a < x < c and b < y < d
            
def handle_rect(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.start(x,y)
        
    if event == cv2.EVENT_RBUTTONUP:
        param.kill(x,y)
    
    if event == cv2.EVENT_MBUTTONUP:
        param.toggle_lock(x,y)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        param.move(x,y)
        
    elif event == cv2.EVENT_LBUTTONUP:
        param.end(x,y)
        
class ShiftModel(object):
    def __init__(self):
        pass
        
    def transform(self, par, old):
        x,y,s = par
        
        est_new = old * s + np.array([x,y])
        return est_new
    
    def error(self, par, old, new):        
        
        est_new = self.transform(par, old)        
        return np.sum((new-est_new)**2)
         
        
# compute ransac for a straight line
def ransac(x,y,tol=0.2,k=3, N=20):
    best_model = [0,0,0]
    ix = np.arange(len(x))
    fixed_ix = np.arange(len(x))
    s = ShiftModel()
    for i in range(N):
        # randomly choose k matched points from x and y
        np.random.shuffle(ix)
        x_pts = x[ix[0:k]]
        y_pts = y[ix[0:k]]                
        res = scipy.optimize.minimize(s.error, (0,0,1), args=(x_pts, y_pts))
        # compute the inliers
        inlier_distances = [(i,s.error(res.x, xf, yf)) for i,xf,yf in zip(fixed_ix,x,y)]                
        inliers = [d[0] for d in inlier_distances if d[1]<tol]
        # store the best model
        if len(inliers)>best_model[0]:
            best_model[0] = len(inliers)
            best_model[1] = res.x
    return best_model[0], best_model[1]
        
class LKTracker(object):
    def __init__(self):
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 200,
                               qualityLevel = 0.1,
                               minDistance = 9,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (25,25),
                          maxLevel = 8,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.first_frame = True
        self.color=(100,0,0)
        
    def reset(self, old, new):
        self.p0 = cv2.goodFeaturesToTrack(old, mask = None, **self.feature_params)
        self.first_frame = False
        
    def refresh(self):
        self.first_frame = True
        
    def update(self, old, new):        
        # set initial points
        if self.first_frame:
            self.reset(old, new)
            
        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, self.p0, None, **self.lk_params)
        if p1!=None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
            self.p0 = good_new.reshape(-1,1,2)
            return good_old, good_new
        else:
            return [], []
            
class SIFTTracker(object):
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher()

        self.first_frame = True
        self.color=(0,100,0)
        
    def reset(self, old, new):        
        self.first_frame = False
        
    def refresh(self):
        self.first_frame = True
        
    def update(self, old, new):        
        # set initial points
        if self.first_frame:
            self.reset(old, new)
                    
        old_kp, old_des = self.sift.detectAndCompute(old,None)            
        new_kp, new_des = self.sift.detectAndCompute(new,None)                   
        
        matches = self.matcher.knnMatch(old_des, new_des, k=2)
        
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
            
        old = []
        new = []
        for match in good:
            m = match[0]            
            old.append(np.array(old_kp[m.queryIdx].pt))
            new.append(np.array(new_kp[m.trainIdx].pt))        
        return old, new
            
        
def signed_pc(x,n):
    ixs = np.argsort(np.abs(x))
    ix = len(x) * n
    return x[ixs[int(ix)]]

def draw_rect(r, img, color, thick=1):
    """Draw rectangle r on img in the given color and thickness"""
    a,b,c,d = r
    cv2.rectangle(img,(int(a),int(b)),(int(c),int(d)),color,thick)

    
def update_rect(r, updates):
    if len(updates)<1 or r.locked:
        return
    old, new = updates[0]
    off_x, off_y = [], []
    old_inside, new_inside = [], []
    for i,(new,old) in enumerate(zip(new, old)):
        x1,y1 = new.ravel()
        x2,y2 = old.ravel()          
        if r.inside(x2,y2):        
            off_x.append(x1-x2)
            off_y.append(y1-y2)
            old_inside.append(old)
            new_inside.append(new)
            
    if len(off_x)>0:        
        m, res = ransac(np.array(old_inside), np.array(new_inside))
        if m>2:
            x = res[0]
            y = res[1]            
            r.translate(x,y)        

        
def frame_skip(video, offset_seconds):
    pos = video.get(cv2.CAP_PROP_POS_MSEC)
    video.set(cv2.CAP_PROP_POS_MSEC, pos + offset_seconds * 1000)
    
        
def track_video(fname, tags):
    cap = cv2.VideoCapture(fname)

    rect_handler = RectHandler()        
    cv2.namedWindow('image')       
    cv2.setMouseCallback('image',handle_rect,param=rect_handler)
    fade = 0.9
    trackers = [SIFTTracker()]
    old_gray = None
    mask = None
    pause = False
    

    def print_tags(img, tags):
        x = 5
        y = 20
        for k,v in tags.iteritems():
            cv2.putText(img, k, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            cv2.putText(img, v, (x+20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,48,0))
            y += 15
            

    while True:
        if (not pause and rect_handler.finished) or mask is None:
            show_pause = False
            ret,frame = cap.read()    
            rect_handler.draw_mask(frame)
            
            h, w, d = frame.shape
            
            frame = cv2.resize(frame, (640, int(640 * (h/float(w)))), interpolation=cv2.INTER_AREA)
            if frame is None:
                # stop at end of video
                    break

            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
            
            if mask is None:
                # create the mask for drawing the tracks
                mask = np.zeros_like(frame)
                rect_img = np.zeros_like(frame)
                tag_img = np.zeros_like(frame)
                print_tags(tag_img, tags)

            updates = []    
            
            # track
            if old_gray is not None:
                for tracker in trackers:
                    new, old = tracker.update(old_gray, new_gray)                                
                    # draw the tracks            
                    for p1, p2 in zip(old, new):                    
                        x1,y1 = p1.ravel()
                        x2,y2 = p2.ravel()
                        cv2.line(mask, (int(x1),int(y1)),(int(x2),int(y2)), tracker.color, 2)
                        cv2.circle(frame,(int(x1),int(y1)),5, tracker.color,-1)

                    # store the shift for the whole image
                    updates.append((new,old))

            for r in rect_handler.rect_stack:            
                update_rect(r, updates)
                
            # Now update the previous frame 
            old_gray = new_gray.copy()    
            mask = mask * fade
        else:
            show_pause = True
            

        # add the mask on to the image    
        for r in rect_handler.rect_stack:            
            # draw each active rectangle
            r.draw(rect_img)
            
            

        rect_handler.current_rect.draw(rect_img)
        
        img = cv2.add(frame,mask.astype(np.uint8))    
        img = cv2.add(img,rect_img)
        img = cv2.add(img,tag_img)
        rect_img *= 0
        
        if show_pause:
            draw_rect([20,20,40,100], img, (255,255,255), thick=-1)
            draw_rect([60,20,80,100], img, (255,255,255), thick=-1)
        
        
        
       
        
        # show the image and check for key activity
        cv2.imshow('image',img)
        
        
        k = cv2.waitKey(20) 
        
        if k in key_map:
            key = key_map[k]
            if key=='left':
                frame_skip(cap, -5)
            if key=='right':
                frame_skip(cap, 5)
                
            
        if k & 0xff == 27:
            rect_handler.abort()
        
        key_chr = chr(k&0xff)
        
        if key_chr=='+':
            rect_handler.scale(1.08)
        
        if key_chr=='-':
            rect_handler.scale(0.92)
        
       
        if key_chr==' ':
            pause = not pause
        if key_chr=='r':
            for t in trackers:
                t.refresh()
            
        if key_chr in tags:        
            rect_handler.add_tag(tags[key_chr])
            
        if key_chr=='\r':        
            rect_handler.set_tags()
        
    cv2.destroyAllWindows()
    cap.release()

tags = {"p":"pedestrian", "w":"wall"}    
track_video("videos/walking.avi", tags = {"p":"Pedestrian", "w":"Wall"})


