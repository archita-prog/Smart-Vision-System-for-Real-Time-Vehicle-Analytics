#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


# In[3]:


class Tracker:
    def __init__(self):
        self.center_points = {}
        #keep count of IDS
        # each time a new object is detected, the count will increase by one
        self.id_count = 0
    
    def update(self, objects_rect):
        objects_bb_ids = []
        
        #get centre point of new point
        for rect in objects_rect:
            x,y,w,h = rect
            cx = (x+x+w) // 2
            cy = (y+y+h) //2
            
            #find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy -pt[1])
                
                if dist<35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x,y,w,h,id])
                    same_object_detected = True
                    break
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x,y,w,h, self.id_count])
                    self.id_count += 1
                new_center_points = {}
                for obj_bb_id in objects_bbs_ids:
                    _,_,_,_, object_id = obj_bb_id
                    center = self.center_points[object_id]
                    new_center_points[object_id] = center
                
                self.center_points = new_center_points.copy()
                return objects_bbs_ids


# In[2]:


# tracker.py

tracker_id = 0
center_points = {}

def update(boxes):
    global tracker_id, center_points
    new_center_points = {}
    objects_bbs_ids = []

    for rect in boxes:
        x1, y1, x2, y2 = rect
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        matched = False
        for id, pt in center_points.items():
            dist = ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5
            if dist < 25:
                new_center_points[id] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, id])
                matched = True
                break

        if not matched:
            new_center_points[tracker_id] = (cx, cy)
            objects_bbs_ids.append([x1, y1, x2, y2, tracker_id])
            tracker_id += 1

    center_points = new_center_points.copy()
    return objects_bbs_ids



# In[ ]:




