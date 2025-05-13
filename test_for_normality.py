#!/usr/bin/env python3

# quick script to parse and plot the dx/dy data
# turn this into something that can do a statistical test for normality

import re
import matplotlib.pyplot

pattern = re.compile('.*Object_([0-9]+).*cX= *([0-9]+).*cY= *([0-9]+).*Frame= *([0-9]+)')

objects = {}

with open('DeadObjectXYs.txt') as data:
    for line in data:
        m = pattern.match(line)
        assert m
        object_id, x, y, frame = map(int, m.groups())

        if object_id not in objects:
            objects[object_id] = []

        objects[object_id].append((frame, x, y))

all_dx = []
all_dy = []
for object_id in objects:
    objects[object_id].sort()

    frame, x, y = objects[object_id][0]
    for next_frame, next_x, next_y in objects[object_id][1:]:
        dx = (next_x - x) / (next_frame - frame)
        dy = (next_y - y) / (next_frame - frame)
        all_dx.append(dx)
        all_dy.append(dy)


matplotlib.pyplot.hist(all_dx, 100)

matplotlib.pyplot.show()

matplotlib.pyplot.hist(all_dy, 100)

matplotlib.pyplot.show()
