from mitsuba.core import *
from math import *

def calc_min_dist_segment_segment(p11, p12, p21, p22):
    # Direction vector of the line segment 1 and 2
    v = [p12 - p11, p22 - p21] # in mitsuba, type(Point3 - Point3) = Vector3
    
    # Vector from the base point to the end point of the another segment
    vp = [[p21 - p11, p22 - p11], [p11 - p21, p12 - p21]]

    # The cross product value of v and vp
    cross_to_vp = [[cross(v[0], vp[0][0]), cross(v[0], vp[0][1])], [cross(v[1], vp[1][0]), cross(v[1], vp[1][1])]]

    # Calculate the minimum distance
    segments = [[p11, p12], [p21, p22]]
    min_dist = float('inf')

    for s in range(2):
        dot_vv = dot(v[s], v[s])
        dist = 0
        is_inner = True

        for i in range(2):
            # Calculate the parameter of the target point
            t = dot(vp[s][i], v[s]) / dot_vv
            if (t >= 0 and t <= 1):
                # Inner the line segment
                dist = sqrt(absDot(cross_to_vp[s][i], cross_to_vp[s][i]) / dot_vv)
            elif (t < 0):
                # Outer the line segment and closer to the base point
                temp_v = segments[(s+1)%2][i] - segments[s][0]
                dist = sqrt(absDot(temp_v, temp_v)) 
                is_inner = False
            else:
                # Outer the line segment and closer to the end point
                temp_v = segments[(s+1)%2][i] - segments[s][1]
                dist = sqrt(absDot(temp_v, temp_v))  
                is_inner = False
                
            min_dist = dist if (dist < min_dist) else min_dist
            
        # If both points are inside the line segment, it's completed!
        if is_inner == True:
            break

    return min_dist

def calc_min_dist_segment_point(p11, p12, p3):
    # Direction vector of the line segment 1
    v = p12 - p11 # in mitsuba, type(Point3 - Point3) = Vector3
    
    # Vector from the base point to the end point of the another segment
    vp = p3 - p11

    # The cross product value of v and vp
    cross_to_vp = cross(v, vp)

    # Calculate the minimum distance
    min_dist = float('inf')
    
    dot_vv = dot(v, v)
    dist = 0
    # Calculate the parameter of the target point
    t = dot(vp, v) / dot_vv
    if (t >= 0 and t <= 1):
        # Inner the line segment
        dist = sqrt(absDot(cross_to_vp, cross_to_vp) / dot_vv)
    elif (t < 0):
        # Outer the line segment and closer to the base point
        temp_v = p3 - p11
        dist = sqrt(absDot(temp_v, temp_v)) 
    else:
        # Outer the line segment and closer to the end point
        temp_v = p3 - p12
        dist = sqrt(absDot(temp_v, temp_v))  

    return min_dist

def calc_min_dist_point_point(p1, p2):
    v = p2 - p1 # in mitsuba, type(Point3 - Point3) = Vector3
    min_dist = sqrt(dot(v, v))
    return min_dist
