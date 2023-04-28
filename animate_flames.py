import os
import numpy as np
import xml.etree.ElementTree as ET
import math
import copy


def shift_from_origin(x, o):
    return x[0] - o[0], x[1] - o[1]


def shift_to_origin(x, o):
    return x[0] + o[0], x[1] + o[1]


def get_vector_length(x, o):
    return math.sqrt(((x[0] - o[0])**2) + ((x[1] - o[1])**2))


def get_angle(a, r):
    return math.degrees(np.arccos(a/r))


def get_point_from_deg(r, deg):
    a = r * np.cos(math.radians(deg))
    b = r * np.sin(math.radians(deg))

    return a, b


def rotate_x_y(x, y, o, x_angles):
    x1, x2 = shift_from_origin(x, o)
    y1, y2 = shift_from_origin(y, o)

    rx = get_vector_length(x, o)
    ry = get_vector_length(y, o)

    x1p, x2p = shift_to_origin(get_point_from_deg(rx, x_angles[0]), o)
    y1p, y2p = shift_to_origin(get_point_from_deg(ry, x_angles[1]), o)

    return (x1p, x2p), (y1p, y2p)


def get_coefs_from_string(s):
    vals = [float(c) for c in s.split(' ')]
    return (vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])


def get_string_from_coefs(x, y, o):
    return str(x[0]) + ' ' + str(x[1]) + ' ' + str(y[0]) + ' ' + str(y[1]) + ' ' + str(o[0]) + ' ' + str(o[1])


def set_rotation_xml(xtree, x_angles, label, xform_ind=0):
    xt = copy.deepcopy(xtree)
    xt.find('flame').attrib['name'] = label
    xforms = xt.findall('.//xform')

    for i, xf in enumerate(xforms):
        x, y, o = get_coefs_from_string(xf.attrib['coefs'])
        xp, yp = rotate_x_y(x, y, o, x_angles[i])

        xf.attrib['coefs'] = get_string_from_coefs(xp, yp, o)

    return xt


def get_xform_angles(xt):
    xforms = xt.findall('.//xform')
    x_angles = np.zeros((len(xforms), 2))

    for i, xf in enumerate(xforms):
        x, y, o = get_coefs_from_string(xf.attrib['coefs'])

        x1, x2 = shift_from_origin(x, o)
        y1, y2 = shift_from_origin(y, o)

        rx = get_vector_length(x, o)
        ry = get_vector_length(y, o)

        tx = get_angle(x1, rx)
        ty = get_angle(y1, ry)

        x_angles[i, 0] = tx
        x_angles[i, 1] = ty

    return x_angles


def animate_flames(flame_path, duration, frames_per_sec, deg_per_frame, note_insts, label, out_dir='flames'):
    n_frames = duration * frames_per_sec
    label_size = math.ceil(math.log10(n_frames))
    measure_length = 8.0
    frames_per_measure = frames_per_sec * measure_length

    root = ET.Element('Flames')
    root.set('name', 'flamepack')

    with open(flame_path, 'rt') as f:
        tree = ET.parse(f)

    tree.find('flame').attrib['name'] = '0'*label_size

    root.append(tree.getroot())

    note_ind = 0

    x_angles = get_xform_angles(tree)

    for i in range(n_frames):
        x_angles += deg_per_frame
        x_angles[0, :] += deg_per_frame*2

        tree1 = set_rotation_xml(tree, x_angles, str(i).zfill(label_size))

        sec = (i % frames_per_measure) / frames_per_sec

        if note_ind == 16 and sec == 0.0:
            note_ind = 0

        if note_insts[note_ind][2] <= sec:
            note_ind += 1
            note_ind = note_ind % len(note_insts)

        # TODO: Take the note at note_ind and use it to define the colors
        if note_insts[note_ind][1] < sec < note_insts[note_ind][2]:
            tree1.find('flame').attrib['brightness'] = '12'
        else:
            tree1.find('flame').attrib['brightness'] = '4'

        # tree1.find('flame').attrib['name'] = str(i).zfill(label_size)
        root.append(tree1.getroot())

        tree = tree1

    all_frames = ET.ElementTree(root)

    all_frames.write(out_dir + label + '.flam3')


note_insts = [
    (2, 0.25, 0.5),
    (2, 0.75, 1.0),
    (9, 1.0, 1.375),
    (7, 1.5, 2.0),

    (2, 2.25, 2.5),
    (2, 2.75, 3.0),
    (9, 3.0, 3.375),
    (7, 3.5, 4.0),

    (2, 4.25, 4.625),
    (2, 4.75, 5.0),
    (9, 5.0, 5.375),
    (7, 5.5, 6.0),

    (11, 6.25, 6.5),
    (11, 6.5, 6.75),
    (2, 6.75, 7.25),
    (6, 7.25, 7.7),
    (11, 7.75, 8.0)
]

note_insts2 = [
    (2, 0.25, 0.5),
    (2, 0.75, 1.0),
    (9, 1.0, 1.375),
    (7, 1.5, 2.0),

    (2, 2.25, 2.5),
    (2, 2.75, 3.0),
    (11, 3.0, 3.38),
    (9, 3.5, 3.75),
    (6, 3.75, 4.0),

    (4, 4.0, 4.25),
    (2, 4.25, 4.75),
    (9, 4.75, 5.0),
    (7, 5.0, 5.25),
    (6, 5.25, 5.5),
    (7, 5.5, 5.75),

    (6, 6.25, 6.5),
    (4, 6.5, 7.0),
    (2, 7.0, 7.5),
    (1, 7.75, 8.0)
]

animate_flames('flames/final_v1.flam3', 32, 10, 3, note_insts, 'final_v6_anim', 'flames/')
