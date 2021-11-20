import math
import click
import os

from threading import main_thread
import numpy as np
from scipy.ndimage import measurements
import scipy.spatial
import skimage.draw
import skimage.segmentation
import skimage.measure
import skimage.morphology

import tqdm
import echonet

def getPerpCoord(aX, aY, bX, bY, pointX, pointY, length):
    vX = bX-aX
    vY = bY-aY
    #print(str(vX)+" "+str(vY))
    # if(vX == 0 or vY == 0):
    #     return 0, 0, 0, 0
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = pointX + vX * length
    cY = pointY + vY * length
    dX = pointX - vX * length
    dY = pointY - vY * length
    return int(cX), int(cY), int(dX), int(dY)


def compute_measurements(logit,video,w):


    measurements = dict()
    # Pixel Boundaries (perimeter) - Returns Boolean
    boundary = skimage.segmentation.find_boundaries(logit > 0, mode='thick')
    measurements['boundary'] = boundary
    boundary_outer = skimage.segmentation.find_boundaries(logit > 0, mode='outer')
    # Transform boolean to labeled image (0,1)
    labeled_logit = skimage.measure.label(logit > 0)

    longest_axis_size = []
    short_axis_1_size = []
    short_axis_2_size = []
    measurements['boundary_coordinates'] = []
    measurements['centroid'] = []

    rr_prev_top = None
    cc_prev_top = None
    rr_prev_bottom = None
    cc_prev_bottom = None
    rr_prev_left = None
    cc_prev_left = None
    rr_prev_right = None
    cc_prev_right = None

    velocities = dict()
    velocities['topX'] = []
    velocities['topY'] = []
    velocities['bottomX'] = []
    velocities['bottomY'] = []
    velocities['lengthTop'] = []
    velocities['lengthBottom'] = []
    velocities['leftX'] = []
    velocities['leftY'] = []
    velocities['rightX'] = []
    velocities['rightY'] = []
    velocities['lengthLeft'] = []
    velocities['lengthRight'] = []

    perimeters = []

    for (count, frame) in enumerate(labeled_logit):
        if frame.sum() > 15:
            # compute boundary box (perimeter pixels)
            props = skimage.measure.regionprops(frame)[0]
            min_row, min_col, max_row, max_col = props.bbox
            measurements['boundary_coordinates'].append(np.transpose(boundary_outer[count].nonzero()))
            measurements['centroid'].append(props.centroid)
            max_column_size = max_col - min_col

            # Compute perimeter
            perimeters.append(skimage.measure.perimeter((logit>0)[count]))

            # Smallest convex polygon
            convex = skimage.morphology.convex_hull_image(frame)

            # Candidate coordinates combining the convex and the perimeter
            candidates_boolean = np.logical_and(convex, boundary[count, :, :])
            candidates = np.column_stack(np.where(candidates_boolean))

            # Distance matrix - max distance coordinates - draw line
            dist_mat = scipy.spatial.distance_matrix(candidates, candidates)
            max_i, max_j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
            rr, cc = skimage.draw.line(candidates[max_i][0], candidates[max_i][1], candidates[max_j][0], candidates[max_j][1])

            # Check if line inside the shape, if not find the second largest line etc.
            if labeled_logit[count, rr, cc].sum() != rr.shape[0]:
                # Make largest distance zero and find the second largest distance
                dist_mat[max_i, max_j] = 0
                max_i, max_j = np.unravel_index(
                    dist_mat.argmax(), dist_mat.shape)
                rr, cc = skimage.draw.line(
                    candidates[max_i][0], candidates[max_i][1], candidates[max_j][0], candidates[max_j][1])

            longest_axis_size.append(scipy.spatial.distance.euclidean([rr[0],cc[0]],[rr[-1],cc[-1]]))

            # Find perpendicular lines to the longest axis
            frame_short_lines = []
            center_flag = False
            for (line_count, (r_long, c_long)) in enumerate(zip(rr, cc)):
                # for each point in the line find its perpendicular
                cX, cY, dX, dY = getPerpCoord(
                    candidates[max_i][0], 
                    candidates[max_i][1], 
                    candidates[max_j][0], 
                    candidates[max_j][1], 
                    r_long, c_long, 160)

                rr_per, cc_per = skimage.draw.line(
                    cX, cY, dX, dY)
                new_line_points = []
                for r_short, c_short in zip(rr_per, cc_per):
                    if r_short > 111 or c_short > 111 or r_short < 0 or c_short < 0:
                        continue
                    if boundary[count, r_short, c_short]:
                        new_line_points.append(
                            [r_short, c_short])
                center_coord = int(rr.shape[0]/2)
                if len(new_line_points) > 1:
                    rline_rr, rline_cc = skimage.draw.line(
                        new_line_points[0][0], new_line_points[0][1], new_line_points[1][0], new_line_points[1][1])
                    if line_count == center_coord:
                        rr_central, cc_central = rline_rr, rline_cc
                    frame_short_lines.append(
                        [rline_rr, rline_cc])
                # else: # quick fix
                #     if line_count == center_coord:
                #         if center_flag:
                #             print("Error!!")
                #         center_coord += 1
                #         center_flag = True

            longest_short_line_index = -1
            longest_short_line_length = 0
            for c, short_line in enumerate(frame_short_lines):
                if len(short_line[0]) > longest_short_line_length:
                    longest_short_line_index = c
                    longest_short_line_length = len(
                        short_line[0])

            rr_short_axis = frame_short_lines[longest_short_line_index][0]
            cc_short_axis = frame_short_lines[longest_short_line_index][1]

            # video[count,0,min_row,min_col:max_col] = [255.] * (max_col - min_col)
            # video[count,0,min_row:max_row,max_col] = [255.] * (max_row - min_row)
            # video[count,0,min_row:max_row,min_col] = [255.] * (max_row - min_row)
            # video[count,0,max_row,min_col:max_col] = [255.] * (max_col - min_col)
            video[count, :, rr, 2*w+cc] = [255, 0, 0]
            # video[count,0,rr_per,2*w+cc_per] = 255
            video[count, :, rr_short_axis, 2*w +cc_short_axis] = [255, 255, 255]
            video[count, :, rr_central, 2 *w+cc_central] = [0, 255, 255]

            # Longest line perpendicular to long axis
            shortest_line = frame_short_lines[longest_short_line_index]
            # short_axis_1_size.append(
            #     frame_short_lines[longest_short_line_index][0].shape[0])
            short_axis_1_size.append(scipy.spatial.distance.euclidean([shortest_line[0][0],shortest_line[1][0]],[shortest_line[0][-1],shortest_line[1][-1]]))
            # Centered line perpendicular to long axis
            # short_axis_2_size.append(rr_central.shape[0])
            short_axis_2_size.append(scipy.spatial.distance.euclidean([rr_central[0],cc_central[0]],[rr_central[-1],cc_central[-1]]))

            # Velocity and Speed
            if rr_prev_top != None and rr_prev_left != None:
                velocities['topX'].append(rr[0] - rr_prev_top)
                velocities['topY'].append(cc[0] - cc_prev_top)
                velocities['bottomX'].append(
                    rr[-1] - rr_prev_bottom)
                velocities['bottomY'].append(
                    cc[-1] - cc_prev_bottom)
                velocities['lengthTop'].append(scipy.spatial.distance.euclidean(
                    [rr[0], cc[0]], [rr_prev_top, cc_prev_top]))
                velocities['lengthBottom'].append(scipy.spatial.distance.euclidean(
                    [rr[-1], cc[-1]], [rr_prev_bottom, cc_prev_bottom]))
                velocities['leftX'].append(
                    rr_short_axis[0] - rr_prev_left)
                velocities['leftY'].append(
                    cc_short_axis[0] - cc_prev_left)
                velocities['rightX'].append(
                    rr_short_axis[-1] - rr_prev_right)
                velocities['rightY'].append(
                    cc_short_axis[-1] - cc_prev_right)
                velocities['lengthLeft'].append(scipy.spatial.distance.euclidean(
                    [rr_short_axis[0], cc_short_axis[0]], [rr_prev_left, cc_prev_left]))
                velocities['lengthRight'].append(scipy.spatial.distance.euclidean(
                    [rr_short_axis[-1], cc_short_axis[-1]], [rr_prev_right, cc_prev_right]))
            else:
                velocities['topX'].append(0)
                velocities['topY'].append(0)
                velocities['bottomX'].append(0)
                velocities['bottomY'].append(0)
                velocities['lengthTop'].append(0)
                velocities['lengthBottom'].append(0)
                velocities['leftX'].append(0)
                velocities['leftY'].append(0)
                velocities['rightX'].append(0)
                velocities['rightY'].append(0)
                velocities['lengthLeft'].append(0)
                velocities['lengthRight'].append(0)

            rr_prev_top = rr[0]
            rr_prev_bottom = rr[-1]
            cc_prev_top = cc[0]
            cc_prev_bottom = cc[-1]
            rr_prev_left = rr_short_axis[0]
            rr_prev_right = rr_short_axis[-1]
            cc_prev_left = cc_short_axis[0]
            cc_prev_right = cc_short_axis[-1]
        else:
            velocities['topX'].append(0)
            velocities['topY'].append(0)
            velocities['bottomX'].append(0)
            velocities['bottomY'].append(0)
            velocities['lengthTop'].append(0)
            velocities['lengthBottom'].append(0)
            velocities['leftX'].append(0)
            velocities['leftY'].append(0)
            velocities['rightX'].append(0)
            velocities['rightY'].append(0)
            velocities['lengthLeft'].append(0)
            velocities['lengthRight'].append(0)
            perimeters.append(0)
            longest_axis_size.append(0)
            short_axis_1_size.append(0)
            short_axis_2_size.append(0)
            measurements['centroid'].append(0)
            measurements['boundary_coordinates'].append(np.array([0]))

    measurements['perimeters'] = perimeters
    measurements['velocities'] = velocities
    measurements['longest_axis_size'] = longest_axis_size 
    measurements['short_axis_1_size'] = short_axis_1_size 
    measurements['short_axis_2_size'] = short_axis_2_size
    return video, measurements

# Compute measurements on training and validation set
@click.command("measurements")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output_dir", type=click.Path(file_okay=False), default=None)
@click.option("--set_type", type=str, default="train")
def run(data_dir=None,output_dir=None,set_type="train"):
    dataset = dict()
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split=set_type))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]

    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }
    dataset[set_type] = echonet.datasets.Echo(root=data_dir, split=set_type, **kwargs)
    with open(os.path.join(output_dir, "{}.csv".format(set_type)), "w") as g:
        g.write("Filename,Frame,Size,Perimeter,HumanLarge,HumanSmall,LongAxis,ShortAxis1,ShortAxis2,TopX,TopY,BottomX,BottomY,LeftX,LeftY,RightX,RightY,LTop,LBottom,LLeft,LRight\n")
        for (count,(_, (large_frame, small_frame, large_trace, small_trace))) in enumerate(tqdm.tqdm(dataset[set_type])):
            video = np.array([large_frame,small_frame])
            logit = np.array([large_trace,small_trace])
            f, c, h, w = video.shape
            video = np.concatenate((video, video, video), 3)
            video[:, 0, :, w:2*w] = np.maximum(255. * (logit > 0), video[:, 0, :, w:2*w])  # pylint: disable=E1111
            video, measurements = compute_measurements(logit,video,w)

            velocities = measurements['velocities']
            longest_axis_size = measurements['longest_axis_size'] 
            short_axis_1_size  = measurements['short_axis_1_size']
            short_axis_2_size = measurements['short_axis_2_size']
            size = (logit > 0).sum((1, 2))
            # Draw perimeter
            # video[:, 1, :, 2*w:] = np.maximum(255. * measurements['boundary'], video[:, 0, :, 2*w:])  # pylint: disable=E1111
            
            
            for (frame, s) in enumerate(size):
                p = measurements['perimeters'][frame]
                g.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(dataset[set_type].fnames[count], frame, s, p, 1 if frame ==
                        0 else 0, 1 if frame == 1 else 0, 
                        longest_axis_size[frame], 
                        short_axis_1_size[frame], 
                        short_axis_2_size[frame],
                        velocities['topX'][frame],velocities['topY'][frame],
                        velocities['bottomX'][frame],velocities['bottomY'][frame],
                        velocities['leftX'][frame],velocities['leftY'][frame],
                        velocities['rightX'][frame],velocities['rightY'][frame],
                        velocities['lengthTop'][frame],velocities['lengthBottom'][frame],
                        velocities['lengthLeft'][frame],velocities['lengthRight'][frame]))
            # Rearrange dimensions and save
            # video = video.transpose(1, 0, 2, 3)
            # video = video.astype(np.uint8)
            # # echonet.utils.savevideo(os.path.join(output_dir, "videos", dataset[set_type].fnames[count]), video, 50)
            # pass


if __name__ == '__main__':
    run()