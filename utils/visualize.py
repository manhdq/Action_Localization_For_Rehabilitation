import cv2
import os
import time
import datetime
import numpy as np
import pandas as pd

from torchvision.io import read_video


def save_video(video, segments_info, label_list, save_dir='output', gt_infos=None, heatmap_extractor=None, kpts_list=None, heatmaps_list=None, save_rehab_info=False, track_clip=False, track_conf=False, track_score_each_action=False):
    assert not isinstance(video, list) and not isinstance(segments_info, list), 'current support for 1 video'
    os.makedirs(save_dir, exist_ok=True)

    if gt_infos is not None:
        segments_gt, actions_gt = gt_infos

    if isinstance(video, str):
        # if video is a file
        vframes, _, _ = read_video(video)
    else:
        vframes = video
    vframes = vframes.numpy()
    src = segments_info['video_id']
    segments = segments_info['segments']
    segments_frames = segments_info['segments_frames']
    scores = segments_info['scores']
    labels = segments_info['labels']

    # frame_width = 800
    # frame_height = 420
    video_num_frames = len(vframes)
    img_shape = (860, 480)
    track_height = 0
    track_gap = 0

    if track_clip:
        fps = 30.0
        track_height = 40
        track_gap = 50

        vid_track = np.ones((40*2 + 50*2, img_shape[0], 3)).astype(np.uint8) * 80
        vid_track = cv2.putText(vid_track, 'Prediction:', (20, track_height - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        vid_track = cv2.putText(vid_track, 'Ground Truth:', (20, track_height*2 + track_gap - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        vid_track[track_height:track_height+track_gap, :, :] = 0
        vid_track[track_height*2+track_gap:, :, :] = 0

        for segment in segments:
            frame_segments = (int(segment[0] * fps / video_num_frames * img_shape[0]), int(segment[1] * fps / video_num_frames * img_shape[0]))
            vid_track[track_height:track_height+track_gap, frame_segments[0]: frame_segments[1]] = vid_track[130:, frame_segments[0]: frame_segments[1]] + np.array([0, 255, 0], dtype=np.uint8)
        if gt_infos is not None:
            for segment in segments_gt:
                frame_segments = (int(segment[0] * fps / video_num_frames * img_shape[0]), int(segment[1] * fps / video_num_frames * img_shape[0]))
                vid_track[track_height*2+track_gap:, frame_segments[0]: frame_segments[1]] = vid_track[40:90, frame_segments[0]: frame_segments[1]] + np.array([242, 235, 29], dtype=np.uint8)

    out = cv2.VideoWriter(os.path.join(save_dir, 'test.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (img_shape[0],int(img_shape[1] + (track_height + track_gap) * 2)))
    current_actions = []
    frame_idx = 0

    if save_rehab_info:
        num_actions = len(segments)
        times = [seg[1] - seg[0] for seg in segments]
        total_action_times = np.sum(times)
        total_times = segments[-1][1] - segments[0][0] / 60.

        ts = datetime.datetime.fromtimestamp(int(time.time()))
        
        rehab_infos = {
            'video_id': [src],
            'time': [ts],
            'num_actions': [num_actions],
            'total_rehabilitation_times (min)': [total_times],
            'total_action_times (min)': [total_action_times],
            'velocity (s/action)': [total_action_times / num_actions],
            'action_name': [label_list[labels[0]]],
            'action_details': [segments],
        }
        rehab_csv = pd.DataFrame(rehab_infos)
        rehab_csv.to_csv(os.path.join(save_dir, 'rehab_doc.csv'), index=False)

    for idx, frame in enumerate(vframes):
        frame_idx += 1

        # RGB -> BGR
        frame = frame[..., ::-1].astype(np.uint8)
        if kpts_list is not None and len(kpts_list) > idx:
            if kpts_list[idx]:
                kpts = kpts_list[idx][0][0]
                if len(kpts)==17:
                    heatmap_extractor.draw_pose(kpts, frame)
        
        frame = cv2.resize(frame, img_shape)
        # if heatmaps_list is not None and len(heatmaps_list) > idx:
        #     heatmaps = heatmaps_list[idx]
        #     heatmaps = np.sum(heatmaps, axis=-1) * 255.
        #     heatmaps = heatmaps.clip(0, 255).astype(np.uint8)
        #     heatmaps = np.stack([heatmaps] * 3, axis=-1)
        #     heatmaps = cv2.resize(heatmaps, (frame_width, frame_height))
        #     frame = cv2.addWeighted(frame, 0.5, heatmaps, 0.5, 0.0)
        
        # Add action to frame
        num_add_this_frame = 0
        for idx, segment_frame in enumerate(segments_frames):
            if segment_frame[0] <= frame_idx:
                action = {
                    'segment': segments[idx],
                    'segment_frame': segment_frame,
                    'score': scores[idx],
                    'label': label_list[labels[idx]]
                }
                current_actions.append(action)
                num_add_this_frame += 1
            else:
                break
        segments = segments[num_add_this_frame:]
        segments_frames = segments_frames[num_add_this_frame:]
        scores = scores[num_add_this_frame:]
        labels = labels[num_add_this_frame:]
        # for _ in range(num_add_this_frame):
        #     segments.pop(0)
        #     segments_frames.pop(0)
        #     scores.pop(0)
        #     labels.pop(0)
        # print(len(segments_frames))

        # Delete action from frame
        num_del_this_frame = 0
        action_del_idxs = []
        for idx, action in enumerate(current_actions):
            if action['segment_frame'][1] < frame_idx:
                action_del_idxs.append(idx)
                num_del_this_frame += 1
        for action_del_idx in action_del_idxs[::-1]:
            current_actions.pop(action_del_idx)

        if len(current_actions) == 0:
            cv2.putText(frame, 'No Action', (10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=3, lineType=cv2.LINE_AA)
        else:
            for idx, action in enumerate(current_actions):
                loc = (10, 40*(idx + 1))
                text = f"{action['label']} - {action['score']:.2f}"
                cv2.putText(frame, text, loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

        # print(frame.shape)
        # print(len(current_actions))
        # cv2.imwrite(f"test/{frame_idx}.jpg", frame)
        if track_clip:
            frame_vid_track = vid_track.copy()
            frame_window_track_center = frame_idx / video_num_frames * img_shape[0]
            frame_window_track_left = max(int(frame_window_track_center - 0.008 * img_shape[0]), 0)
            frame_window_track_right = min(int(frame_window_track_center + 0.008 * img_shape[0]), img_shape[0])
            frame_vid_track[track_height:track_height+track_gap, frame_window_track_left:frame_window_track_right] = np.ones_like(frame_vid_track[40:90, frame_window_track_left:frame_window_track_right]) * np.array([255, 0, 0], dtype=np.uint8)
            frame_vid_track[track_height*2+track_gap:, frame_window_track_left:frame_window_track_right] = np.ones_like(frame_vid_track[130:, frame_window_track_left:frame_window_track_right]) * np.array([255, 0, 0], dtype=np.uint8)
            frame = np.concatenate((frame, frame_vid_track), axis=0)
        # print(frame.shape)
        # exit(0)
        out.write(frame)

    cv2.destroyAllWindows()
    out.release()