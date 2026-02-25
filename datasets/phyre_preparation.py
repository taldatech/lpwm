"""
Adapted from: Solving Physics Puzzles by Reasoning about Paths
https://github.com/ndrwmlnk/PHYRE-Reasoning-about-Paths
"""

import phyre
import cv2
import pickle
import pathlib
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import gzip


# import torch


def collect_images():
    tries = 0
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
             '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
             '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
             '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
             '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']

    tasks = ["00019:612"]
    base_path = "fiddeling"
    number_to_solve = 20

    for task in tasks:
        sim = phyre.initialize_simulator([task], 'ball')
        solved = 0
        while solved < number_to_solve:
            tries += 1
            action = sim.sample()
            res = sim.simulate_action(0, action, need_featurized_objects=True)
            if res.status.is_solved():
                print("solved " + task + " with", tries, "tries")
                tries = 0
                solved += 1
                # print(res.images.shape)
                for i, scene in enumerate(res.images):
                    img = phyre.observations_to_uint8_rgb(scene)
                    path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
                    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(path_str + f"/{str(i)}.jpg", img)
                    with open(path_str + "/objects.pickle", 'wb') as handle:
                        pickle.dump(res.featurized_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        # print(res.featurized_objects)
                    with open(path_str + "/action.txt", 'w') as handle:
                        handle.write(str(action))


def collect_solving_observations(path, tasks, n_per_task=10, collect_base=True, stride=10, size=(32, 32)):
    end_char = '\r'
    tries = 0
    max_tries = 100
    base_path = path
    number_to_solve = n_per_task
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    lib_dict = dict()

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        solved = 0
        while solved < number_to_solve:
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
            if not os.path.exists(path_str + "/observations.pickle"):
                print(f"collecting {task}: trial {solved} with {tries + 1} tries", end=end_char)
                tries += 1
                action = actions[cache.load_simulation_states(task) == 1]
                if len(action) == 0:
                    print("WARNING no solution action in cache at task", task)
                    action = [np.random.rand(3)]
                action = random.choice(action)
                res = sim.simulate_action(idx, action,
                                          need_featurized_objects=True, stride=stride)
                if res.status.is_solved():
                    tries = 0
                    solved += 1
                    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                    with open(path_str + "/observations.pickle", 'wb') as handle:
                        rollout = np.array(
                            [[cv2.resize((scene == ch).astype(float), size, cv2.INTER_MAX) for ch in range(1, 7)] for
                             scene in res.images])
                        pickle.dump(rollout, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if tries > max_tries:
                    break
            else:
                solved += 1
                print(f"skipping {task}: trial {solved}", end=end_char)

        # COLLECT BASE
        if collect_base and tries <= max_tries:
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/base"
            if not os.path.exists(path_str + "/observations.pickle"):
                print(f"collecting {task}: base", end=end_char)
                # 10 tries to make increase chance of one action being valid
                for _ in range(10):
                    action = sim.sample()
                    action[2] = 0.01
                    res = sim.simulate_action(idx, action,
                                              need_featurized_objects=True, stride=stride)
                    if not res.status.is_invalid():
                        break
                pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                with open(path_str + "/observations.pickle", 'wb') as handle:
                    rollout = np.array(
                        [[cv2.resize((scene == ch).astype(float), size, cv2.INTER_LINEAR) for ch in range(1, 7)] for
                         scene in res.images])
                    pickle.dump(rollout, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f"skipping {task}: base", end=end_char)

    print("FINISH collecting rollouts!")


def collect_solving_dataset(path, tasks, n_per_task=10, collect_base=True, stride=10, size=(32, 32), solving=True,
                            proposal_dict=None, dijkstra=True, pertempl=False):
    end_char = '\r'
    tries = 0
    max_tries = 510
    base_path = path
    number_to_solve = n_per_task
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array

    print(pertempl)
    if pertempl:
        templates = set(t[:5] for t in tasks)
        tmp = []
        for templ in templates:
            possible_tasks = [t for t in tasks if t.__contains__(templ)]
            for _ in range(n_per_task):
                tmp.append(random.choice(possible_tasks))
        assert len(templates) * n_per_task == len(tmp)
        tasks = tmp
        number_to_solve = 1

    path_idxs = [1, 2, 0]
    channels = range(1, 7)
    data = []
    acts = []
    lib_dict = dict()
    task_list = []

    sim = phyre.initialize_simulator(tasks, 'ball')
    for task_idx, task in enumerate(tasks):
        # COLLECT SOLVES
        solved = 0
        if proposal_dict is not None:
            proposal_list = [item for item in proposal_dict[task]]
        cache_list = actions[cache.load_simulation_states(task) == (1 if solving else -1)]
        while solved < number_to_solve:
            print(f"collecting {task}: trial {solved} with {tries + 1} tries", end=end_char)
            tries += 1
            if tries < max_tries - 10 and (proposal_dict is not None):
                actionlist = proposal_list
            else:
                actionlist = cache_list

            if len(actionlist) == 0:
                print("WARNING no solution action in cache at task", task)
                actionlist = [np.random.rand(3)]
            action = random.choice(actionlist)
            res = sim.simulate_action(task_idx, action,
                                      need_featurized_objects=True, stride=stride)

            # IF SOLVED PROCESS ROLLOUT
            if (res.status.is_solved() == solving) and not res.status.is_invalid():
                acts.append(action)
                tries = 0
                solved += 1

                # FORMAT AND EXTRACT DATA
                paths = np.zeros((len(path_idxs), len(res.images), size[0], size[1]))
                alpha, gamma = 1, 1
                for i, image in enumerate(res.images):
                    # extract color codings from channels
                    chans = np.array([(image == ch).astype(float) for ch in channels])

                    # at first frame extract init scene
                    if not i:
                        init_scene = np.array([(cv2.resize(chans[ch], size, cv2.INTER_MAX) > 0).astype(float) for ch in
                                               range(len(channels))])

                    # add path_idxs channels to paths
                    for path_i, idx in enumerate(path_idxs):
                        paths[path_i, i] = alpha * (cv2.resize(chans[idx], size, cv2.INTER_MAX) > 0).astype(float)
                    alpha *= gamma

                # COLLECT BASE
                if collect_base:
                    print(f"collecting {task}: base", end=end_char)
                    # 1000 tries make sure one action is valid
                    for _ in range(1000):
                        action = sim.sample()
                        action[2] = 0.001
                        res = sim.simulate_action(task_idx, action,
                                                  need_featurized_objects=False, stride=stride)
                        if not res.status.is_invalid():
                            break
                    base_frames = \
                        np.array([cv2.resize((scene == 2).astype(float), size, cv2.INTER_MAX) for scene in res.images])[
                            None]

                # combine channels
                # flip y axis and concat init scene with paths and base
                paths = np.flip(np.max(paths, axis=1).astype(float), axis=1)
                base = np.flip(np.max(base_frames, axis=1).astype(float), axis=1)
                init_scene = np.flip(init_scene, axis=1)

                # make distance map
                if dijkstra:
                    dm_init_scene = sim.initial_scenes[task_idx]
                    img = cv2.resize(phyre.observations_to_float_rgb(dm_init_scene), size, cv2.INTER_MAX)  # read image
                    target = np.logical_or(init_scene[2] == 1, init_scene[3] == 1)
                    # cv2.imwrite('maze-initial.png', img)
                    distance_map = find_distance_map_obj(img, target) / 255
                    combined = (255 * np.concatenate([init_scene, base, paths, distance_map[None]])).astype(np.uint8)
                else:
                    combined = (255 * np.concatenate([init_scene, base, paths, base * 0])).astype(np.uint8)

                # append data set and lib_dict
                data.append(combined)
                task_list.append(task)
                if task in lib_dict:
                    lib_dict[task].append(len(data) - 1)
                else:
                    lib_dict[task] = [len(data) - 1]

            if tries > max_tries:
                break

    os.makedirs(path, exist_ok=True)
    file = gzip.GzipFile(path + '/data.pickle', 'wb')
    pickle.dump((data, acts), file)
    file.close()
    with open(path + '/index.pickle', 'wb') as fp:
        pickle.dump(task_list, fp)

    print(f"FINISH collecting {'solving' if solving else 'failing'} dataset!")


def collect_specific_channel_paths(path, tasks, channel, stride=10, size=(256, 256)):
    end_char = '\r'
    tries = 0
    max_tries = 100
    base_path = path
    number_to_solve = 10
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    data_dict = dict()

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        solved = 0
        while solved < number_to_solve:
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
            if not os.path.exists(path_str + "/observations.pickle"):
                print(f"collecting channel {channel} from {task}: trial {solved} with {tries + 1} tries", end=end_char)
                tries += 1
                action = actions[cache.load_simulation_states(task) == 1]
                if len(action) == 0:
                    print("no solution action in cache at task", task)
                    action = [np.random.rand(3)]
                action = random.choice(action)
                res = sim.simulate_action(idx, action,
                                          need_featurized_objects=True, stride=stride)
                if res.status.is_solved():
                    tries = 0
                    solved += 1
                    # pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                    rollout = np.array(
                        [[cv2.resize((scene == ch).astype(float), size, cv2.INTER_MAX) for ch in range(1, 7)] for scene
                         in res.images])
                    extracted_path = np.max(rollout[:, channel], axis=0)
                    # Collect index
                    key = task
                    if key in data_dict:
                        data_dict[key].append(extracted_path)
                    else:
                        data_dict[key] = [extracted_path]
                if tries > max_tries:
                    break
            else:
                solved += 1
                print(f"skipping {task}: trial {solved}", end=end_char)

    # Save data_dict
    with open(f'{base_path}/channel_paths.pickle', 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting channel {channel} paths!")


def collect_interactions(save_path, tasks, number_per_task, step_size=20, size=(64, 64), static=0, show=False):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    # base_path = 'data/fiddeling'
    data = []
    info = {'tasks': [], 'pos': [], 'action': []}
    print("Amount per task", number_per_task)

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task) == 1]
            print(f"collecting {n_collected + 1} interactions from {task} with {tries} tries", end=end_char)
            if len(action) == 0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                                      need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                # print(res.images.shape)
                # print(len(res.bitmap_seq))
                # print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                # print(idx1, idx2)
                # print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx, red_idx]]) / 2
                for i, m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        pos = res.featurized_objects.features[i, [green_idx, red_idx], :2]
                        dist = np.linalg.norm(pos[1] - pos[0])
                        # print(dist, target_dist)
                        if not dist < target_dist + 0.005:
                            continue

                        # print(res.featurized_objects.diameters[[green_idx,red_idx]])
                        # print(res.featurized_objects.features[i,green_idx])
                        # print(res.featurized_objects.features[i, red_idx])
                        # print(i+2)
                        # for i, scene in enumerate(res.images):
                        #    img = phyre.observations_to_uint8_rgb(scene)
                        #    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
                        #    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                        #    cv2.imwrite(path_str+f"/{str(i)}.jpg", img[:,:,::-1])

                        red_radius = res.featurized_objects.diameters[red_idx] * 4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i + 1, pos[0], action_at_interaction, target_dist)

                return (False, 0, (0, 0), 0, 0)

            contact, i_step, green_pos, red_pos, summed_radii = check_contact(res)
            if contact:
                tries = 0
                n_collected += 1

                # setting up parameters for cutting out selection
                width = round(256 * summed_radii * 4)
                if static:
                    width = static
                wh = width // 2
                starty = round((green_pos[1]) * 256)
                startx = round(green_pos[0] * 256)
                step_size = step_size
                # check whether contact happend too early
                if i_step - step_size <= 0 or i_step + step_size >= len(res.images):
                    continue

                selected_rollout = np.array([[(scene == ch).astype(float) for ch in range(1, 7)] for scene in
                                             res.images[[i_step - step_size, i_step, i_step + step_size, 0]]])
                # selected_rollout = np.flip(selected_rollout, axis=2)
                # print(selected_rollout.shape)

                # Padding
                border = 8
                padded_selected_rollout = np.pad(selected_rollout, ((0, 0), (0, 0), (border, border), (border, border)),
                                                 constant_values=1)
                padded_selected_rollout = np.pad(padded_selected_rollout, (
                    (0, 0), (0, 0), (wh - border, wh - border), (wh - border, wh - border)))
                # print(padded_selected_rollout.shape)

                # Cutting out
                extracted_scene = padded_selected_rollout[:, :, starty:starty + width, startx:startx + width]

                # Correcting for flipped indexing from Phyre
                extracted_scene = np.flip(extracted_scene, axis=2)

                # Formatting and resizing
                es = extracted_scene
                channel_formatted_scene = np.stack(
                    (es[0, 1], es[1, 1], es[2, 1], np.max(es[1, 2:], axis=0), es[0, 0], es[1, 0], es[3, 0]))
                size_formatted_scene = [cv2.resize(img, size, cv2.INTER_MAX) for img in channel_formatted_scene]

                # saving extracted scene
                data.append(size_formatted_scene)
                info['tasks'].append(task)
                info['pos'].append(green_pos)
                info['action'].append(red_pos)

                if show:
                    print(starty, startx, width)
                    plt.imshow(phyre.observations_to_uint8_rgb(res.images[i_step]))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(channel_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*channel_formatted_scene], axis=1))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(size_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*size_formatted_scene], axis=1))
                    plt.show()

            if tries > max_tries:
                break

    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/interactions.pickle', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{base_path}/info.pickle', 'wb') as fp:
        pickle.dump(info, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting interactions!")


def collect_flow_interactions(save_path, tasks, number_per_task, step_size=20, size=(64, 64), static=0, show=False):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    # base_path = 'data/fiddeling'
    data = []
    info = {'tasks': [], 'pos': [], 'action': []}
    print("Amount per task", number_per_task)

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task) == 1]
            print(f"collecting {n_collected + 1} interactions from {task} with {tries} tries", end=end_char)
            if len(action) == 0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                                      need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                # print(res.images.shape)
                # print(len(res.bitmap_seq))
                # print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                # print(idx1, idx2)
                # print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx, red_idx]]) / 2
                for i, m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        all_green_pos = res.featurized_objects.features[:, green_idx, :2]
                        all_red_pos = res.featurized_objects.features[:, red_idx, :2]
                        pos = all_green_pos[i]
                        dist = np.linalg.norm(pos[1] - pos[0])
                        # print(dist, target_dist)
                        if not dist < target_dist + 0.005:
                            continue

                        # print(res.featurized_objects.diameters[[green_idx,red_idx]])
                        # print(res.featurized_objects.features[i,green_idx])
                        # print(res.featurized_objects.features[i, red_idx])
                        # print(i+2)
                        # for i, scene in enumerate(res.images):
                        #    img = phyre.observations_to_uint8_rgb(scene)
                        #    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
                        #    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                        #    cv2.imwrite(path_str+f"/{str(i)}.jpg", img[:,:,::-1])

                        red_radius = res.featurized_objects.diameters[red_idx] * 4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i + 1, all_green_pos, all_red_pos, action_at_interaction, target_dist)

                return (False, 0, (0, 0), 0, 0, 0)

            contact, i_step, all_green_pos, all_red_pos, act_at_interact, summed_radii = check_contact(res)
            if contact:
                tries = 0
                n_collected += 1
                green_pos = all_green_pos[i_step]

                # setting up parameters for cutting out selection
                width = round(256 * summed_radii * 4)
                if static:
                    width = static
                wh = width // 2
                starty = round((green_pos[1]) * 256)
                startx = round(green_pos[0] * 256)
                step_size = step_size
                # check whether contact happend too early
                if i_step - step_size <= 0 and i_step + step_size < len(res.images):
                    continue

                selected_rollout = np.array([[(scene == ch).astype(float) for ch in range(1, 7)] for scene in
                                             res.images[i_step - step_size:i_step + step_size + 1:step_size]])
                # selected_rollout = np.flip(selected_rollout, axis=2)
                # print(selected_rollout.shape)

                # Padding
                padded_selected_rollout = np.pad(selected_rollout, ((0, 0), (0, 0), (wh, wh), (wh, wh)))
                # print(padded_selected_rollout.shape)

                # Cutting out
                extracted_scene = padded_selected_rollout[:, :, starty:starty + width, startx:startx + width]

                # Correcting for flipped indexing from Phyre
                extracted_scene = np.flip(extracted_scene, axis=2)
                es = extracted_scene

                # FORMATTING FLOW DATA
                starty = (all_red_pos[i_step, 1] * 256)
                startx = (all_red_pos[i_step, 0] * 256)
                minusy = (all_red_pos[i_step - step_size, 1] * 256)
                minusx = (all_red_pos[i_step - step_size, 0] * 256)

                xdelta = float(startx - minusx) / 15
                ydelta = float(starty - minusy) / 15

                for pos in all_green_pos:
                    # print(pos)
                    pass
                print(xdelta, ydelta)

                xvel = np.zeros_like(es[0, 0])
                xvel[es[1, 0] > 0] = max(-1, min(xdelta, 1))

                yvel = np.zeros_like(es[0, 0])
                yvel[es[1, 0] > 0] = max(-1, min(ydelta, 1))

                # Formatting and resizing
                channel_formatted_scene = np.stack(
                    (es[0, 1], es[1, 1], es[2, 1], np.max(es[1, 2:], axis=0), es[1, 0], xvel, yvel))
                size_formatted_scene = [cv2.resize(img, size, cv2.INTER_MAX) for img in channel_formatted_scene]

                # saving extracted scene
                data.append(size_formatted_scene)
                info['tasks'].append(task)
                info['pos'].append(green_pos)
                info['action'].append(act_at_interact)

                if show:
                    print(starty, startx, width)
                    plt.imshow(phyre.observations_to_uint8_rgb(res.images[i_step]))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(channel_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*channel_formatted_scene], axis=1))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(size_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*size_formatted_scene], axis=1))
                    plt.show()

            if tries > max_tries:
                break

    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/interactions.pickle', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{base_path}/info.pickle', 'wb') as fp:
        pickle.dump(info, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting interactions!")


def collect_delta_interactions(save_path, tasks, number_per_task, step_size=20, size=(64, 64), static=0, show=False):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    # base_path = 'data/fiddeling'
    data = []
    info = {'tasks': [], 'pos': [], 'action': [], 'deltas': []}
    print("Amount per task", number_per_task)

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task) == 1]
            print(f"collecting {n_collected + 1} interactions from {task} with {tries} tries", end=end_char)
            if len(action) == 0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                                      need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                # print(res.images.shape)
                # print(len(res.bitmap_seq))
                # print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                # print(idx1, idx2)
                # print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx, red_idx]]) / 2
                for i, m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        all_green_pos = res.featurized_objects.features[:, green_idx, :2]
                        all_red_pos = res.featurized_objects.features[:, red_idx, :2]
                        pos = all_green_pos[i]
                        dist = np.linalg.norm(pos[1] - pos[0])
                        # print(dist, target_dist)
                        if not dist < target_dist + 0.005:
                            continue

                        # print(res.featurized_objects.diameters[[green_idx,red_idx]])
                        # print(res.featurized_objects.features[i,green_idx])
                        # print(res.featurized_objects.features[i, red_idx])
                        # print(i+2)
                        # for i, scene in enumerate(res.images):
                        #    img = phyre.observations_to_uint8_rgb(scene)
                        #    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
                        #    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                        #    cv2.imwrite(path_str+f"/{str(i)}.jpg", img[:,:,::-1])

                        red_radius = res.featurized_objects.diameters[red_idx] * 4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i + 1, all_green_pos, all_red_pos, action_at_interaction, target_dist)

                return (False, 0, (0, 0), 0, 0, 0)

            contact, i_step, all_green_pos, all_red_pos, act_at_interact, summed_radii = check_contact(res)
            if contact:
                tries = 0
                n_collected += 1
                green_pos = all_green_pos[i_step]

                # setting up parameters for cutting out selection
                width = round(256 * summed_radii * 4)
                if static:
                    width = static
                wh = width // 2
                starty = round((green_pos[1]) * 256)
                startx = round(green_pos[0] * 256)
                step_size = step_size
                # check whether contact happend too early
                if i_step - step_size <= 0 and i_step + step_size < len(res.images):
                    continue

                selected_rollout = np.array([[(scene == ch).astype(float) for ch in range(1, 7)] for scene in
                                             res.images[i_step - step_size:i_step + step_size + 1:step_size]])
                # selected_rollout = np.flip(selected_rollout, axis=2)
                # print(selected_rollout.shape)

                # Padding
                padded_selected_rollout = np.pad(selected_rollout, ((0, 0), (0, 0), (wh, wh), (wh, wh)))
                # print(padded_selected_rollout.shape)

                # Cutting out
                extracted_scene = padded_selected_rollout[:, :, starty:starty + width, startx:startx + width]

                # Correcting for flipped indexing from Phyre
                extracted_scene = np.flip(extracted_scene, axis=2)
                es = extracted_scene

                # FORMATTING FLOW DATA
                starty = (all_red_pos[i_step, 1] * 256)
                startx = (all_red_pos[i_step, 0] * 256)
                minusy = (all_red_pos[i_step - step_size, 1] * 256)
                minusx = (all_red_pos[i_step - step_size, 0] * 256)

                xdelta = float(startx - minusx)
                ydelta = float(starty - minusy)

                for pos in all_green_pos:
                    # print(pos)
                    pass
                print(xdelta, ydelta)

                # Formatting and resizing
                channel_formatted_scene = np.stack((es[0, 1], es[1, 1], es[2, 1], np.max(es[1, 2:], axis=0), es[1, 0]))
                size_formatted_scene = [cv2.resize(img, size, cv2.INTER_MAX) for img in channel_formatted_scene]

                # saving extracted scene
                data.append(size_formatted_scene)
                info['tasks'].append(task)
                info['pos'].append(green_pos)
                info['action'].append(act_at_interact)
                info['deltas'].append([xdelta, ydelta])

                if show:
                    print(starty, startx, width)
                    plt.imshow(phyre.observations_to_uint8_rgb(res.images[i_step]))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(channel_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*channel_formatted_scene], axis=1))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(size_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*size_formatted_scene], axis=1))
                    plt.show()

            if tries > max_tries:
                break

    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/interactions.pickle', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{base_path}/info.pickle', 'wb') as fp:
        pickle.dump(info, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting interactions!")


def collect_fullsize_interactions(save_path, tasks, number_per_task, stride=1, show=False):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    # base_path = 'data/fiddeling'
    data = []
    info = {'tasks': [], 'pos': [], 'action': []}
    print("Amount per task", number_per_task)

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task) == 1]
            print(f"collecting {n_collected + 1} interactions from {task} with {tries} tries", end=end_char)
            if len(action) == 0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                                      need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                # print(res.images.shape)
                # print(len(res.bitmap_seq))
                # print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                # print(idx1, idx2)
                # print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx, red_idx]]) / 2
                for i, m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        pos = res.featurized_objects.features[i, [green_idx, red_idx], :2]
                        dist = np.linalg.norm(pos[1] - pos[0])
                        # print(dist, target_dist)
                        if not dist < target_dist + 0.005:
                            continue

                        # print(res.featurized_objects.diameters[[green_idx,red_idx]])
                        # print(res.featurized_objects.features[i,green_idx])
                        # print(res.featurized_objects.features[i, red_idx])
                        # print(i+2)
                        # for i, scene in enumerate(res.images):
                        #    img = phyre.observations_to_uint8_rgb(scene)
                        #    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
                        #    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                        #    cv2.imwrite(path_str+f"/{str(i)}.jpg", img[:,:,::-1])

                        red_radius = res.featurized_objects.diameters[red_idx] * 4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i + 1, pos[0], action_at_interaction, target_dist)

                return (False, 0, (0, 0), 0)

            contact, i_step, green_pos, red_pos, summed_radii = check_contact(res)
            if contact:
                tries = 0
                n_collected += 1

                step_size = 20
                # check whether contact happend too early
                if i_step - step_size < 0:
                    continue

                selected_rollout = np.array([[(scene == ch).astype(float) for ch in range(1, 7)] for scene in
                                             res.images[i_step - step_size:i_step + step_size + 1:step_size]])
                # selected_rollout = np.flip(selected_rollout, axis=2)
                # print(selected_rollout.shape)

                # Formatting and resizing
                se = selected_rollout
                channel_formatted_scene = np.stack(
                    (se[0, 1], se[1, 1], se[2, 1], np.max(se[1, 2:], axis=0), se[0, 0], se[1, 0]))

                # saving extracted scene
                data.append(channel_formatted_scene)
                info['tasks'].append(task)
                info['pos'].append(green_pos)
                info['action'].append(red_pos)

                if show:
                    plt.imshow(phyre.observations_to_uint8_rgb(res.images[i_step]))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(channel_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*channel_formatted_scene], axis=1))
                    plt.show()
                    fig, ax = plt.subplots(1, 6, sharex=True, sharey=True)
                    for i, img in enumerate(channel_formatted_scene):
                        ax[i].imshow(img)
                    # plt.imshow(np.concatenate([*size_formatted_scene], axis=1))
                    plt.show()

            if tries > max_tries:
                break

    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/interactions.pickle', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{base_path}/info.pickle', 'wb') as fp:
        pickle.dump(info, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting interactions!")


def visualize_interactions(save_path, tasks, number_per_task, stride=1):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    # base_path = 'data/fiddeling'
    data = []
    info = {'tasks': [], 'pos': [], 'action': []}
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)

    print("Amount per task", number_per_task)

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task) == 1]
            print(f"collecting {n_collected + 1} interactions from {task} with {tries} tries", end=end_char)
            if len(action) == 0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                                      need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                # print(res.images.shape)
                # print(len(res.bitmap_seq))
                # print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                # print(idx1, idx2)
                # print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx, red_idx]]) / 2
                for i, m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        pos = res.featurized_objects.features[i, [green_idx, red_idx], :2]
                        dist = np.linalg.norm(pos[1] - pos[0])
                        # print(dist, target_dist)
                        if not dist < target_dist + 0.005:
                            continue

                        # print(res.featurized_objects.diameters[[green_idx,red_idx]])
                        # print(res.featurized_objects.features[i,green_idx])
                        # print(res.featurized_objects.features[i, red_idx])
                        # print(i+2)
                        # for i, scene in enumerate(res.images):
                        #    img = phyre.observations_to_uint8_rgb(scene)
                        #    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
                        #    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                        #    cv2.imwrite(path_str+f"/{str(i)}.jpg", img[:,:,::-1])

                        red_radius = res.featurized_objects.diameters[red_idx] * 4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i + 1, pos[0], action_at_interaction, target_dist)

                return (False, 0, (0, 0), 0)

            contact, i_step, green_pos, red_pos, summed_radii = check_contact(res)
            if contact:
                green_idx = res.featurized_objects.colors.index('GREEN')
                tries = 0
                step_range = 20
                # check whether contact happend too early
                if i_step - step_range < 0:
                    continue
                n_collected += 1

                zero_pos = res.featurized_objects.features[i_step, green_idx, :2]
                for j in range(i_step - step_range, i_step + step_range + 1, stride):
                    pos = res.featurized_objects.features[j, green_idx, :2]
                    delta = pos - zero_pos
                    scene = res.images[j]
                    img = Image.fromarray(phyre.observations_to_uint8_rgb(scene))
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0), f"{j - i_step} {tuple(delta * 256)}", (15, 15, 15), font=font)

                    os.makedirs(base_path, exist_ok=True)
                    img.save(base_path + f'/{task}_{n_collected}_{j}.png')

            if tries > max_tries:
                break

    print(f"FINISH collecting interaction visualizations!")


def collect_gridded_observations(path, n_per_task=10):
    tries = 0
    tasks = ['00012:002', '00011:004', '00008:062', '00002:047']
    base_path = path
    number_to_solve = n_per_task

    for task in tasks:
        sim = phyre.initialize_simulator([task], 'ball')
        r = 0.2
        # Gridding:
        for (x, y) in [(x, y) for x in np.linspace(0.1, 0.9, 10) for y in np.linspace(0.1, 0.9, 10)]:
            tries = 0
            while tries < 20:
                tries += 1
                action = [x + (np.random.rand() - 0.5) * 0.1, y + (np.random.rand() - 0.5) * 0.1, r]
                res = sim.simulate_action(0, action,
                                          need_featurized_objects=True, stride=15)
                if not res.status.is_invalid():
                    break
            if res.status.is_invalid():
                continue
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/{x}_{y}_{r}"
            pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
            with open(path_str + "/observations.pickle", 'wb') as handle:
                pickle.dump(res.images, handle, protocol=pickle.HIGHEST_PROTOCOL)


def collect_all_observations(path, n_per_task=10):
    tries = 0
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
             '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
             '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
             '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
             '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']
    tasks = [f'000{"0" + str(t) if t < 10 else t}:0{"0" + str(v) if v < 10 else v}' for t in range(2, 100) for v in
             range(100)]
    # tasks = ['00000:001']

    base_path = path
    number_to_solve = n_per_task

    for task in tasks:
        print("trying", task)
        try:
            sim = phyre.initialize_simulator([task], 'ball')
        except Exception:
            continue
        solved = 0
        while solved < number_to_solve:
            tries += 1
            action = sim.sample()
            res = sim.simulate_action(0, action,
                                      need_featurized_objects=True, stride=20)
            if res.status.is_solved():
                print("solved " + task + " with", tries, "tries")
                tries = 0
                solved += 1
                path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
                pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                with open(path_str + "/observations.pickle", 'wb') as handle:
                    pickle.dump(res.images, handle, protocol=pickle.HIGHEST_PROTOCOL)


def collect_base_observations(path):
    tries = 0
    tasks = [f'000{"0" + str(t) if t < 10 else t}:0{"0" + str(v) if v < 10 else v}' for t in range(0, 25) for v in
             range(100)]
    # tasks = ['00000:001']
    base_path = path

    for task in tasks:
        print("trying", task)
        try:
            sim = phyre.initialize_simulator([task], 'ball')
        except Exception:
            continue
        print("running", task)
        action = sim.sample()
        action[2] = 0.01
        res = sim.simulate_action(0, action,
                                  need_featurized_objects=True, stride=20)
        path_str = f"{base_path}/{task[:5]}/{task[6:]}/base"
        pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
        with open(path_str + "/observations.pickle", 'wb') as handle:
            pickle.dump(res.images, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_available_tasks():
    tasks = [f'000{"0" + str(t) if t < 10 else t}:0{"0" + str(v) if v < 10 else v}' for t in range(0, 25) for v in
             range(100)]
    # tasks = ['00000:001']

    available_tasks = []

    for task in tasks:
        print("trying", task)
        try:
            sim = phyre.initialize_simulator([task], 'ball')
            available_tasks.append(task)
        except Exception:
            continue
    print(available_tasks)
    json.dump(available_tasks, open("most_tasks.txt", 'w'))


def load_phyre_rollouts(path, image=False, base=True):
    s = "/"
    for task in os.listdir(path):
        for variation in os.listdir(path + s + task):
            tmp = []
            for trialfolder in os.listdir(path + s + task + s + variation):
                for fp in os.listdir(path + s + task + s + variation + s + trialfolder):
                    if fp == "observations.pickle":
                        final_path = path + s + task + s + variation + s + trialfolder + s + fp
                        if image:
                            yield (Image.open(final_path))
                        else:
                            with open(final_path, 'rb') as handle:
                                if base and trialfolder == "base":
                                    tmp.insert(0, pickle.load(handle))
                                else:
                                    tmp.append(pickle.load(handle))
            yield (tmp)


class Vertex:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord
        self.d = float('inf')  # distance from source
        self.parent_x = None
        self.parent_y = None
        self.processed = False
        self.index_in_queue = None


# Return neighbor directly above, below, right, and left
def get_neighbors(mat, r, c):
    shape = mat.shape
    neighbors = []
    # ensure neighbors are within image boundaries
    if r > 0 and not mat[r - 1][c].processed:
        neighbors.append(mat[r - 1][c])
    if r < shape[0] - 1 and not mat[r + 1][c].processed:
        neighbors.append(mat[r + 1][c])
    if c > 0 and not mat[r][c - 1].processed:
        neighbors.append(mat[r][c - 1])
    if c < shape[1] - 1 and not mat[r][c + 1].processed:
        neighbors.append(mat[r][c + 1])
    return neighbors


def bubble_up(queue, index):
    if index <= 0:
        return queue
    p_index = (index - 1) // 2
    if queue[index].d < queue[p_index].d:
        queue[index], queue[p_index] = queue[p_index], queue[index]
        queue[index].index_in_queue = index
        queue[p_index].index_in_queue = p_index
        quque = bubble_up(queue, p_index)
    return queue


def bubble_down(queue, index):
    length = len(queue)
    lc_index = 2 * index + 1
    rc_index = lc_index + 1
    if lc_index >= length:
        return queue
    if lc_index < length and rc_index >= length:  # just left child
        if queue[index].d > queue[lc_index].d:
            queue[index], queue[lc_index] = queue[lc_index], queue[index]
            queue[index].index_in_queue = index
            queue[lc_index].index_in_queue = lc_index
            queue = bubble_down(queue, lc_index)
    else:
        small = lc_index
        if queue[lc_index].d > queue[rc_index].d:
            small = rc_index
        if queue[small].d < queue[index].d:
            queue[index], queue[small] = queue[small], queue[index]
            queue[index].index_in_queue = index
            queue[small].index_in_queue = small
            queue = bubble_down(queue, small)
    return queue


def get_distance(img, u, v):
    return 0.1 + (float(img[v][0]) - float(img[u][0])) ** 2 + (float(img[v][1]) - float(img[u][1])) ** 2 + (
            float(img[v][2]) - float(img[u][2])) ** 2


def get_distance_obj(img, u, v, obj):
    if obj[u] and obj[v]:
        # return 0.0 + (float(img[v][0]) - float(img[u][0])) ** 2 + (float(img[v][1]) - float(img[u][1])) ** 2 + (float(img[v][2]) - float(img[u][2])) ** 2
        return 0.0
    else:
        # return 0.1 + (float(img[v][0]) - float(img[u][0])) ** 2 + (float(img[v][1]) - float(img[u][1])) ** 2 + (float(img[v][2]) - float(img[u][2])) ** 2
        return 1 if (img[v] == img[u]).all() or obj[u] else 1000


def drawPath(img, path, thickness=2):
    '''path is a list of (x,y) tuples'''
    x0, y0 = path[0]
    for vertex in path[1:]:
        x1, y1 = vertex
        cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), thickness)
        x0, y0 = vertex


def find_shortest_path(img, src, dst):
    pq = []  # min-heap priority queue
    source_x = src[0]
    source_y = src[1]
    dest_x = dst[0]
    dest_y = dst[1]
    imagerows, imagecols = img.shape[0], img.shape[1]
    matrix = np.full((imagerows, imagecols), None)  # access by matrix[row][col]
    for r in range(imagerows):
        for c in range(imagecols):
            matrix[r][c] = Vertex(c, r)
            matrix[r][c].index_in_queue = len(pq)
            pq.append(matrix[r][c])
    matrix[source_y][source_x].d = 0
    pq = bubble_up(pq, matrix[source_y][source_x].index_in_queue)

    while len(pq) > 0:
        u = pq[0]
        u.processed = True
        pq[0] = pq[-1]
        pq[0].index_in_queue = 0
        pq.pop()
        pq = bubble_down(pq, 0)
        neighbors = get_neighbors(matrix, u.y, u.x)
        for v in neighbors:
            dist = get_distance(img, (u.y, u.x), (v.y, v.x))
            if u.d + dist < v.d:
                v.d = u.d + dist
                v.parent_x = u.x
                v.parent_y = u.y
                idx = v.index_in_queue
                pq = bubble_down(pq, idx)
                pq = bubble_up(pq, idx)

    path = []
    iter_v = matrix[dest_y][dest_x]
    path.append((dest_x, dest_y))
    while (iter_v.y != source_y or iter_v.x != source_x):
        path.append((iter_v.x, iter_v.y))
        iter_v = matrix[iter_v.parent_y][iter_v.parent_x]

    path.append((source_x, source_y))
    return path


def find_distance_map(img, src):
    pq = []  # min-heap priority queue
    source_x = src[0]
    source_y = src[1]
    imagerows, imagecols = img.shape[0], img.shape[1]
    matrix = np.full((imagerows, imagecols), None)  # access by matrix[row][col]
    for r in range(imagerows):
        for c in range(imagecols):
            matrix[r][c] = Vertex(c, r)
            matrix[r][c].index_in_queue = len(pq)
            pq.append(matrix[r][c])
    matrix[source_y][source_x].d = 0
    pq = bubble_up(pq, matrix[source_y][source_x].index_in_queue)

    while len(pq) > 0:
        u = pq[0]
        u.processed = True
        pq[0] = pq[-1]
        pq[0].index_in_queue = 0
        pq.pop()
        pq = bubble_down(pq, 0)
        neighbors = get_neighbors(matrix, u.y, u.x)
        for v in neighbors:
            dist = get_distance(img, (u.y, u.x), (v.y, v.x))
            if u.d + dist < v.d:
                v.d = u.d + dist
                v.parent_x = u.x
                v.parent_y = u.y
                idx = v.index_in_queue
                pq = bubble_down(pq, idx)
                pq = bubble_up(pq, idx)

    distance_map = np.ones((imagerows, imagecols)) * 256.0  # access by matrix[row][col]
    for r in range(imagerows):
        for c in range(imagecols):
            distance_map[r][c] = float(matrix[r][c].d)
    trg_norm = True
    if trg_norm:
        dd = distance_map.copy()
        dd[dd > 255.] = 0.  # numpy.unique(np.array(dd, dtype=int))
        dmax = np.max(dd)
        # print('dmax =', dmax)
        distance_map = distance_map / dmax * 255.
    distance_map[distance_map > 255.] = 255.
    distance_map = 255. - distance_map
    return distance_map


def find_distance_map_obj(img, obj, trg_norm=False, len_norm=True):
    pq = []  # min-heap priority queue
    src = np.transpose(np.where(obj))
    source_y = src[0][0]
    source_x = src[0][1]
    imagerows, imagecols = img.shape[0], img.shape[1]
    matrix = np.full((imagerows, imagecols), None)  # access by matrix[row][col]
    for r in range(imagerows):
        for c in range(imagecols):
            matrix[r][c] = Vertex(c, r)
            matrix[r][c].index_in_queue = len(pq)
            pq.append(matrix[r][c])
    matrix[source_y][source_x].d = 0
    pq = bubble_up(pq, matrix[source_y][source_x].index_in_queue)

    while len(pq) > 0:
        u = pq[0]
        u.processed = True
        pq[0] = pq[-1]
        pq[0].index_in_queue = 0
        pq.pop()
        pq = bubble_down(pq, 0)
        neighbors = get_neighbors(matrix, u.y, u.x)
        for v in neighbors:
            # dist = get_distance(img, (u.y, u.x), (v.y, v.x))
            dist = get_distance_obj(img, (u.y, u.x), (v.y, v.x), obj)
            if u.d + dist < v.d:
                v.d = u.d + dist
                v.parent_x = u.x
                v.parent_y = u.y
                idx = v.index_in_queue
                pq = bubble_down(pq, idx)
                pq = bubble_up(pq, idx)

    distance_map = np.ones((imagerows, imagecols)) * 255.0  # access by matrix[row][col]
    for r in range(imagerows):
        for c in range(imagecols):
            distance_map[r][c] = float(matrix[r][c].d)

    if trg_norm:
        dd = distance_map.copy()
        dd[dd > 255.] = 255.  # numpy.unique(np.array(dd, dtype=int))
        dmax = np.max(dd)
        print('dmax =', dmax)
        distance_map = distance_map / dmax * 255.

    if len_norm:
        distance_map = 255 * distance_map / (img.shape[0] * 2)

    distance_map[distance_map > 255.] = 255.
    distance_map = 255. - distance_map
    return distance_map


# def make_mono_dataset(path, size=(32, 32), tasks=[], batch_size=32, solving=True, n_per_task=1, shuffle=True,
#                       proposal_dict=None, dijkstra=False, pertempl=False):
#     if os.path.exists(path + "/data.pickle") and os.path.exists(path + "/index.pickle"):
#         try:
#             with gzip.open(path + '/data.pickle', 'rb') as fp:
#                 X, Y = pickle.load(fp)
#                 X = torch.tensor(X).float()
#                 Y = torch.tensor(Y).float()
#         except OSError as e:
#             print("WARNING still unzipped data file at", path)
#             with open(path + '/data.pickle', 'rb') as fp:
#                 X, Y = pickle.load(fp)
#                 X = torch.tensor(X).float()
#                 Y = torch.tensor(Y).float()
#         with open(path + '/index.pickle', 'rb') as fp:
#             index = pickle.load(fp)
#
#         # TRAIN TEST SPLIT
#         print(f"Loaded dataset from {path} with shape:", X.shape)
#     else:
#         if proposal_dict is None:
#             train_ids, dev_ids, test_ids = phyre.get_fold("ball_within_template", 0)
#             all_tasks = train_ids + dev_ids + test_ids
#         else:
#             all_tasks = tasks
#         collect_solving_dataset(path, all_tasks, n_per_task=n_per_task, stride=5, size=size, solving=solving,
#                                 proposal_dict=proposal_dict, dijkstra=dijkstra, pertempl=pertempl)
#         with gzip.open(path + '/data.pickle', 'rb') as fp:
#             X, Y = pickle.load(fp)
#         with open(path + '/index.pickle', 'rb') as fp:
#             index = pickle.load(fp)
#         X = torch.tensor(X).float()
#         Y = torch.tensor(Y).float()
#         print(f"Loaded dataset from {path} with shape:", X.shape)
#
#     # MAKE CORRECT SELECTION
#     selection = [i for (i, task) in enumerate(index) if task in tasks]
#     # print(len(index), len(tasks), len(selection))
#     X = X[selection]
#     Y = Y[selection]
#     index = [index[s] for s in selection]
#     # L.info(f"Loaded dataset from {path} with shape: {X.shape}")
#
#     assert len(X) == len(Y) == len(index), "All should be of equal length"
#
#     X = X / 255  # correct for uint8 encoding
#     I = torch.arange(len(X), dtype=int)
#     dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, I), batch_size, shuffle=shuffle)
#     return dataloader, index


def collect_solving_dataset_img(path, tasks, n_per_task=10, stride=10, size=(32, 32), solving=True, exclude_tasks=()):
    os.makedirs(path, exist_ok=True)
    end_char = '\r'
    tries = 0
    max_tries = 510
    number_to_solve = n_per_task
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array

    data = []
    acts = []
    lib_dict = dict()
    task_list = []

    episode = 0
    sim = phyre.initialize_simulator(tasks, 'ball')
    for task_idx, task in enumerate(tasks):
        if int(task.split(':')[0]) in exclude_tasks:
            print(f'skipping task: {task}')
            continue
        # COLLECT SOLVES
        solved = 0
        cache_list = actions[cache.load_simulation_states(task) == (1 if solving else -1)]
        while solved < number_to_solve:
            print(f"collecting {task}: trial {solved} with {tries + 1} tries, ep: {episode}", end=end_char)
            tries += 1
            actionlist = cache_list

            if len(actionlist) == 0:
                print("WARNING no solution action in cache at task", task)
                actionlist = [np.random.rand(3)]
            action = random.choice(actionlist)
            res = sim.simulate_action(task_idx, action,
                                      need_featurized_objects=True, stride=stride)

            # IF SOLVED PROCESS ROLLOUT
            if (res.status.is_solved() == solving) and not res.status.is_invalid():
                # save
                ep_dir = os.path.join(path, f'{episode:05d}')
                os.makedirs(ep_dir, exist_ok=True)
                # save images
                for k, im in enumerate(res.images):
                    img = phyre.observations_to_float_rgb(im)
                    img = (255 * img).astype(np.uint8)
                    img = cv2.resize(img, size, cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(ep_dir, f'{k:05d}.png'), img)
                acts.append(action)
                tries = 0
                solved += 1
                episode += 1

                # FORMAT AND EXTRACT DATA
                # paths = np.zeros((len(path_idxs), len(res.images), size[0], size[1]))
                # alpha, gamma = 1, 1
                # for i, image in enumerate(res.images):
                # extract color codings from channels
                # chans = np.array([(image == ch).astype(float) for ch in channels])
                #
                # # at first frame extract init scene
                # if not i:
                #     init_scene = np.array([(cv2.resize(chans[ch], size, cv2.INTER_MAX) > 0).astype(float) for ch in
                #                            range(len(channels))])
                #
                # # add path_idxs channels to paths
                # for path_i, idx in enumerate(path_idxs):
                #     paths[path_i, i] = alpha * (cv2.resize(chans[idx], size, cv2.INTER_MAX) > 0).astype(float)
                # alpha *= gamma

                # COLLECT BASE
                # print(f"collecting {task}: base", end=end_char)
                # 1000 tries make sure one action is valid
                # for _ in range(1000):
                #     action = sim.sample()
                #     action[2] = 0.001
                #     res = sim.simulate_action(task_idx, action,
                #                               need_featurized_objects=False, stride=stride)
                #     if not res.status.is_invalid():
                #         break
                # base_frames = \
                #     np.array([cv2.resize((scene == 2).astype(float), size, cv2.INTER_MAX) for scene in res.images])[
                #         None]

                # combine channels
                # flip y axis and concat init scene with paths and base
                # paths = np.flip(np.max(paths, axis=1).astype(float), axis=1)
                # base = np.flip(np.max(base_frames, axis=1).astype(float), axis=1)
                # init_scene = np.flip(init_scene, axis=1)

                # make distance map
                # if dijkstra:
                #     dm_init_scene = sim.initial_scenes[task_idx]
                #     img = cv2.resize(phyre.observations_to_float_rgb(dm_init_scene), size, cv2.INTER_MAX)  # read image
                #     target = np.logical_or(init_scene[2] == 1, init_scene[3] == 1)
                #     # cv2.imwrite('maze-initial.png', img)
                #     distance_map = find_distance_map_obj(img, target) / 255
                #     combined = (255 * np.concatenate([init_scene, base, paths, distance_map[None]])).astype(np.uint8)
                # else:
                # combined = (255 * base_frames).astype(np.uint8)

                # append data set and lib_dict
                # data.append(combined)
                task_list.append(task)
                if task in lib_dict:
                    lib_dict[task].append(len(data) - 1)
                else:
                    lib_dict[task] = [len(data) - 1]

            if tries > max_tries:
                break

    file = gzip.GzipFile(path + '/data.pickle', 'wb')
    pickle.dump(acts, file)
    file.close()
    with open(path + '/index.pickle', 'wb') as fp:
        pickle.dump(task_list, fp)

    print(f"FINISH collecting {'solving' if solving else 'failing'} dataset!")


def create_phyre_dataset(path, size=(128, 128), solving=True, n_per_task=1, setup="all-tasks", fold=0, stride=25,
                         exclude_tasks=()):
    available_setups = ['ball_within_template', 'ball_cross_template', 'two_balls_cross_template',
                        'two_balls_within_template']
    assert setup in ['all-tasks', 'ball_within_template', 'ball_cross_template', 'two_balls_cross_template',
                     'two_balls_within_template']
    fold_id = fold
    eval_setup = setup

    path_train = os.path.join(path, 'train')
    os.makedirs(path_train, exist_ok=True)
    path_valid = os.path.join(path, 'valid')
    os.makedirs(path_valid, exist_ok=True)
    path_test = os.path.join(path, 'test')
    os.makedirs(path_test, exist_ok=True)

    if setup == 'all-tasks':
        for eval_setup in available_setups:
            train_ids, val_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
            # train
            print(f'{eval_setup}: collection train data...')
            collect_solving_dataset_img(path_train, train_ids, n_per_task=n_per_task, stride=stride, size=size,
                                        solving=solving, exclude_tasks=exclude_tasks)
            # valid
            print(f'{eval_setup}: collection validation data...')
            collect_solving_dataset_img(path_valid, val_ids, n_per_task=1, stride=stride, size=size,
                                        solving=solving, exclude_tasks=exclude_tasks)

            # test
            print(f'{eval_setup}: collection test data...')
            collect_solving_dataset_img(path_test, test_ids, n_per_task=1, stride=stride, size=size,
                                        solving=solving, exclude_tasks=exclude_tasks)
    else:
        train_ids, val_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
        # train
        print(f'{eval_setup}: collection train data...')
        collect_solving_dataset_img(path_train, train_ids, n_per_task=n_per_task, stride=stride, size=size,
                                    solving=solving, exclude_tasks=exclude_tasks)
        # valid
        print(f'{eval_setup}: collection validation data...')
        collect_solving_dataset_img(path_valid, val_ids, n_per_task=1, stride=stride, size=size, solving=solving,
                                    exclude_tasks=exclude_tasks)

        # test
        print(f'{eval_setup}: collection test data...')
        collect_solving_dataset_img(path_test, test_ids, n_per_task=1, stride=stride, size=size, solving=solving,
                                    exclude_tasks=exclude_tasks)


if __name__ == "__main__":
    path = './'
    setup = 'ball_within_template'
    n_per_task = 2
    # n_per_task = 10
    # stride = 6  # original
    stride = 7
    # exclude_tasks = []
    exclude_tasks = [12, 13, 16, 20, 21]  # full
    # exclude_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]  # subset
    create_phyre_dataset(path, size=(128, 128), solving=True, n_per_task=n_per_task, setup=setup, fold=0, stride=stride,
                         exclude_tasks=exclude_tasks)
