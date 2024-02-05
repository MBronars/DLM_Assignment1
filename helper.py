import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import imageio
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.tensor_utils as TensorUtils
from copy import deepcopy
import os
from collections import deque
import time


np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def playback_trajectory(env, video_writer, demo_key):
    """
    Simple helper function to playback the trajectory stored under the hdf5 group @demo_key and
    write frames rendered from the simulation to the active @video_writer.
    """

    # robosuite datasets store the ground-truth simulator states under the "states" key.
    # We will use the first one, alone with the model xml, to reset the environment to
    # the initial configuration before playing back actions.
    init_state = f["data/{}/states".format(demo_key)][0]
    model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
    initial_state_dict = dict(states=init_state, model=model_xml)

    # reset to initial state
    env.reset_to(initial_state_dict)

    # playback actions one by one, and render frames
    actions = f["data/{}/actions".format(demo_key)][:]
    for t in range(actions.shape[0]):
        env.step(actions[t])
        video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
        video_writer.append_data(video_img)

def playback_demos(video_path, dataset_path, num_rollouts):
  video_writer = imageio.get_writer(video_path, fps=20)
  # create simulation environment from environment metedata=
  env = EnvUtils.create_env_from_metadata(
      env_meta= FileUtils.get_env_metadata_from_dataset(dataset_path),
      render=False,            # no on-screen rendering
      render_offscreen=True,   # off-screen rendering to support rendering video frames
  )

  f = h5py.File(dataset_path, "r")

  # each demonstration is a group under "data".  each demonstration is named "demo_#" where # is a number, starting from 0
  demos = list(f["data"].keys())

  # playback the first 5 demos
  for demo_key in demos[:5]:
      print("Playing back demo key: {}".format(demo_key))
      init_state = f["data/{}/states".format(demo_key)][0]
      model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
      initial_state_dict = dict(states=init_state, model=model_xml)
      # reset to initial state
      env.reset_to(initial_state_dict)

      # playback actions one by one, and render frames
      actions = f["data/{}/actions".format(demo_key)][:]
      for t in range(actions.shape[0]):
          env.step(actions[t])
          video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
          video_writer.append_data(video_img)

  # playback the first 5 demos
#   for demo_key in demos[:5]:
#       print("Playing back demo key: {}".format(demo_key))
      
#       start_time = time.time()
      
#       init_state = f["data/{}/states".format(demo_key)][0]
#       model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
#       initial_state_dict = dict(states=init_state, model=model_xml)
      
#       # reset to initial state
#       env.reset_to(initial_state_dict)
      
#       reset_time = time.time()
#       print("Time taken for resetting environment: {:.2f} seconds".format(reset_time - start_time))
      
#       # playback actions one by one, and render frames
#       actions = f["data/{}/actions".format(demo_key)][:]
#       for t in range(actions.shape[0]):
#           action_start_time = time.time()
          
#           env.step(actions[t])
          
#           step_time = time.time()
#           print("Time taken for step {}: {:.2f} seconds".format(t, step_time - action_start_time))
          
#           video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
          
#           render_time = time.time()
#           print("Time taken to render frame {}: {:.2f} seconds".format(t, render_time - step_time))
          
#           video_writer.append_data(video_img)
      
#       total_time = time.time()
#       print("Total time for demo key {}: {:.2f} seconds".format(demo_key, total_time - start_time))

  # done writing video
  video_writer.close()

def rollout(
    policy, 
    dataset_path, 
    horizon, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None, 
    obs_keys = None, 
    obs_len = 1, 
    num_rollouts = 10):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.
    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
    """
    env = EnvUtils.create_env_from_metadata(
      env_meta= FileUtils.get_env_metadata_from_dataset(dataset_path),
      render=False,            # no on-screen rendering
      render_offscreen=True,   # off-screen rendering to support rendering video frames
    )

    full_success = 0
    for i in range(num_rollouts):
      policy.set_eval()
      policy.reset()
      obs = env.reset()
      state_dict = env.get_state()

      # hack that is necessary for robosuite tasks for deterministic action playback
      obs = env.reset_to(state_dict)

      ob = TensorUtils.to_tensor(obs)
      ob = TensorUtils.to_batch(ob)
      ob = TensorUtils.to_device(ob, device)
      ob = TensorUtils.to_float(ob)

      ob = torch.cat([value for key, value in ob.items() if key in obs_keys], dim=-1)

      # initialize action and state deques
      state_deque = deque([ob] * obs_len, maxlen = obs_len)
      action_deque = deque()

      results = {}
      video_count = 0  # video frame counter
      total_reward = 0
      try:
          for step_i in range(horizon):

              policy_obs = torch.stack(list(state_deque), dim=0)
              if obs_len > 1:
                policy_obs = torch.stack(list(state_deque), dim=1)

              if len(action_deque) == 0:
                # get action from policy
                act = policy.get_action(policy_obs)
                if  len(act.shape) == 3:
                  act = TensorUtils.to_numpy(act[0])
                else:
                  act = TensorUtils.to_numpy(act)
                for i in range(len(act)):
                  action_deque.append(act[i])

              act = action_deque.popleft()

              # play action
              next_obs, r, done, _ = env.step(act)

              # compute reward
              total_reward += r
              success = env.is_success()["task"]

              if video_writer is not None:
                  if video_count % video_skip == 0:
                      video_img = []
                      camera_names = ["agentview"]
                      for cam_name in camera_names:
                          video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                      video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                      video_writer.append_data(video_img)
                  video_count += 1

              # break if done or if success
              if done or success:
                  break

              # update for next iter
              obs = deepcopy(next_obs)
              state_dict = env.get_state()


              ob = TensorUtils.to_tensor(obs)
              ob = TensorUtils.to_batch(ob)
              ob = TensorUtils.to_device(ob, device)
              ob = TensorUtils.to_float(ob)

              policy_obs = torch.cat([value for key, value in ob.items() if key in obs_keys], dim=-1)
              state_deque.append(policy_obs)

      except env.rollout_exceptions as e:
          print("WARNING: got rollout exception {}".format(e))

      stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
      full_success += float(success)

    return full_success/num_rollouts

def process_batch(batch):
    """
    Process a batch of data.
    """
    input_batch = dict()
    # concat all the items in batch["obs"]
    input_batch["obs"] = torch.cat([value for value in batch['obs'].values()], dim=-1)

    input_batch["actions"] = batch["actions"]
    return TensorUtils.to_float(TensorUtils.to_device(input_batch, device))

def run_epoch(model, data_loader, validate = False):
    """
    Run a single epoch of training.
    """
    data_loader_iter = iter(data_loader)
    if validate:
        model.eval()
    else:
        model.train()
    total_loss = 0
    for batch_i, batch in enumerate(data_loader_iter):
        inputs = process_batch(batch)
        loss = model.train_on_batch(inputs, validate)
        total_loss += loss
    return total_loss / len(data_loader)

def train(model, train_loader, valid_loader = None, num_epochs = 100, save_path=None):
    """
    Train a model using the algorithm.
    """
    train_losses = []
    valid_losses = []
    for epoch_i in range(num_epochs):

        train_loss = run_epoch(model, train_loader)
        train_losses.append((train_loss, model.epoch))

        if valid_loader is not None:
            valid_loss = run_epoch(model, valid_loader, validate = True)
            valid_losses.append((valid_loss, model.epoch))
        if epoch_i % 10 == 0:
            print("Epoch: {} Train Loss: {} Valid Loss: {}".format(epoch_i, train_loss, valid_loss))
        if epoch_i % 50 == 0:
            model.save(os.path.join(save_path, "epoch_{}.pth".format(model.epoch)))
        model.epoch += 1
    return train_losses, valid_losses


def load_data_for_training(dataset_path, obs_keys, seq_len = 1, batch_size = 100, normalize = False, frame_stack = 1):
    """
    Load data for training.
    """
    train_set = SequenceDataset(
        hdf5_path = dataset_path,
        obs_keys = obs_keys,
        dataset_keys = ["actions"],
        load_next_obs = False,
        frame_stack = frame_stack,
        seq_length = seq_len,
        pad_frame_stack = True,
        pad_seq_length = True,
        get_pad_mask = False,
        goal_mode = None,
        hdf5_cache_mode = "all",
        hdf5_use_swmr = False,
        hdf5_normalize_obs = normalize,
        filter_by_attribute = 'train',
    )
    valid_set = SequenceDataset(
        hdf5_path = dataset_path,
        obs_keys = obs_keys,
        dataset_keys = ["actions"],
        load_next_obs = False,
        frame_stack = frame_stack,
        seq_length = seq_len,
        pad_frame_stack = True,
        pad_seq_length = True,
        get_pad_mask = False,
        goal_mode = None,
        hdf5_cache_mode = "all",
        hdf5_use_swmr = False,
        hdf5_normalize_obs = normalize,
        filter_by_attribute = 'valid',
    )

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
    valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True, num_workers = 0)

    train_batch = next(iter(train_loader))
    print("batch keys: {}".format(train_batch.keys()))
    print("observation shapes: ")
    for obs, obs_key in train_batch["obs"].items():
        print("{} shape: {}".format(obs , obs_key.shape))
    print("action shape: {}".format(train_batch['actions'].shape))

    return train_loader, valid_loader
