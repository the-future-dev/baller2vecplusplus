import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import torch
import yaml

from paper_functions import gen_imgs_from_basketball_sample
from PIL import Image, ImageDraw
from settings import *
from torch import nn
from toy_dataset import ToyDataset
from train_baller2vecplusplus import init_basketball_datasets, init_model

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

home_dir = os.environ.get('PROJECT_DIR')

shuffle_keys = [
    "player_idxs",
    "player_xs",
    "player_ys",
    "player_hoop_sides",
    "player_x_diffs",
    "player_y_diffs",
    "player_trajs",
]

def simulate_offence_positions():
    """
    Define function to simulate "offence" positions given as input the defence data.
    """
    JOB = "20230927101037"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    samples = 5
    device = torch.device("cuda:0")
    os.makedirs(f"{home_dir}/results", exist_ok=True)

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, _) = init_basketball_datasets(opts)
    n_players = train_dataset.n_players
    player_traj_n = test_dataset.player_traj_n
    model = init_model(opts, train_dataset).to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
    pretrained_dict = {
        k: v for (k, v) in pretrained_dict.items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    model.eval() # Set model to evaluation mode
    seq_len = model.seq_len
    grid_gap = np.diff(test_dataset.player_traj_bins)[-1] / 2

    # Find valid test indices
    cand_test_idxs = []
    for test_idx in range(len(test_dataset.gameids)):
        tensors = test_dataset[test_idx]
        if len(tensors["player_idxs"]) == model.seq_len:
            cand_test_idxs.append(test_idx)

    # Extend player trajectory bins
    player_traj_bins = np.array(list(test_dataset.player_traj_bins) + [5.5])

    # Set seeds for reproducibility
    torch.manual_seed(2010)
    np.random.seed(2010)

    # Choose a test index to pick a game fraction inside the dataset.
    test_idx = 25
    tensors = test_dataset[test_idx]

    # Generate images from the real basketball sample
    (traj_imgs, traj_img) = gen_imgs_from_basketball_sample(tensors, seq_len)
    traj_imgs[0].save(
        f"{home_dir}/results/{test_idx}_truth.gif",
        save_all=True,
        append_images=traj_imgs[1:],
        duration=400,
        loop=0,
    )
    traj_img.save(
        f"{home_dir}/results/{test_idx}_truth.png",
    )

    # Loop through samples to generate and save images of the offence positions
    for sample in range(samples):
        tensors = test_dataset[test_idx] # the defence will remain the same, so we are copying it
        with torch.no_grad(): # Disable gradient computation
            for step in range(seq_len):
                preds_start = 10 * step
                for player_idx in range(n_players):
                    is_attacker = tensors['player_hoop_sides'][step, player_idx] == 1
                    if is_attacker:
                        # Predict the next positions for attackers
                        pred_idx = preds_start + player_idx
                        preds = model(tensors)[pred_idx]
                        probs = torch.softmax(preds, dim=0)
                        samp_traj = torch.multinomial(probs, 1)

                        # Get sample row and column
                        samp_row = samp_traj // player_traj_n
                        samp_col = samp_traj % player_traj_n

                        # Calculate sample x and y coordinates
                        samp_x = (
                            player_traj_bins[samp_col]
                            - grid_gap
                            + np.random.uniform(-grid_gap, grid_gap)
                        )
                        samp_y = (
                            player_traj_bins[samp_row]
                            - grid_gap
                            + np.random.uniform(-grid_gap, grid_gap)
                        )

                        # Update tensor values
                        tensors["player_x_diffs"][step, player_idx] = samp_x
                        tensors["player_y_diffs"][step, player_idx] = samp_y
                        if step < seq_len - 1:
                            tensors["player_xs"][step + 1, player_idx] = (
                                tensors["player_xs"][step, player_idx] + samp_x
                            )
                            tensors["player_ys"][step + 1, player_idx] = (
                                tensors["player_ys"][step, player_idx] + samp_y
                            )
        ###################################################################################################
        # Generate and save new images:                                                                   #
        # the defence will remain the same while the offence will be generated by baller2vec++            #
        # "tensors" contains the generated positions of the attackers where 'player_hoop_sides' == 1      #
        ###################################################################################################
        (traj_imgs, traj_img) = gen_imgs_from_basketball_sample(tensors, seq_len)
        traj_imgs[0].save(
            f"{home_dir}/results/{test_idx}_gen_baller2vec++_{sample}.gif",
            save_all=True,
            append_images=traj_imgs[1:],
            duration=400,
            loop=0,
        )
        traj_img.save(
            f"{home_dir}/results/{test_idx}_gen_baller2vec++_{sample}.png",
        )

    # Archive results and clean up
    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")


if __name__ == '__main__':
    simulate_offence_positions()
