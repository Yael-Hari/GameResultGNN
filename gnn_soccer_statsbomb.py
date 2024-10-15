from tqdm import tqdm
import pandas as pd
import numpy as np
from math import floor
import pickle
from statsbombpy import sb
import math
import const

import warnings
warnings.filterwarnings('ignore')


def map_sector_to_node(sector_x, sector_y):
    return sector_x + (sector_y * const.NUM_WIDTH_SECTORS)


def gen_weighted_adjacency_matrices(game_passes_df, home_team, num_time_intervals=const.NUM_TIME_INTERVALS):
    """
    Generate weighted adjacency matrices for directed graph representation of a soccer match
    :param game_passes_df:
    :param home_team:
    :param num_time_intervals:
    :return:
    """
    width = math.ceil(max(game_passes_df['x'].max(), game_passes_df['pass_end_x'].max())) + 1
    height = math.ceil(max(game_passes_df['y'].max(), game_passes_df['pass_end_y'].max())) + 1
    time_interval = 45 / num_time_intervals

    # compute distances which separate sectors
    x_sep_dist = width / const.NUM_WIDTH_SECTORS
    y_sep_dist = height / const.NUM_HEIGHT_SECTORS

    # Determine number of time intervals for each half (0-44 minutes and 45-90 minutes)

    # create 3D adjacency matrices for both halves (time layers by sectors by sectors)
    adjacency_matrix_home_1st_half = np.zeros((num_time_intervals, const.NUM_SECTORS, const.NUM_SECTORS))
    adjacency_matrix_away_1st_half = np.zeros((num_time_intervals, const.NUM_SECTORS, const.NUM_SECTORS))
    adjacency_matrix_home_2nd_half = np.zeros((num_time_intervals, const.NUM_SECTORS, const.NUM_SECTORS))
    adjacency_matrix_away_2nd_half = np.zeros((num_time_intervals, const.NUM_SECTORS, const.NUM_SECTORS))

    # helper function to populate adjacency matrices
    def populate_adjacency_matrix(pass_data, adj_matrix_home, adj_matrix_away, half_start_minute):
        for ind, row_data in pass_data.iterrows():
            # unpack coordinate data
            x_i = row_data["x"]
            y_i = row_data["y"]
            x_f = row_data["pass_end_x"]
            y_f = row_data["pass_end_y"]

            # compute initial pass sector and terminal pass sector coordinates
            init_sector_x = floor(x_i / x_sep_dist)
            init_sector_y = floor(y_i / y_sep_dist)
            final_sector_x = floor(x_f / x_sep_dist)
            final_sector_y = floor(y_f / y_sep_dist)

            # map the initial pass sector and terminal pass sector to nodes
            init_sector_node = map_sector_to_node(init_sector_x, init_sector_y)
            final_sector_node = map_sector_to_node(final_sector_x, final_sector_y)

            # determine which time layer this pass belongs to
            minute = row_data['minute'] - half_start_minute
            minute = min(44, minute)
            time_layer = int(minute // time_interval)

            # update the appropriate adjacency matrix for the current time layer
            if row_data['possession_team'] == home_team:
                adj_matrix_home[time_layer, init_sector_node, final_sector_node] += 1
            else:
                adj_matrix_away[time_layer, init_sector_node, final_sector_node] += 1

    # Separate the passes into 1st half (0 to 44 minutes) and 2nd half (45 to 90 minutes)
    first_half_passes = game_passes_df[game_passes_df['minute'] <= 44]
    second_half_passes = game_passes_df[game_passes_df['minute'] >= 45]

    # populate matrices for 1st half
    populate_adjacency_matrix(first_half_passes, adjacency_matrix_home_1st_half, adjacency_matrix_away_1st_half, 0)

    # populate matrices for 2nd half
    populate_adjacency_matrix(second_half_passes, adjacency_matrix_home_2nd_half, adjacency_matrix_away_2nd_half, 45)

    return {
        'home_1st_half': adjacency_matrix_home_1st_half,
        'away_1st_half': adjacency_matrix_away_1st_half,
        'home_2nd_half': adjacency_matrix_home_2nd_half,
        'away_2nd_half': adjacency_matrix_away_2nd_half
    }


def create_data_matrices(seasons_df: pd.DataFrame) -> list:
    data_list = []
    # undirected_A_data_list = []

    for _, szn in tqdm(seasons_df.iterrows(), total=len(seasons_df), desc="Processing Season"):

        competition_id = szn["competition_id"]
        season_id = szn["season_id"]

        season_matches = sb.matches(competition_id=competition_id, season_id=season_id)
        matches_list = season_matches['match_id'].tolist()

        # Use Statsbomb API to return events from matches
        events_df = sb.events(match_id=matches_list[0])
        for mid in tqdm(matches_list[1:], total=len(matches_list)-1, desc="getting events"):
            events_df = pd.concat([events_df, sb.events(match_id=mid)], ignore_index=True)

        # Filter events to retain only regular play passes
        season_passes_df = events_df.loc[(events_df['type'] == 'Pass') & (events_df['play_pattern'] == 'Regular Play')]

        # Convert location data from arrays into coordinates
        season_passes_df['x'] = season_passes_df.apply(lambda n: n['location'][0], axis=1)
        season_passes_df['y'] = season_passes_df.apply(lambda n: n['location'][1], axis=1)
        season_passes_df['pass_end_x'] = season_passes_df.apply(lambda n: n['pass_end_location'][0], axis=1)
        season_passes_df['pass_end_y'] = season_passes_df.apply(lambda n: n['pass_end_location'][1], axis=1)

        # Filter columns to those relevant to passes df
        season_passes_df = season_passes_df[
            ['minute', 'duration', 'id', 'match_id', 'x', 'y', 'pass_end_x', 'pass_end_y', 'possession_team']
        ].reset_index(drop=True)

        # adding pertitent metadata to the passes df
        keep_cols = ['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score']
        season_merge_data_df = season_matches[keep_cols]
        season_passes_df = pd.merge(season_passes_df, season_merge_data_df, on='match_id', how='right')

        # gen pass adjacency matrices from each match
        match_id_list = list(set(season_passes_df["match_id"]))
        for match_id in tqdm(match_id_list, total=len(match_id_list), desc="Processing Matches"):
            game_passes_df = season_passes_df[season_passes_df['match_id'] == match_id]
            home_team = list(game_passes_df["home_team"])[0]
            home_score = list(game_passes_df["home_score"])[0]
            away_score = list(game_passes_df["away_score"])[0]
            adj_mats_dict = gen_weighted_adjacency_matrices(
                game_passes_df=game_passes_df,
                home_team=home_team,
            )
            if home_score == away_score:
                label = 0
            elif home_score > away_score:
                label = 1
            else:
                label = 2

            sample_data = [adj_mats_dict, label]
            data_list.append(sample_data)

    return data_list

def build_data(save_pkl_to_path='passes_data.pkl'):
    comps_df = sb.competitions()
    mask = comps_df.competition_gender.isin(['male'])
    seasons_df = comps_df[mask]

    data = create_data_matrices(seasons_df)

    with open(save_pkl_to_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {save_pkl_to_path}")


if __name__ == '__main__':

    pkl_path = f'passes_data_{const.NUM_TIME_INTERVALS}_time_intervals.pkl'

    # build data
    # build_data(save_pkl_to_path=pkl_path)

    # load data
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print()