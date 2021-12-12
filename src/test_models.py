from models import MarkovModel, Model1
import itertools
import pandas as pd
import numpy as np

train_dict = {
    "zone_id": [1, 1, 2, 3, 5, 3, 3, 3, 3],
    "zone_id_dest": [1, 2, 2, 99, 99, 99, 5, 99, 5],
    "possession_outcome": [1, 0, 0, 1, 0, 1, 0, 1, 1],
    "throw_outcome": [1, 1, 1, 1, 0, 1, 0, 1, 1],
    "player_cluster_id": [8, 8, 8, 9, 8, 8, 9, 8, 9],
    "player_cluster_id_dest": [8, 9, 8, 9, 8, 8, 9, 8, 9],
}
train_data = pd.DataFrame().from_dict(train_dict)

test_dict = {
    "zone_id": [5, 2, 3, 3, 4],
    "player_cluster_id": [8, 9, 8, 9, 7],
}
test_data = pd.DataFrame().from_dict(test_dict)

expected_p_trans = {
    1: [0.5, 0.5, 0, 0, 0],
    2: [0, 1, 0, 0, 0],
    3: [0.0, 0.0, 0.0, 0.4, 0.6],
    5: [0, 0, 0, 0, 1],
    99: [0, 0, 0, 0, 0],
}

expected_p_success = {
    1: [1, 1, 0, 0, 0],
    2: [0, 1, 0, 0, 0],
    3: [0.0, 0.0, 0.0, 0.5, 1.0],
    5: [0, 0, 0, 0, 0],
    99: [0, 0, 0, 0, 0],
}
# def test_single_feature():
#     m1 = Model1()

#     # One feature
#     m1.fit(train_data, ["zone_id"], "possession_outcome")
#     assert np.allclose(m1.params.values, [0.5, 0.0, 0.75, 1.0, 1.0])
#     assert np.allclose(
#         m1.predict(test_data["zone_id"]).values,
#         np.array([1, 0, 0.66666667, 0.66666667, 0.63333333]),
#     )


# def test_multi_feature():
#     m1 = Model1()
#     # Two features
#     m1.fit(train_data, ["zone_id", "player_cluster_id"], "possession_outcome")
#     assert np.allclose(m1.params.values, [0.5, 0.0, 0.5, 1.0, 1.0, 1.0])
#     # assert np.allclose(
#     #     m1.predict(test_data[["zone_id", "player_cluster_id"]]),
#     #     [1, 0.6, 0.5, 1, 0.6],
#     # )


def test_markov_single():
    mm = MarkovModel()
    mm.fit(
        train_data,
        ["zone_id"],
        ["zone_id_dest"],
        "throw_outcome",
        states=[1, 2, 3, 5, 99],
    )
    print(mm.p_success)
    for k, arr in expected_p_trans.items():
        assert np.allclose(mm.p_trans[mm.states[k]], arr)
        assert np.allclose(mm.p_success[mm.states[k]], expected_p_success[k])


def test_markov_multiple():
    mm = MarkovModel()

    states = list(itertools.product([1, 2, 3, 5, 99], [8, 9]))

    mm.fit(
        train_data,
        ["zone_id", "player_cluster_id"],
        ["zone_id_dest", "player_cluster_id_dest"],
        "throw_outcome",
        states=states,
        priors=(
            np.ones((len(states), len(states))),
            np.ones((len(states), len(states))) * 2,
        ),
    )

    assert len(mm.states) == 10
    # print(mm.states)
    # print(mm.p_trans)

    print(mm.sample_possession(starting_state=(1, 8), n=10))
