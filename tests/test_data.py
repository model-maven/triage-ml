from triageml.data import load_dataframe, load_split


def test_dataframe_has_required_columns():
    df = load_dataframe()
    assert {"text", "label"}.issubset(df.columns)
    assert len(df) > 0


def test_split_is_stratified_and_disjoint():
    data = load_split(test_size=0.25, random_state=0)
    assert data.n_train > 0 and data.n_test > 0
    # Train and test should not overlap on index-identical rows by construction.
    assert set(data.y_train) == set(data.y_test)  # all classes present in both
