from numbers import Number

import pandas as pd


def _sum_weights(df, source_weights):
    return (
        df
            .merge(source_weights, on="source", how="inner")[["from", "to", "weight"]]
            .groupby(["from", "to"])
            .aggregate("sum")
    )

def construct_weighted_networks(
    lr_network:pd.DataFrame,
    sig_network:pd.DataFrame,
    gr_network:pd.DataFrame,
    source_weights:dict[str, float]|pd.DataFrame,
    n_output_networks:int=2
) -> dict[str, pd.DataFrame]:
    if type(lr_network) is not pd.DataFrame:
        raise TypeError(f"lr_network should have type pandas.DataFrame, was {type(lr_network)}")
    if type(sig_network) is not pd.DataFrame:
        raise TypeError(f"sig_network should have type pandas.DataFrame, was {type(sig_network)}")
    if type(gr_network) is not pd.DataFrame:
        raise TypeError(f"gr_network should have type pandas.DataFrame, was {type(gr_network)}")
    if type(source_weights) is dict:
        for source, weight in source_weights.items():
            if weight < 0 or weight > 1:
                raise ValueError(f"{source} weight was {weight}, should be in the interval [0, 1]")
        source_weights = pd.DataFrame(dict(zip(("source", "weight"), zip(*source_weights.items()))))
    elif type(source_weights) is pd.DataFrame:
        for source, weight in zip(source_weights["source"], source_weights["weight"]):
            if weight < 0 or weight > 1:
                raise ValueError(f"{source} weight was {weight}, should be in the interval [0, 1]")
    else:
        raise TypeError(f"source_weights should have type pandas.DataFrame or dict[str, float], was {type(source_weights)}")
    if type(n_output_networks) is not int:
        raise TypeError(f"n_output_networks should have type int, was {type(n_output_networks)}")
    if n_output_networks != 2 and n_output_networks != 3:
        raise ValueError("n_output_networks should be 2 or 3")
    source_weights = source_weights[source_weights["weight"] > 0]
    gr_network_w = _sum_weights(gr_network, source_weights)
    if n_output_networks == 2:
        ligand_signaling_w = _sum_weights(pd.concat((lr_network, sig_network)), source_weights)
        return {
            "lr_sig": ligand_signaling_w,
            "gr": gr_network_w
        }
    else:
        lr_network_w = _sum_weights(lr_network, source_weights)
        sig_network_w = _sum_weights(sig_network, source_weights)
        return {
            "lr": lr_network_w,
            "sig": sig_network_w,
            "gr": gr_network_w
        }

def apply_hub_correction(
    df:pd.DataFrame,
    hub:float
) -> pd.DataFrame:
    if type(df) is not pd.DataFrame:
        raise TypeError(f"df should haver type pandas.DataFrame, was {type(df)}")
    if not isinstance(hub, Number):
        raise TypeError(f"hub should have type float, was {type(hub)}")
    if hub < 0 or hub > 1:
        raise ValueError("hub should be in the interval [0, 1]")
    if hub == 0:
        return df.copy()
    to_count = df.groupby("to").aggregate("count")
    to_count.rename(columns={"weight": "n"}, inplace=True)
    to_count.reset_index(inplace=True)
    df = df.merge(to_count, on="to", how="inner")
    df["weight"] = df["weight"] / (df["n"] ** hub)
    df.drop(columns="n", inplace=True)
    return df