from nichenetpy.utils import (
    read_csv_cols
)
from nichenetpy.gene_symbol import gene_info

import pandas as pd
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert human networks to mouse networks"
    )
    parser.add_argument(
        "out_path",
        help="path to the directory where the output should be saved",
    )
    parser.add_argument(
        "--network",
        help="path to the network file (csv)",
        action="append",
        default=[]
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    for network in args.network:
        df = pd.DataFrame(read_csv_cols(network))
        df["from"] = list(gene_info.convert_human_to_mouse_symbols(df["from"]))
        df["to"] = list(gene_info.convert_human_to_mouse_symbols(df["to"]))
        df.to_csv(os.path.join(args.out_path, network.replace("human", "mouse")))