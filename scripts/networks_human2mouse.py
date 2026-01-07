from nichenetpy.utils import (
    read_csv_cols
)
from nichenetpy.gene_symbol import gene_info

import pandas as pd
import os
import argparse

def human2mouse(df):
    df["from"] = list(gene_info.convert_human_to_mouse_symbols(df["from"]))
    df["to"] = list(gene_info.convert_human_to_mouse_symbols(df["to"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert human networks to mouse networks"
    )
    parser.add_argument(
        "out_path",
        help="path to the directory where the output should be saved",
    )
    parser.add_argument(
        "--lr_network",
        help="path to the ligand receptor network file",
        default=None
    )
    parser.add_argument(
        "--gr_network",
        help="path to the gene regulatory network file",
        default=None
    )
    parser.add_argument(
        "--sig_network",
        help="path to the signaling network file",
        default=None
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if args.lr_network is not None:
        human2mouse(pd.DataFrame(read_csv_cols(args.lr_network))).to_csv(os.path.join(args.out_path, args.lr_network.replace("human", "mouse")))
    if args.gr_network is not None:
        human2mouse(pd.DataFrame(read_csv_cols(args.gr_network))).to_csv(os.path.join(args.out_path, args.gr_network.replace("human", "mouse")))
    if args.sig_network is not None:
        human2mouse(pd.DataFrame(read_csv_cols(args.sig_network))).to_csv(os.path.join(args.out_path, args.sig_network.replace("human", "mouse")))