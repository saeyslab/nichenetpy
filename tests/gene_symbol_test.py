'''
Here we test if the gene symbol conversion is the same as in NicheNetR. 
'''

import pytest
import os
from nichenetpy.gene_symbol import (
    GeneAliasInfo,
    GeneInfo
)
from common import (
    get_network_files,
    network_path
)


get_network_files()
mouse_alias_info = GeneAliasInfo(os.path.join(network_path, "geneinfo_alias_mouse.csv"))
human_alias_info = GeneAliasInfo(os.path.join(network_path, "geneinfo_alias_human.csv"))
gene_info = GeneInfo(os.path.join(network_path, "geneinfo.csv"))

def template_gene_alias_info(alias_info, alias, exp):
    res = alias_info[alias]
    assert res == exp, f"expected symbol {exp} for alias {alias}, got {res}"

def test_convert_alias_to_gene_mouse_0():
    template_gene_alias_info(mouse_alias_info, "0610005C13Rik", ("0610005C13Rik", 71661))

def test_convert_alias_to_gene_mouse_1():
    template_gene_alias_info(mouse_alias_info, "AI182092", ("0610005C13Rik", 71661))

def test_convert_alias_to_gene_mouse_2():
    template_gene_alias_info(mouse_alias_info, "Smap", ("1110004F10Rik", 56372))

def test_convert_alias_to_gene_mouse_3():
    template_gene_alias_info(mouse_alias_info, "c11orf1", ("1110032A03Rik", 68721))

def test_convert_alias_to_gene_mouse_4():
    template_gene_alias_info(mouse_alias_info, "13.MMHAP42FLE1", ("13.MMHAP42FLE1", 110442))

def test_convert_alias_to_gene_human_0():
    template_gene_alias_info(human_alias_info, "A1BG", ("A1BG", 1))

def test_convert_alias_to_gene_human_1():
    template_gene_alias_info(human_alias_info, "A1B1", ("A1BG", 1))

def test_convert_alias_to_gene_human_2():
    template_gene_alias_info(human_alias_info, "A2M-AS1", ("A2M-AS1", 144571))

def test_convert_alias_to_gene_human_3():
    template_gene_alias_info(human_alias_info, "DJS", ("ABCC2", 1244))

def test_convert_alias_to_gene_human_3():
    template_gene_alias_info(human_alias_info, "PALMCOX", ("ACOX1", 51))

def alias_to_symbol_template(input, exp, sort=False):
    res = mouse_alias_info.alias_to_symbol(input)
    if sort:
        res.sort()
    assert res == exp, f"expected mouse_alias_info({input}) == {exp}, got {res}"

def test_alias_to_symbol_no_doubles_list():
    alias_to_symbol_template(
        ["AI182092", "Bap18", "102g4T7", "10T"],
        ["0610005C13Rik", "0610010K14Rik", "102g4T7", "10T"]
    )

def test_alias_to_symbol_doubles_list():
    alias_to_symbol_template(
        ["1010001B22Rik", "109F12R2", "AW987535", "C79326"],
        ["1010001B22Rik", "109F12R", "AW987535", "1110008P14Rik"]
    )

def test_alias_to_symbol_no_list():
    alias_to_symbol_template(
        {"c11orf1", "AW556386", "Gm24413", "AU041756"},
        ["1110032A03Rik", "1500011B03Rik", "1600017P15Rik", "1700003E16Rik"],
        sort=True
    )

def test_alias_to_symbol_not_iterable():
    with pytest.raises(Exception):
        mouse_alias_info.alias_to_symbol(72)

def convert_human_to_mouse_template(input, exp, sort=False):
    res = list(gene_info.convert_human_to_mouse_symbols(input))
    if sort:
        res.sort()
    assert res == exp, f"expected gene_info.convert_human_to_mouse_symbols({input}) == {exp}, got {res}"

def convert_mouse_to_human_template(input, exp, sort=False):
    res = list(gene_info.convert_mouse_to_human_symbols(input))
    if sort:
        res.sort()
    assert res == exp, f"expected gene_info.convert_mouse_to_human_symbols({input}) == {exp}, got {res}"

def test_convert_human_to_mouse_0():
    convert_human_to_mouse_template(
        ["MSC", "A2M", "CTAGE4", "CYCTP", "TRD", "ENO3", "BRCC3"],
        ["Msc", "A2m", None, None, None, "Eno3", "Brcc3"]
    )

def test_convert_human_to_mouse_1():
    convert_human_to_mouse_template(
        ["FABP9", "SPATA22", "CES3", "CYP4F8"],
        ["Fabp9", "Spata22", "Ces3b", "Cyp4f37"]
    )

def test_convert_human_to_mouse_2():
    convert_human_to_mouse_template(
        ["PRAMEF14", "PRAMEF15", "PRAMEF17", "PRAMEF20"],
        [None, None, None, None]
    )

def test_convert_mouse_to_human_0():
    convert_mouse_to_human_template(
        ["Boll", "Ts(1<13>)70H", "Igkv11-125", "A2m", "Rps27rt", "Gm45261", "A1cf"],
        ["BOLL", None, None, "A2M", "RPS27", None, "A1CF"]
    )

def test_convert_mouse_to_human_1():
    convert_mouse_to_human_template(
        ["Adh1", "Clec2m", "Alg10b", "Fbxw10"],
        [None, None, "ALG10B", "FBXW10"]
    )

def test_convert_mouse_to_human_2():
    convert_mouse_to_human_template(
        ["Ssxb3", "Ssx9", "Ssxb10", "Ssxb9"],
        [None, None, None, None]
    )