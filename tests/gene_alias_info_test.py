from nichenetpy.gene_symbol import mouse_alias_info, human_alias_info

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