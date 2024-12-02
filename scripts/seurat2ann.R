remotes::install_github("scverse/anndataR")
library("Seurat")
library("SeuratObject")
library("hdf5r")
setwd('D:/Data/nichenetpy')
old <- readRDS("./rds/seuratObj3531889.rds")

seuratObj <- CreateSeuratObject(
  counts = GetAssayData(old, layer="counts"),
  data = GetAssayData(old, layer="data"),
  meta.data = old@meta.data
)
seuratObj <- SetAssayData(
  seuratObj,
  layer="scale.data",
  new.data=GetAssayData(old, layer="scale.data")
)
ann <- anndataR::from_Seurat(
  seuratObj,
  "InMemoryAnnData"
)
#anndataR::from_Seurat(
#  seuratObj,
#  "HDF5AnnData",
#  file="./annData/annData3531889.h5"
#)