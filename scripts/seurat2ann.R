remotes::install_github("scverse/anndataR")
library("Seurat")
library("SeuratObject")
library("hdf5r")
setwd('D:/Data/nichenetpy')
old <- readRDS("./seurat/seurat_obj_subset_integrated_zonation.rds")
seuratObj <- old
seuratObj$RNA <- ScaleData(old$RNA)
seuratObj$SCT <- ScaleData(old$SCT)
#seuratObj <- PrepSCTFindMarkers(seuratObj, assay = "SCT")
#seuratObj <- CreateSeuratObject(
#  counts = GetAssayData(old, layer="counts"),
#  data = GetAssayData(old, layer="data"),
#  meta.data = old@meta.data
#)
#seuratObj <- SetAssayData(
#  seuratObj,
#  layer="scale.data",
#  new.data=GetAssayData(old, layer="scale.data")
#)
#seuratObj@assays[["RNA"]]@layers$counts@Dimnames <- old@assays[["RNA"]]$counts@Dimnames
#seuratObj@assays[["RNA"]]@layers$data@Dimnames <- old@assays[["RNA"]]$data@Dimnames
#seuratObj[["RNA"]]@meta.data$gene = old@assays[["RNA"]]$counts@Dimnames[[1]]
#seuratObj[["SCT"]]@meta.data$gene = old@assays[["SCT"]]$counts@Dimnames[[1]]
ann <- anndataR::from_Seurat(
  seuratObj,
  "InMemoryAnnData"
)
anndataR::from_Seurat(
  seuratObj,
  "HDF5AnnData",
  file="./annData/temp.h5",
  mode="w",
  assay_name="SCT"
)