#remotes::install_github("scverse/anndataR")
library("Seurat")
library("SeuratObject")
library("hdf5r")
setwd('D:/Data/nichenetpy')
old <- readRDS("./seurat/seuratObj3531889.rds")
seuratObj <- old
#seuratObj$RNA <- ScaleData(old$RNA)
#seuratObj$SCT <- ScaleData(old$SCT)
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
seuratObj <- UpdateSeuratObject(seuratObj)
DefaultAssay(seuratObj) <- "RNA"
Idents(seuratObj) <- seuratObj$celltype
'seuratObj@misc = list(
  cell_attr=SCTResults(object = seuratObj[["SCT"]], slot = "cell.attributes"),
  model_pars_fit=lapply(
    X = SCTResults(object = seuratObj[["SCT"]], slot = "feature.attributes"),
    FUN = function(x) x[, c("theta", "(Intercept)", "log_umi")]
  ),
  arguments=SCTResults(object = seuratObj[["SCT"]], slot = "arguments")
)'
#seuratObj <- PrepSCTFindMarkers(seuratObj, assay = "SCT", verbose = TRUE)
ann <- anndataR::as_AnnData(
  seuratObj,
  output_class="InMemory",
  assay_name="RNA"
)
anndataR::write_h5ad(
  ann,
  path="./annData/temp.h5",
  mode="w"
)
'anndataR::from_Seurat(
  seuratObj,
  "HDF5AnnData",
  file="./annData/temp.h5",
  mode="w",
  assay_name="SCT"
)'