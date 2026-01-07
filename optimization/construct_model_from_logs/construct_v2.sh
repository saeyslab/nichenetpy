model_version="v2"
model_type="all_sources"
aggregation_method="average"
python C:/Users/victorm/Documents/nichenetpy/optimization/construct_final_model_from_logs.py \
	"D:/Data/nichenetpy/model_optimization/log/${model_version}/${model_type}/GP/folds" \
	--lr_network "D:/Data/nichenetpy/model/human/csv/lr_network_human.csv" \
	--lr_network "D:/Data/nichenetpy/model/mouse/csv/lr_network_mouse.csv" \
	--gr_network "D:/Data/nichenetpy/model/human/csv/gr_human.csv" \
	--gr_network "D:/Data/nichenetpy/model/mouse/csv/gr_mouse.csv" \
	--sig_network "D:/Data/nichenetpy/model/human/csv/lr_sig_human.csv" \
	--sig_network "D:/Data/nichenetpy/model/mouse/csv/lr_sig_mouse.csv" \
	--model_path "D:/Data/nichenetpy/model_optimization/model/${model_version}/${model_type}/human/nichenet_human_${model_version}_${aggregation_method}.pkl" \
	--model_path "D:/Data/nichenetpy/model_optimization/model/${model_version}/${model_type}/mouse/nichenet_mouse_${model_version}_${aggregation_method}.pkl"