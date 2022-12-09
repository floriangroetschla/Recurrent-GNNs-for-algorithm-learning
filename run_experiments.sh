conda activate RecGNN

echo Creating experiment configs
python create_configs.py

mkdir models

CONFIG_ID=0
mkdir runs_recGNN
for FILE in configs_recGNN/* 
do 
	echo Run RecGNN config $CONFIG_ID
	python run_experiment.py --config $FILE > runs_recGNN/output_$CONFIG_ID.out 
	(( CONFIG_ID++ ))
done

python collect_results.py runs_recGNN output_recGNN

CONFIG_ID=0
mkdir runs_itergnn
for FILE in configs_itergnn/* 
do 
	echo Run IterGNN config $CONFIG_ID
	python run_experiment.py --config $FILE > runs_itergnn/output_$CONFIG_ID.out 
	(( CONFIG_ID++ ))
done

python collect_results.py runs_itergnn output_itergnn

CONFIG_ID=0
mkdir runs_gin
for FILE in configs_gin/* 
do 
	echo Run GIN config $CONFIG_ID
	python run_experiment.py --config $FILE > runs_gin/output_$CONFIG_ID.out 
	(( CONFIG_ID++ ))
done

echo Collect results
python collect_results.py runs_gin output_gin

echo Run recGNN extrapolation
python run_extrapolation.py output_recGNN.csv gin-mlp
python run_extrapolation.py output_recGNN.csv gru-mlp

echo Run IterGNN extrapolation
python run_extrapolation_itergnn.py

echo Run GIN extrapolation
python run_extrapolation_gin.py

echo Run recGNN stabilization
python run_stabilization.py gin-mlp
python run_stabilization.py gru-mlp

