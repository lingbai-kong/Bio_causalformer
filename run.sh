echo "BioCausalFormer diamond"
for i in $(seq 0 9)
do
    echo "running diamond_$i"
    python runner.py -d "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/diamond/data_$i.csv" -g "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/diamond/groundtruth.csv" > "./logs/diamond_$i.log" 2>&1
done
echo "BioCausalFormer fork"
for i in $(seq 0 9)
do
    echo "running fork_$i"
    python runner.py -d "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/fork/data_$i.csv" -g "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/fork/groundtruth.csv" > "./logs/fork_$i.log" 2>&1
done
echo "BioCausalFormer mediator"
for i in $(seq 0 9)
do
    echo "running mediator_$i"
    python runner.py -d "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/mediator/data_$i.csv" -g "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/mediator/groundtruth.csv" > "./logs/mediator_$i.log" 2>&1
done
echo "BioCausalFormer v"
for i in $(seq 0 9)
do
    echo "running v_$i"
    python runner.py -d "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/v/data_$i.csv" -g "/remote-home/share/dmb_nas/konglingbai/causality/data/basic/v/groundtruth.csv" > "./logs/v_$i.log" 2>&1
done
echo "BioCausalFormer lorenz96"
for i in $(seq 0 9)
do
    echo "running lorenz96_$i"
    python runner.py -d "/remote-home/share/dmb_nas/konglingbai/causality/data/lorenz96/timeseries$i.csv" -g "/remote-home/share/dmb_nas/konglingbai/causality/data/lorenz96/groundtruth.csv" > "./logs/lorenz96_$i.log" 2>&1
done
echo "BioCausalFormer fMRI"
for i in $(seq 1 28)
do
    echo "running fMRI_${i}"
    python runner.py -d "/remote-home/share/dmb_nas/konglingbai/causality/data/fMRI/timeseries$i.csv" -g "/remote-home/share/dmb_nas/konglingbai/causality/data/fMRI/sim${i}_gt_processed.csv" > "./logs/fMRI_$i.log" 2>&1
done