experiment=mnist_38
mkdir out/${experiment}-standard/abcrown_results
rm -rf out/${experiment}-standard/abcrown_results/*
mkdir out/${experiment}-standard/abcrown_times
rm -rf out/${experiment}-standard/abcrown_times/*

start=`date +%s.%N`

norm=1
for epsilon in 2 4 6 8 10; do
    python lib/alpha-beta-CROWN/complete_verifier/abcrown.py --config lib/alpha-beta-CROWN/complete_verifier/exp_configs/custom_${experiment}.yaml --epsilon=$epsilon --norm=${norm} --save_file=${experiment}_${epsilon}_${norm}.npy --batch_size=32 --batch_size_primal=100
    mv ${experiment}_${epsilon}_${norm}.npy out/${experiment}-standard/abcrown_results/${epsilon}_${norm}.npy 
done

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo $runtime >> out/${experiment}-standard/abcrown_times/time_l1.txt
start=`date +%s.%N`

norm=2
for epsilon in 0.5 1 1.5 2 2.5; do
    python lib/alpha-beta-CROWN/complete_verifier/abcrown.py --config lib/alpha-beta-CROWN/complete_verifier/exp_configs/custom_${experiment}.yaml --epsilon=$epsilon --norm=${norm} --save_file=${experiment}_${epsilon}_${norm}.npy --batch_size=32 --batch_size_primal=100
    mv ${experiment}_${epsilon}_${norm}.npy out/${experiment}-standard/abcrown_results/${epsilon}_${norm}.npy 
done

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo $runtime >> out/${experiment}-standard/abcrown_times/time_l2.txt
start=`date +%s.%N`

norm=inf
for epsilon in 0.025 0.05 0.075 0.1 0.125; do
    python lib/alpha-beta-CROWN/complete_verifier/abcrown.py --config lib/alpha-beta-CROWN/complete_verifier/exp_configs/custom_${experiment}.yaml --epsilon=$epsilon --norm=${norm} --save_file=${experiment}_${epsilon}_${norm}.npy --batch_size=32 --batch_size_primal=100
    mv ${experiment}_${epsilon}_${norm}.npy out/${experiment}-standard/abcrown_results/${epsilon}_${norm}.npy 
done

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo $runtime >> out/${experiment}-standard/abcrown_times/time_linf.txt

python scripts/abcrown/abcrown_combine.py --data=mnist_38
