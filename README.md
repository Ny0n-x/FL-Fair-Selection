# FL-Fair-Selection
FL client selection via fair division

# for baseline models run (default 100 rounds, random)
python3 main.py --rounds 5
python3 --selector power_of_choice
python3 main.py --selector divfl --alpha 0.5 --seed 1