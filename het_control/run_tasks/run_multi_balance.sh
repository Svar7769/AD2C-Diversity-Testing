#!/bin/bash
# Run with different SND values and agents_with_same_goal configurations

snd_values=(0.0)
goal_values=(4.0)
seeds=(0 1 2 3 4)

for snd in "${snd_values[@]}"; do
    for goals in "${goal_values[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "=================================================="
            echo "Running with SND: $snd, package_mass: $goals, seed: $seed"
            echo "=================================================="
            python ./AD2C-Diversity-Testing/het_control/run_tasks/run_balance.py \
                model.desired_snd=$snd \
                task.package_mass=$goals \
                task.n_agents=3 \
                seed=$seed \

            # Optional: check if the run was successful
            if [ $? -ne 0 ]; then
                echo "ERROR: Run failed for SND=$snd, goals=$goals, seed=$seed"
                # Uncomment the next line if you want to stop on first error
                # exit 1
            fi
            
            echo ""
        done
    done
done

echo "All experiments completed!"