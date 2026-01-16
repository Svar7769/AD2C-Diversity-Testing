#!/bin/bash
# Run with different SND values and agents_with_same_goal configurations

snd_values=(0.0)
goal_values=(2 1)

for snd in "${snd_values[@]}"; do
    for goals in "${goal_values[@]}"; do
        echo "=================================================="
        echo "Running with SND: $snd, agents_with_same_goal: $goals"
        echo "=================================================="
        python ./AD2C-Diversity-Testing/het_control/run_tasks/run_navigation.py \
            model.desired_snd=$snd \
            task.agents_with_same_goal=$goals
        
        # Optional: check if the run was successful
        if [ $? -ne 0 ]; then
            echo "ERROR: Run failed for SND=$snd, goals=$goals"
            # Uncomment the next line if you want to stop on first error
            # exit 1
        fi
        
        echo ""
    done
done

echo "All experiments completed!"