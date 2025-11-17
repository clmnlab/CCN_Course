import numpy as np
import random
import matplotlib.pyplot as plt

def plot_adversarial_strategy(session_data, target_action_id=0, title='Adversarial Strategy Visualization'):
    """
    Visualizes the interaction between an adversary and a learner, 
    inspired by Figure 3 of Dezfouli et al. (2020) PNAS.

    This function assumes:
    - Target action is '1' (blue), Non-Target is '0' (red).
    - session_data[t][0] (adv_action) meaning:
        - 0: No reward assigned
        - 1: Reward assigned to Target
        - 2: Reward assigned to Non-Target
        - 3: Reward assigned to BOTH Target and Non-Target
    - session_data[t][1] (learner_action): The learner's actual choice.

    Args:
        session_data (list): A list of tuples, where each tuple is
                             (adv_action, learner_action, reward_to_learner, adv_reward).
        target_action_id (int): The ID (0 or 1) that represents the "Target" action.
        p_target_probs (list or np.array, optional): 
                                A list of probabilities (0-1) representing 
                                the learner model's P(Target) at each trial.
        title (str): The title of the plot.
    """
    
    # --- 1. Unpack data ---
    num_trials = len(session_data)
    trials = np.arange(num_trials)
    
    adv_actions = [s[0] for s in session_data]
    learner_actions = [s[1] for s in session_data]
    p_target_probs = [s[2] for s in session_data]
    non_target_action_id = 1 - target_action_id

    # --- 2. Create Plot ---
    fig, ax = plt.subplots(figsize=(16, 6))

    # --- 3. Plot P(Target) (Green Shaded Area) ---
    if len(p_target_probs) != num_trials:
        print(f"Warning: p_target_probs length ({len(p_target_probs)}) does not match session_data length ({num_trials}). Skipping plot.")
    else:
        # Fill the area
        ax.fill_between(trials, 0.5, p_target_probs, 
                        color='green', alpha=0.3, label='Learner Model P(Target)')
        # Plot the line
        ax.plot(trials, p_target_probs, color='green', alpha=0.7)
        # Plot chance level
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance (0.5)')

    # --- 4. Plot Adversary's Actions (Vertical Lines) ---
    # *MODIFIED LOGIC*
    # Based on the new definition: 0=None, 1=Target, 2=Non-Target, 3=Both
    
    # Plot Target reward line if adv_action is 1 (Target) OR 3 (Both)
    target_reward_trials = [t for t in trials if adv_actions[t] in [1, 3]]
    
    # Plot Non-Target reward line if adv_action is 2 (Non-Target) OR 3 (Both)
    nontarget_reward_trials = [t for t in trials if adv_actions[t] in [2, 3]]

    # Plot all at once
    if target_reward_trials:
        # *** UNCHANGED: Adjusted y-range to "touch" the dots at y=0.8 ***
        ax.vlines(target_reward_trials, 0.5, 1.0, 
                  color='blue', linewidth=2.5, label='Reward Assigned to Target')
    if nontarget_reward_trials:
        # *** CHANGED (as requested): Move bar to attach to dots (y=0.8) and point down ***
        ax.vlines(nontarget_reward_trials, 0, 0.5, 
                  color='red', linewidth=2.5, label='Reward Assigned to Non-Target')
        
    # Note: On trials where adv_action == 3, both lines will be drawn.

    # --- 5. Plot Learner's Actions (Dots) ---
    target_choice_trials = [t for t in trials if learner_actions[t] == target_action_id]
    target_y = [0.5] * len(target_choice_trials) # Plot at y=0
    
    nontarget_choice_trials = [t for t in trials if learner_actions[t] == non_target_action_id]
    nontarget_y = [0.5] * len(nontarget_choice_trials) # <-- *** CHANGED: Plot ALSO at y=0 ***

    if target_choice_trials:
        ax.plot(target_choice_trials, target_y, 'o', color='blue', 
                markersize=6, alpha=0.8, label='Learner Chose Target')
    if nontarget_choice_trials:
        ax.plot(nontarget_choice_trials, nontarget_y, 'o', color='red', 
                markersize=6, alpha=0.8, label='Learner Chose Non-Target')

    # --- 6. Set Plot Aesthetics ---
    # ax.set_yticks([0.2, 0.8]) # <-- REMOVED
    # ax.set_yticklabels(['Non-Target Choice', 'Target Choice']) # <-- REMOVED
    ax.set_yticks([]) # Remove all y-ticks
    ax.set_ylim(-0.2, 1.2) # Make space for lines and area plot
    
    ax.set_xlabel('Trial Number', fontsize=16)
    ax.set_ylabel('Target Probability', fontsize=16) # <-- UPDATED LABEL
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    # --- 7. Create a clean legend ---
    # Get unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()