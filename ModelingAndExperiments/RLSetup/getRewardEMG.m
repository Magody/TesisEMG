function [reward, new_state, finish] = getRewardEMG(state, action_selected, all_rewards, context) 

    real_action = context('real_action');
    
    finish = -1;  % unknown here, outside yes
    
    if real_action == action_selected
        reward = all_rewards.correct;
    else
        reward = all_rewards.incorrect;
    end
    new_state = state;
end