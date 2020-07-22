function [assignment, totalcost] = munkres2(n_states, estseq_states, true_seq, est_seq)
    %First create the cost matrix to compare
    cost_matrix = zeros(n_states);
    if(estseq_states>n_states) %the algorithm has fewer states, fill it with zeros because the cost matrix should be square
        cost_matrix = zeros(estseq_states);
    else
        cost_matrix = zeros(n_states);
    end
    
    if length(true_seq)~=length(est_seq)
        disp('ERROR: Sequences dont have the same length.');
    else
        for n=1:size(true_seq,2) %number of sequences
            for i=1:size(true_seq,1) %length of sequences (ASSUMED SAME LENGTH!)
                %increase by one the position in the matrix
                %where the row is the estimated state and the column is the
                %true state
                cost_matrix(est_seq(i,n), true_seq(i,n)) = cost_matrix(est_seq(i,n), true_seq(i,n))+1;
            end
        end
    end
    
    
    
    %Find the permutations of the cost matrix which give lowest total cost
    %indexes are tuples with the association between the indexes of the true model 
    %and the corresponding indexes in the predicted model
    %due to the identifiability problem, the states are not always estimated in the right order
    
    %We input -cost_matrix because the lowest cost corresponds to the
    %highest counts
    [assignment, totalcost] = munkres(-cost_matrix);
    
    %Print state associations
    disp('State associations: (algorithm state, original state)');
    for i=1:length(assignment)
        fprintf('(%d,%d)\n', i, assignment(i));
    end
    
end
    
    
    