%%
%clear all
%keySet = {'n_states','length_dur','max_obs','original_lambdas', 'original_PAI', 'original_A', 'original_P', 'original_B', 'N', 'dim', 'Vk', 'obs_seq', 'state_seq', 'lambdas_est', 'PAI_est_final', 'A_est', 'B_est_final', 'P_est_final', 'Qest_final', 'lkh', 'll', 'elapsed_time',  'hit_rate', 'total_ll'};

kmax = 10;
kmin = 2;
k=kmax;

C = zeros(kmax-kmin+1, 1);
C2 = zeros(kmax-kmin+1, 1);
C3 = zeros(kmax-kmin+1, 1);
C4 = zeros(kmax-kmin+1, 1);
C_bic = zeros(kmax-kmin+1,1);

%Define the number of states of the markov chain
n_states = 5;

%Define the maximum length of state duration
length_dur = 4;

%Max observable value[fList,pList] = matlab.codetools.requiredFilesAndProducts('myFun.m');
max_obs = 20;

%Create the markov modulated poisson process original model
%original_model = MMPP(n_states, length_dur, max_obs);
original_lambdas = [2,15,36,49,105];

rng(1);
%Initialize the state initial distribution vector of the markov chain with random values
original_PAI = generate_random_matrix([1,n_states], 'normal')';

rng(1);
%Initialize the transition matrix of the markov chain with random values
original_A = generate_random_matrix([n_states,n_states], 'zero-diag');

%% REPLACED original_P = generate_random_matrix([n_states, length_dur], 'normal'); BY
original_PM = [0.05, 0.12, 0.26, 0.59, 0.83]; %poisson durations

rng(1);
original_B = generate_random_matrix([n_states, max_obs], 'normal');
%original_B = [0.05, 0.9, 0.05; 0.9, 0.05, 0.05; 0, 0.1, 0.9];

%Number of observations
N = 2;

%Dimension of the observations
dim = 1000;

iteration = 0;
results_mmdl = cell(50,12);
results_bic = cell(50,9);
originals = cell(1,8);
sampled_iter = cell(50,2);

originals{1} = original_lambdas;
originals{2} = original_PAI;
originals{3} = original_A;
originals{4} = original_B;
originals{5} = original_PM;
originals{6} = n_states;
originals{7} = N;
originals{8} = dim;


for iter=1:50
    %[obs_seq, state_seq, state_trans] = original_model.sample2(N, dim);
    %% REPLACED [Vk, obs_seq, state_seq] = hsmmSample(original_PAI,original_A,original_P,original_B, original_lambdas, dim,N); BY
    [Vk, obs_seq, state_seq] = hsmmSample(original_PAI,original_A,original_PM,original_B, original_lambdas, dim,N);

   
    K = sum(unique(Vk));
    if N~=1
        x=cell2mat(obs_seq)';
        s=cell2mat(state_seq)';
    else
        x=obs_seq';
        s=state_seq';
    end
    
    sampled_iter{iter,1} = x;
    sampled_iter{iter,2} = s;
    %assert sample is OK! estimated_pi must be close to original_pi,
    %estimated_T to original_A and estimated_D to original_P!

    %[estimated_pi, estimated_T, estimated_D] = analize_sequence(cell2mat(state_seq)', n_states, length_dur)
    %%
    %obs_seq = obs_seq';
    %state_seq = state_seq' +1;

    %% BIC    
    k=kmax;
    while(k >= kmin)
        [A,B,PM,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x,k,length_dur,K,dim*ones(N,1));
        
        tic
        %% REPLACED [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,P,x,100,dim*ones(N,1), 1e-10, Vk); BY
        [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,PM_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,PM, x,100,dim*ones(N,1), 1e-100, Vk);

        elapsed_time = toc
        
        %compute total loglikelihood
        total_ll = sum(ll,2);
        
        models_bic{k-kmin+1,:}={A_est,PAI_est,lambdas_est,B_est,PM_est, Qest, total_ll(end),ir};
        
        C_bic(k-kmin+1) = total_ll(ir) - ((k^2+k)/2)*log(N*dim);
        k=k-1;
    end
    %compute optimal state
    C_bic = abs(C_bic/(N*dim));
    [optimal_state,~] = find(min(C_bic)==C_bic);
    %optimal model is the one from the iteration corresponding to the optimal
    %state
    optimal_model = models_bic{optimal_state,:};
    optimal_state = optimal_state-1+kmin;

    optimal_A = optimal_model{1};
    optimal_PAI = optimal_model{2};
    optimal_lambdas = optimal_model{3};
    optimal_B = optimal_model{4};
    optimal_PM = optimal_model{5};
    optimal_Qest = optimal_model{6};
    
    %for i=1:1%N
    %    figure
    %    plot(cell2mat(obs_seq(i,:)));
    %    hold on
    %    plot(optimal_lambdas(optimal_Qest(:,i)));
    %end
    
    if optimal_state >= n_states
        %MUNKRES Find the optimal assginment of states
        [assignment, total_cost] = munkres2(n_states, optimal_state, s, optimal_Qest);

        %Compute final estimated state sequence according to the assignment
        Qest_final = zeros(size(optimal_Qest));
        for i=1:length(assignment)
            Qest_final(optimal_Qest==i) = assignment(i);
        end

        %Compute final parameters according to the assignment
        %With the indexes association, permute the rows and columns of the predicted 
        %transition matrix so that it matches the true transition matrix
        %Same for the duration probability density function (but only rows, since rows are states and columns are durations)
        B_est_final = zeros(size(optimal_B));
        %% REPLACED P_est_final = zeros([n_states, length_dur]); BY
        PM_est_final = zeros(1,n_states);
        PAI_est_final = zeros(n_states,1);
        lambdas_est_final = zeros(1, n_states);
        prev_assign = [];
        for i=1:length(assignment)
            %% REPLACED P_est_final(assignment(i),:) = P_est(i,:); BY
            PM_est_final(assignment(i))=optimal_PM(i);
            B_est_final(assignment(i),:) = optimal_B(i,:);
            PAI_est_final(assignment(i)) = optimal_PAI(i);
            lambdas_est_final(assignment(i)) = optimal_lambdas(i);
            [v,idx] = find(i==prev_assign(:));

            if (~isempty(v) && assignment(i) == v) || i==assignment(i) %the swap was already performed
                prev_assign = [prev_assign, assignment(i)];
                continue;
            else
                optimal_A([i, assignment(i)], :) = optimal_A([assignment(i),i], :);     % swap rows.
                optimal_A(:, [i, assignment(i)]) = optimal_A(:, [assignment(i),i]);     % swap columns.
                prev_assign = [prev_assign, assignment(i)];
            end 
        end 
    else
        Qest_final = optimal_Qest;
        lambdas_est_final = optimal_lambdas;
        A_est = optimal_A;
        B_est_final = optimal_B;
        PM_est_final = optimal_PM;
        
    end

    hit_rate = (sum(sum(s == Qest_final))/(N*dim));
    fprintf('Percentage of right state estimates %.2f %%\n', hit_rate*100);

    %for i=1:1%N
    %    figure
    %    plot(x(:,i));
    %    hold on
    %    plot(lambdas_est_final(Qest_final(:,i)));
    %end
    %mse?
    total_ll = optimal_model{7};
    ir = optimal_model{8};
    
    results_bic{iter,1} = lambdas_est_final';
    results_bic{iter,2} = PAI_est_final;
    results_bic{iter,3} = A_est;
    results_bic{iter,4} = B_est_final;
    %% REPLACED results{iter,5} = P_est_final; BY
    results_bic{iter,5} = PM_est_final;
    results_bic{iter,6} = Qest_final;
    results_bic{iter,7} = hit_rate;
    results_bic{iter,8} = total_ll;
    results_bic{iter,9} = ir;
    
    %% MMDL PRUNNING
    k=kmax;
    
    %% REPLACED [A,B,P,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x,k,length_dur,K,dim*ones(N,1)); BY
    [A,B,PM,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x,k,length_dur,K,dim*ones(N,1));
    %Vk = unique(cell2mat(obs_seq)');
    %if sum(Vk==0)>0
    %    Vk=Vk(2:end);
    %end
    %models = cell(kmax-kmin+1,1);
    %models{k-kmin+1,:} = {A,B,P,PAI};

   
    while(k >= kmin)

        tic
        %% REPLACED [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,P,x,100,dim*ones(N,1), 1e-10, Vk); BY
        [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,PM_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,PM, x,100,dim*ones(N,1), 1e-100, Vk);

        elapsed_time = toc

        %compute total loglikelihood
        total_ll = sum(ll,2);

       
        %store the model
        %% REPLACED models{k-kmin+1,:}={A_est,PAI_est,lambdas_est,B_est,P_est, Qest, total_ll(end)}; BY
        models{k-kmin+1,:}={A_est,PAI_est,lambdas_est,B_est,PM_est, Qest, total_ll(end),ir};


        %FIRST WAY: HMMS
        %compute the stationary probability distribution vector p_infty
        [V,D,W]=eig(A_est);
        [r,c] = find(abs(D-1)<1e-10);
        p_infty = abs(W(:,r));

        %normalize
        p_infty = p_infty/sum(p_infty);

        %estimator of p_infty
        jumps = cell(N,1);
        counting_process_NI = zeros(N,k);
        counting_process_N = zeros(N,1);
        for i=1:N
            jumps{i} = [Qest(1,i); Qest([0;diff(Qest(:,i))~=0]>0,i)];
            counting_process_N(i) = length(jumps{i})-1;
            for j=1:k
                %number of visits of the semi markov chain to state j up to time T
                %in the ith observation sequence
                counting_process_NI(i,j) = sum(jumps{i}==j);
            end
        end

        %sum the counting process for all obs sequences ???
        counting_process_NI = sum(counting_process_NI, 1);
        length_dur = floor(log(0.001)/log(1-min(PM_est)));
        d=[1:1:length_dur];
        clear P_est;
        for aux=1:k
            P_est(aux,:) = PM_est(aux)*(1-PM_est(aux)).^(d-1); %poisspdf(d,PM(aux)+1e-100); %geopdf(d,PM(aux)); 
            P_est(aux,:)=P_est(aux,:)/sum(P_est(aux,:));
        end
        p_infty_ = (counting_process_NI/sum(counting_process_N))';
        m=zeros(1,k);
        for i=1:k
            m(i)=P_est(i,:)*d';
        end
        p_infty_hsmm = p_infty.*m';
        p_infty_hsmm = p_infty_hsmm/sum(p_infty_hsmm);
        %or
        p_infty_hsmm2 = p_infty_.*m';
        p_infty_hsmm2 = p_infty_hsmm2/sum(p_infty_hsmm2);

        %THIRD WAY: I invented
        p_infty2 = A_est'*P_est;
        for i=1:size(p_infty2,2)
            p_infty2(:,i) = p_infty2(:,i)*i; %average state occupation
        end
        %sum state occupation for all state durations
        p_infty2 = sum(p_infty2,2);

        %normalize
        p_infty2=p_infty2/sum(p_infty2);


        %compute mmdl
        %p_infty for HMMs (eigenvector of A)
        %C(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-(length_dur/2)*sum(log(N*dim*p_infty));
        C(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-sum(log(N*dim*p_infty));
        
        %My way, A^T*p*[123..k]D*[1]
        %C2(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-(length_dur/2)*sum(log(N*dim*p_infty2));
        C2(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-sum(log(N*dim*p_infty2));
        
        %HSMM using p_infty (HMMs)
        %C3(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-(length_dur/2)*sum(log(N*dim*p_infty_hsmm));
        C3(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-sum(log(N*dim*p_infty_hsmm));
        
        %HSMM using the estimator for p_infty (HMMs)
        %C4(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-(length_dur/2)*sum(log(N*dim*p_infty_hsmm2));
        C4(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-sum(log(N*dim*p_infty_hsmm2));


        %compute least probable state
        [M,I] = min(p_infty_hsmm2);
        least_probable_state = I;

        %compute least probable state
        [M,I] = min(p_infty2);
        least_probable_state2 = I;

        %compute least probable state
        [M,I] = min(p_infty_hsmm);
        least_probable_state_hsmm = I;

        %compute least probable state
        [M,I] = min(p_infty_hsmm2);
        least_probable_state_hsmm2 = I;

        %prune
        A_est(:,least_probable_state) = [];
        A_est(least_probable_state,:) = [];
        PAI_est(least_probable_state)=[];
        lambdas_est(least_probable_state)=[];
        B_est(least_probable_state,:)=[];
        %% REPLACED P_est(least_probable_state,:)=[]; BY
        PM_est(least_probable_state)=[];

        k=k-1;
        %normalize again
        PAI_est = PAI_est/sum(PAI_est);
        for j=1:k
            A_est(j,:)=A_est(j,:)/sum(A_est(j,:));
        end


        %initial reduced model
        A=A_est;
        PAI=PAI_est;
        lambdas = lambdas_est;
        B = B_est;
        %% REPLACED P = P_est; BY
        PM = PM_est;
    end
    
    %compute optimal state
    C4= abs(C4/(N*dim));
    [optimal_state,~] = find(min(C4)==C4);
    %optimal model is the one from the iteration corresponding to the optimal
    %state
    optimal_model = models{optimal_state,:};
    optimal_state = optimal_state-1+kmin;

    optimal_A = optimal_model{1};
    optimal_PAI = optimal_model{2};
    optimal_lambdas = optimal_model{3};
    optimal_B = optimal_model{4};
    optimal_PM = optimal_model{5};
    optimal_Qest = optimal_model{6};

    %for i=1:1%N
    %    figure
    %    plot(cell2mat(obs_seq(i,:)));
    %    hold on
    %    plot(optimal_lambdas(optimal_Qest(:,i)));
    %end
    
    if optimal_state>=n_states
        %MUNKRES Find the optimal assginment of states
        [assignment, total_cost] = munkres2(n_states, optimal_state, s, optimal_Qest);

        %Compute final estimated state sequence according to the assignment
        Qest_final = zeros(size(optimal_Qest));
        for i=1:length(assignment)
            Qest_final(optimal_Qest==i) = assignment(i);
        end

        %Compute final parameters according to the assignment
        %With the indexes association, permute the rows and columns of the predicted 
        %transition matrix so that it matches the true transition matrix
        %Same for the duration probability density function (but only rows, since rows are states and columns are durations)
        B_est_final = zeros(size(optimal_B));
        %% REPLACED P_est_final = zeros([n_states, length_dur]); BY
        PM_est_final = zeros(1,n_states);
        PAI_est_final = zeros(n_states,1);
        lambdas_est_final = zeros(1, n_states);
        prev_assign = [];
        for i=1:length(assignment)
            %% REPLACED P_est_final(assignment(i),:) = P_est(i,:); BY
            PM_est_final(assignment(i))=optimal_PM(i);
            B_est_final(assignment(i),:) = optimal_B(i,:);
            PAI_est_final(assignment(i)) = optimal_PAI(i);
            lambdas_est_final(assignment(i)) = optimal_lambdas(i);
            [v,idx] = find(i==prev_assign(:));

            if (~isempty(v) && assignment(i) == v) || i==assignment(i) %the swap was already performed
                prev_assign = [prev_assign, assignment(i)];
                continue;
            else
                optimal_A([i, assignment(i)], :) = optimal_A([assignment(i),i], :);     % swap rows.
                optimal_A(:, [i, assignment(i)]) = optimal_A(:, [assignment(i),i]);     % swap columns.
                prev_assign = [prev_assign, assignment(i)];
            end 
        end 
    else
        Qest_final = optimal_Qest;		
        lambdas_est_final = optimal_lambdas;		
        A_est = optimal_A;		
        B_est_final = optimal_B;		
        PM_est_final = optimal_PM;		
        		
    end
        

    hit_rate = (sum(sum(s == Qest_final))/(N*dim));
    fprintf('Percentage of right state estimates %.2f %%\n', hit_rate*100);

    %for i=1:1%N
    %    figure
    %    plot(x(:,i));
    %    hold on
    %    plot(lambdas_est_final(Qest_final(:,i)));
    %end
    %mse?
    total_ll = optimal_model{7};
    it = optimal_model{8};
    
    results_mmdl{iter,1} = lambdas_est_final';
    results_mmdl{iter,2} = PAI_est_final;
    results_mmdl{iter,3} = A_est;
    results_mmdl{iter,4} = B_est_final;
    %% REPLACED results{iter,5} = P_est_final; BY
    results_mmdl{iter,5} = PM_est_final;
    results_mmdl{iter,6} = Qest_final;
    results_mmdl{iter,7} = hit_rate;
    results_mmdl{iter,8} = total_ll;
    results_mmdl{iter,9} = ir;

end



%%
% keySet = {'n_states','length_dur','max_obs','original_lambdas', 'original_PAI', 'original_A', 'original_P', 'original_B', 'N', 'dim', 'Vk', 'obs_seq', 'state_seq', 'lambdas_est', 'PAI_est_final', 'A_est', 'B_est_final', 'P_est_final', 'Qest_final', 'lkh', 'll', 'elapsed_time',  'hit_rate', 'total_ll', 'iterations', 'max_iterations'};
% valueSet = {n_states, length_dur, max_obs, original_lambdas, original_PAI, original_A, original_P, original_B, N, dim, Vk, obs_seq, state_seq, lambdas_est, PAI_est_final, A_est, B_est_final, P_est_final, Qest_final, lkh, ll, elapsed_time, hit_rate, total_ll, ir, 500};
% iteration = iteration +1;
% 
% M = containers.Map(keySet,valueSet);
% save(['results_' num2str(iteration) '.mat'],'M');
% clear M;
