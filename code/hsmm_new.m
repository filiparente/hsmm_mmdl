%% REPLACED function [lambdas, lambdas_tt, PAI,A,B,P,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,P,MO,IterationNo,MT, tolerance, Vk)
function [lambdas, lambdas_tt, PAI,A,B,PM,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,PM, MO,IterationNo,MT, tolerance, Vk)
% 
% Author: Shun-Zheng Yu
% Available: http://sist.sysu.edu.cn/~syu/Publications/hsmm_new.m
% 
% HSMM solve three fundamental problems for Hidden Semi-Markov Model using a new Forward-Backward algorithm
% Usage: [PAI,A,B,P,Stateseq,Loglikelihood]=hsmm_new(PAI,A,B,P,O,IterationNo,MT)
% MaxIterationNo=0: estimate StateSeq and calculate Loglikelihood only; 
% MaxIterationNo>1: re-estimate parameters, estimate StateSeq and Loglikelihood.
% First use [A,B,P,PAI,Vk,O,K]=hsmmInitialize(O,M,D,K) to initialize
% 
% Ref: Practical Implementation of an Efficient Forward-Backward Algorithm for an Explicit Duration Hidden Markov Model
% by Shun-Zheng Yu and H. Kobayashi
% IEEE Transactions on Signal Processing, Vol. 54, No. 5, MAY 2006, pp. 1947-1951 
% 
%  This program is free software; you can redistribute it and/or
%  modify it under the terms of the GNU General Public License
%  as published by the Free Software Foundation; either version
%  2 of the License, or (at your option) any later version.
%  http://www.gnu.org/licenses/gpl.txt
%  
%  Updated Nov.2014
%
%  N               Number of observation sequences
%  MT              Lengths of the observation sequences: MT=(T_1,...,T_N)
%  MO              Set of the observation sequences: MO=[O^1,...,O^N], O^n is the n'th obs seq.
%++++++++ Markov Model +++++++++++
M=length(PAI);               %The total number of states
N=size(MO,2);                 %Number of observation sequences
%D=size(P,2);                 %The maximum duration of states BY D IN FUNCTION SINTAX
K=size(B,2);                 %The total number of observation values
T=size(MO,1);
%----------------------------------------------------

%Obtain the maximum allowed duration from the
%duration parameters
%duration such that the probability of P(D>dmax) is very small (e.g.: 0.01)
%geometric dist: P(D>dmax)=1-P(D<=dmax)=1-(1-(1-p)^dmax) = 0.01
%dmax=log(0.01)/log(1-p).

dmax = floor(log(0.001)/log(1-min(PM)));
D = dmax;

ALPHA=zeros(M,D);
bmx=zeros(M,T);
S=zeros(M,T);
E=zeros(M,T);
BETA=ones(M,D);
Ex=ones(M,D);
Sx=ones(M,D);
GAMMA=zeros(M,1);
Pest=zeros(M,D);
Aest=zeros(M,M);
Best=zeros(M,K);
PAIest=zeros(M,1);
Qest=zeros(T, N);
d=[1:D];
store_GAMMA=zeros(M,T,N);
lkh=zeros(1,N);
ll = [];
lambdas_tt = [];

ir1=max(1,IterationNo);

for ir=1:ir1
    tic
    fprintf('Iteration nº: %d\n', ir);
    %dmax = floor(log(0.001)/log(1-min(PM)));
    %D = dmax;
    %d = [1:D];
    Pest=zeros(M,D);
    Aest=zeros(M,M);
    Best=zeros(M,K);
    PAIest=zeros(M,1);
    %Aest(:)=0;
    %Pest(:)=0;
    %Best(:)=0;
    %PAIest(:)=0;
    for on=1:N	 % for each observation sequence

        
        O=MO(:,on);	 % the n'th observation sequence
        T=MT(on);        % the length of the n'th obs seq
        
        %    starttime=clock;
        %++++++++++++++++++     Forward     +++++++++++++++++
        %---------------    Initialization    ---------------
        %% REPLACED ALPHA(:)=0; ALPHA=repmat(PAI,1,D).*P;		%Equation (13) 
        clear P;
        for aux=1:M
            P(aux,:) = PM(aux)*(1-PM(aux)).^(d-1); %poisspdf(d,PM(aux)+1e-100); %geopdf(d,PM(aux)); 
            P(aux,:)=P(aux,:)/sum(P(aux,:));
        end
            ALPHA(:)=0; ALPHA=repmat(PAI,1,D).*P;
        
        r=((poisspdf(O(1),lambdas)+1e-100)*sum(ALPHA,2));			%Equation (3)
        
        bmx(:,1)=(poisspdf(O(1),lambdas)+1e-100)./r;				%Equation (2)
        E(:)=0; E(:,1)=bmx(:,1).*ALPHA(:,1);		%Equation (5)
        S(:)=0; S(:,1)=A'*E(:,1);			%Equation (6)
        lkh(on)=log(r);
        %---------------    Induction    ---------------
        for t=2:T
            ALPHA=[repmat(S(:,t-1),1,D-1).*P(:,1:D-1)+repmat(bmx(:,t-1),1,D-1).*ALPHA(:,2:D),S(:,t-1).*P(:,D)];		%Equation (12)
            r=((poisspdf(O(t),lambdas)+1e-100)*sum(ALPHA,2));		%Equation (3)
            bmx(:,t)=(poisspdf(O(t),lambdas)+1e-100)./r;			%Equation (2)
            E(:,t)=bmx(:,t).*ALPHA(:,1);		%Equation (5)
            S(:,t)=A'*E(:,t);				%Equation (6)
            lkh(on)=lkh(on)+log(r);
        end
        %++++++++ To check if the likelihood is increased ++++++++
        if ir>1
            %    clock-starttime
%             if(abs(lkh1(on))<abs(lkh(on))) %likelihood did not increase
%                 disp('ERROR: likelihood did not increase in this iteration');
%             end
        end
        
        
        %++++++++ Backward and Parameter Restimation ++++++++
        %---------------    Initialization    ---------------
        
        Aest=Aest+E(:,T)*ones(1,M);  %Since T_{T|T}(m,n) = E_{T}(m) a_{mn}
        GAMMA=bmx(:,T).*sum(ALPHA,2);
        if IterationNo>0 && O(T)~=0
            Best(:,min(find(O(T)==Vk)))=Best(:,min(find(O(T)==Vk)))+GAMMA;
        end
        [X,Qest(T,on)]=max(GAMMA);
        store_GAMMA(:,T,on)=GAMMA;
        
        BETA=repmat(bmx(:,T),1,D);				%Equation (7)
        Ex=sum(P.*BETA,2);					%Equation (8)
        Sx=A*Ex;						%Equation (9)
        
        %---------------    Induction    ---------------
        for t=(T-1):-1:1
            %% for estimate of A
            Aest=Aest+E(:,t)*Ex';
            %% for estimate of P
            Pest=Pest+repmat(S(:,t),1,D).*BETA;
            %% for estimate of state at time t
            GAMMA=GAMMA+E(:,t).*Sx-S(:,t).*Ex; %equation (16)
            GAMMA(GAMMA<0)=0;           % eleminate errors due to inaccurace of the computation.
            [X,Qest(t,on)]=max(GAMMA);
            store_GAMMA(:,t,on)=GAMMA;
            %% for estimate of B
            %Best(:,O(t))=Best(:,O(t))+GAMMA;
            if IterationNo>0 && O(t)~=0
                Best(:,min(find(O(t)==Vk)))=Best(:,min(find(O(t)==Vk)))+GAMMA;
            end
            
            BETA=repmat(bmx(:,t),1,D).*[Sx,BETA(:,1:D-1)];	%Equation (14)
            [xx,yy]=find(BETA==Inf);
            if ~isempty(xx) && ~isempty(yy)
                BETA(xx,yy) = 1e+308;
            end
            Ex=sum(P.*BETA,2);					%Equation (8)
            Sx=A*Ex;						%Equation (9)
            if(any(any(isnan(Aest))))
                disp('erro');
            end
            if(any(any(isnan(Pest))))
                disp('erro');
            end
            if(any(any(isnan(Best))))
                disp('erro');
            end
            
        end
        
        Pest=Pest+repmat(PAI,1,D).*BETA;    %Since D_{1|T}(m,d) = \PAI(m) P_{m}(d) \Beta_{1}(m,d)
        PAIest=PAIest+GAMMA./sum(GAMMA);
    end % End for multiple observation sequences
    
    
    
    if ir>1
        if ((lkh-lkh1)/abs(lkh1))<tolerance
            %Stop if the increase in the likelihood is too small
            lkh1=lkh;
            ll = [ll;lkh]; %add the likelihood for the N observations sequences in the current iteration
            lambdas_tt = [lambdas_tt;lambdas];
            disp('Likelihood increase too small. Exiting...');
            break
        end
    end
    lkh1=lkh;
    ll = [ll;lkh]; %add the likelihood for the N observations sequences in the current iteration
    
    if IterationNo>0            % re-estimate parameters
        Aest=Aest.*A;  
        idxs = find(round(sum(Aest,2))==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.
        if ~isempty(idxs)
            %FIRST OPTION: REMOVE
            %remove them from all estimated parameters
            
            %Aest(idxs,:)=[];%ones(M,1)/M;
            %Aest(:,idxs)=[];
            %Best(idxs,:)=[];
            %Pest(idxs,:)=[];
            %P(idxs,:)=[];
            %PAIest(idxs)=[];
            %M=M-length(idxs);
            %ALPHA=zeros(M,D);
            %bmx=zeros(M,T);
            %S=zeros(M,T);
            %E=zeros(M,T);
            %BETA=ones(M,D);
            %Ex=ones(M,D);
            %Sx=ones(M,D);
            %GAMMA=zeros(M,1);
            
            %SECOND OPTION: SET TO UNIFORM DISTRIBUTION
            Aest(idxs,:)=ones(length(idxs),M)/M;
            Best(idxs,:)=ones(length(idxs),K)/K;
            Pest(idxs,:)=ones(length(idxs),D)/D;

            clear idxs;
        end
        PAI=PAIest./sum(PAIest);
        A=Aest./repmat(sum(Aest,2),1,M);
        
        %smoothing transition probabilities
        %A = smooth_transition_probs(A,M);
        
        B=Best./repmat(sum(Best,2),1,K);
        clear lambdas;
        for i=1:size(B,1)
            lambdas(i) = B(i,1:end)*Vk;
%             for on=1:N
%                 O=MO(:,on);
%                 for el=1:length(O)
%                     lambdas(i) = lambdas(i) + O(el)*store_GAMMA(i,el,on);
%                 end
%             end
%             lambdas(i)=lambdas(i)/(N*T);
        end
        %% REPLACED Pest=Pest.*P;   P=Pest./repmat(sum(Pest,2),1,D);
        Pest=Pest.*P; 
        Pest=Pest./repmat(sum(Pest,2),1,D); 
        den = Pest.*repmat([1:D]', 1,M)';
        den = sum(den,2);
        PM = 1./den;
        %% OU
        %Pest=Pest.*P; den = Pest.*repmat([1:D]', 1,M)'; den = sum(den,2);
        %PM = sum(store_GAMMA(:,1:end-1), 2)./den;
        %PM = den;
        %for kk=1:M
        %    PM(kk) = (Pest(kk,:)/sum(Pest(kk,:)))*[1:D]';
        %end
        %PM(PM>1)=1-1e-16;
        if any(any(isnan(A)))
            disp('erro A');
        end
        if any(any(isnan(B)))
            disp('erro B');
        end
        if any(any(isnan(P)))
            disp('erro P');
        end
        if any(any(isnan(PAI)))
            disp('erro PAI');
        end
        
        lambdas_tt = [lambdas_tt;lambdas];
    end
    toc
end
end