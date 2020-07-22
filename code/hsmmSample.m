function [Vk, observed, hidden] = hsmmSample(varargin)
%function [sim, observed, hidden] = hsmmSample(PAI,A,P,B,len,nsamples)
% or
%function [sim, observed, hidden] = hsmmSample(model,len,nsamples)
%
%   sim - simulated data with DEcanonization - Vk(Observation)
%   observed - Observation
%   hidden - hidden states
D=0;
if nargin == 3
    model = varargin{1};          % 1st param Model
    len = varargin{2};            % 2nd param len
    nsamples = varargin{3};       % 3rd param nsamples
else
    model.PAI=varargin{1};
    model.A=varargin{2};
    %% REPLACED 
    %model.P=varargin{3};
    model.PM = varargin{3};
    model.B=varargin{4}; %emission
    model.lambdas=varargin{5};
    len = varargin{6};
    nsamples = varargin{7};
end

SetDefaultValue(3, 'nsamples', 1);
E = model.B;    %define emission matrix
lambdas = model.lambdas;
if numel(len) == 1
    len = repmat(len, nsamples, 1);
end
hidden   = cell(nsamples, 1);
observed = cell(nsamples, 1);

for i=1:nsamples
    T = len(i);
    hidden{i}   = rowvec(markovMySample(model, T, 1));
    observed{i} = zeros(1, T);
    for t=1:T
        %observed{i}(t) = sampleDiscrete(E(hidden{i}(t), :));
        %%POISSON
        observed{i}(t) = poissrnd(lambdas(hidden{i}(t)));
    end
end
if nsamples == 1
    hidden = hidden{1};
    observed = observed{1};
    [x,~] = sort(observed);
else
    [x,~]=sort([observed{:}]);
end
x=x';
tmp=([1;diff(x)]~=0);
Vk=x(tmp);                  %observable values
model.Vk = Vk;
%sim = model.Vk([observed{:}]);       % save simulated data with canonization


function S = markovMySample(model, len, nsamples)
    % Sample from a markov distribution
    % S is of size nsamples-by-len
    %
    
    % This file is from pmtk3.googlecode.com
    
    if nargin < 3, nsamples = 1; end
    pi = model.PAI;
    A = model.A;
    %P = model.P;
    PM=model.PM;
    
    S = zeros(nsamples, len);
    
    for ii=1:nsamples
        S(ii, 1) = sampleDiscrete(pi);
        %% REPLACED 
        %duration = sampleDiscrete(P(S(ii,1),:));  %pick up the duration for init state BY
        u = rand;
        
        duration = ceil(log(1-u) / log(1-PM(S(ii,1)))); %poissrnd(PM(S(ii,1))) %geornd(PM(S(ii,1)))
        
        S(ii, 1:duration) = S(ii, 1);
        tt=duration+1;
        while tt<=len
            S(ii, tt) = sampleDiscrete(A(S(ii, tt-1), :));
            %% REPLACED 
            %duration = sampleDiscrete(P(S(ii,tt),:));
            u = rand;
            duration = ceil(log(1-u) / log(1-PM(S(ii,tt)))); %poissrnd(PM(S(ii,tt))) %geornd(PM(S(ii,tt)))
            
            S(ii, tt:min(tt+duration-1, len)) = S(ii, tt);
            tt=tt+duration;
        end
        
    end
end

end