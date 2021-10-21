function [Ent] = getEntBoson_(inputMatrix, entPara)
%
%--------------------------------------------------------------------------
% Covariance matrix method to compute entanglement quantities for bosonic
% system. Benchmarked by harmonic_chain_sample.m
%
%   AUTHOR:
%       Yuhan Liu (yuhanliu@uchicago.edu)
%
%   Input: inputMatrix - covariance matrix
%          entPata - entanglement parameter
%   Output: Ent - a field data
%   
%   If isBlockZero == 0, using covariance matrix sigma
%       sigma is defined by tr(rho (xi_i xi_j + xi_j xi_i))
%   If isBlockZero == 1, using two correlation matrix, X and P
%       X is defined by tr(rho (q_i q_j + q_j q_i))/2
%       P is defined by tr(rho (p_i p_j + p_j p_i))/2
%       Note there is a convention of 1/2.
%
%   EXAMPLE INPUT:
%     entPara = struct('isBlockZero', 1);
%     Ent = getEntBoson_({X, P}, entPara);
%--------------------------------------------------------------------------

if isfield(entPara,'partition')
    partition = entPara.partition;
       
    if length(partition)==2
        NA = partition(1);
        NB = partition(2);
        disp = 0;
    else
        
        % disp is the distance between region A and region B
        % the length of X, P is NA+NB+d
        NA = partition(1);
        NB = partition(2);
        disp = partition(3);
    end     
        
end

if isfield(entPara,'type')
	type = entPara.type;
else
    type = 1;
end

if isfield(entPara,'renyi')
    renyi = entPara.renyi;
end

if isfield(entPara,'isBlockZero')
	isBlockZero = entPara.isBlockZero;
    
else
    isBlockZero = 0;
end 

%% Transformation
if isBlockZero == 0

    sigma = inputMatrix;
    N = length(sigma)/2;
    % # of sites of a single string
    % sigma is for two strings.

    omega = kron([0,1;-1,0], eye(N));
    J = -1i*sigma*omega;
    
    % TODO: check whether eigenvalues are real
    % For a bona fide covariance matrix, we need mu>=1
    mu = sort(real(eig(J)));

else
    % In this case, the input is sigma = X*P
    
    X0 = inputMatrix{1};
    P0 = inputMatrix{2};
    
    [XA,XAB,XBA,XB] = partitionMatrix_(X0,NA,NB,disp);
    [PA,PAB,PBA,PB] = partitionMatrix_(P0,NA,NB,disp);
    
    if disp>0
        X = [XA, XAB; XBA, XB];
        P = [PA, PAB; PBA, PB];
%         C0 = X0*P0*4;
    else
        X = X0;
        P = P0;
    end
        
    C = X*P*4;
    CA = XA*PA*4;
    CB = XB*PB*4;
    
    mu = sqrt(eig(C));
    
end
    
%% Entanglement entropy
Ent.S = getS_(mu);

if exist('renyi','var')
        
    Renyi = getSRenyi_(mu,renyi);
    Ent.Renyi = Renyi;
end

%% Logarithmic negativity

if isBlockZero == 0
    %P = diag([ones(1, NSingle),-ones(1,NSingle),ones(1, 2*NSingle)]);
    R = diag([ones(1, NA),-ones(1,NA),ones(1, 2*NB)]);
    % DEBUG FROM HERE...

    sigmaTr = R*sigma*R;
    JTr = -1i*sigmaTr*omega;
    nu = sort(real(eig(JTr)));
    
else
    
    R = diag([-ones(1,NA),ones(1,NB)]);
    CTr = X*(R*P*R)*4;
    nu = sqrt(sort(real(eig(CTr))));
    
end

% TODO: CHECK THE DISTRIBUTION OF NU
nu = nu(nu>=0);
nuNega = nu(nu<1);
xi = real(log(prod(1./nuNega)));

Ent.xi = xi;

Ent.mu = mu;
Ent.nuNega = nuNega;
Ent.nuPosi = nu(nu>1);


%% Reflected entropy
if any(type == 3)
% reference: 2008.11373
    
    XP = X*P;
    g = (XP-1/4*eye(length(X)))^(1/2)/((XP)^(1/2));
    gX = g*X;
    Pg = P*g;
    
    Phi = [X(1:NA,1:NA), gX(1:NA,1:NA); 
           gX(1:NA,1:NA), X(1:NA,1:NA)];
    Pi = [P(1:NA,1:NA), -Pg(1:NA,1:NA);
          -Pg(1:NA,1:NA), P(1:NA,1:NA)];
    
    CRef = Phi*Pi*4; 
    muARef = sqrt(eig(CRef));
    Ent.refEnt = getS_(muARef);
    
    if exist('renyi','var')
        Ent.RenyiRefEnt = getSRenyi_(muARef, renyi);
    end

end

%% Mutual information
if any(type == 4)

    SA = getS_(sqrt(eig(CA)));
    SB = getS_(sqrt(eig(CB)));
    
    RenyiA = getSRenyi_(sqrt(eig(CA)),renyi);
    RenyiB = getSRenyi_(sqrt(eig(CA)),renyi);
    
%     if disp > 0
%         Ent.MI = SA + SB - getS_(sqrt(eig(C0)));
%     else
%         Ent.MI = SA + SB - Ent.S;
%     end
    
    Ent.MI = SA + SB - Ent.S;
    
    
    Ent.MIRenyi = RenyiA + RenyiB - Renyi;
    
end

%% Examine the region A. Useful for pure state limit
if any(type == 5)
    % This is used to verify S^{(n)} = S_R^{(n)}/2 for pure state
    
    muA = sqrt(eig(CA));   
    Ent.SA = getS_(muA);
        
    if exist('renyi','var')              
        Ent.RenyiA = getSRenyi_(muA,renyi);
                
    end
    
end

end


function [S] = getS_(mu)
% Obtain the von Neumann entropy

    mu = mu(mu>1+10^(-6));
    S = (mu/2+1/2)'*log(mu/2+1/2)-(mu/2-1/2)'*log(mu/2-1/2);

end

function [Renyi] = getSRenyi_(mu,renyi)
% Obtain the renyi entropy

    mu = mu(mu>1+10^(-6));
    Renyi = -1/(1-renyi)*sum(log((mu/2+1/2).^renyi-(mu/2-1/2).^renyi));


end

function [XA,XAB,XBA,XB] = partitionMatrix_(X,NA,NB,disp)

    XA = X(1:NA, 1:NA);
    XAB = X(1:NA, disp+NA+1:disp+NA+NB);
    XBA = X(disp+NA+1:disp+NA+NB, 1:NA);
    XB = X(disp+NA+1:disp+NA+NB, disp+NA+1:disp+NA+NB);

end
