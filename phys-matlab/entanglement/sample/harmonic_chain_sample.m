%--------------------------------------------------------------------------
% This is the sample code for bosonic harmonic chain model
%
%   Author: Yuhan Liu (yuhanliu@uchicago.edu) 10/14/2021
%   
% Bosonic model of a harmonic chain. Omega should be zero, but to avoid
% divergence, it is assigned a small number. The dispersion relation is
% linear around k = 0.

% Ref: 1607.02992
% Reproduce c = 1 from entanglement scaling.
%--------------------------------------------------------------------------

%% Parameter

Expr = 1;

if Expr == 1
    
    L = 1000;   
    % Variable is the distance between region A and B.
    varAll = 0:1:40;
    
    LA = 100;
    eta = LA * LA ./((varAll+LA).*(varAll+LA));
    
    varPt = length(eta);
    
elseif Expr == 2
    
    L = 2000;
    % LA is the size of region A.
    dis = 0;
    varAll = 1:1:200;
    varPt = length(varAll);
    
end


renyi = 1/2;
type = [2, 3, 4];


% TODO: debug for finite seperation. The result does not agree with CFT at
% this point...


Omega = 10^(-5);

zeroBlock = zeros(L,L);
siteList = (1:L)';

Res = generateRes_(type,varPt,renyi);

%% Compute the covariance matrix
Omegak= @(k) sqrt(Omega^2+4*(sin(pi*k/L))^2);

posiArray = siteList-siteList';
X = zeros(L,L);
P = zeros(L,L);

for k = 1:L-1
   cosFactor = cos(2*pi*k/L*posiArray);
   X = X + 1/(2*L*Omegak(k))*cosFactor;
   P = P + Omegak(k)/(2*L)*cosFactor;
%    disp(k)
end

X = X + 1/(2*L*Omegak(0));
P = P + Omegak(0)/(2*L);

disp('done')
%% Compute and record the entanglement
for iVar = 1:varPt
    
    if Expr == 1
        
        dis = varAll(iVar);
        lTotal = 2*LA + dis;
        
    elseif Expr == 2
        
        % take lB = lA
        LA = varAll(iVar);
        lTotal = 2*LA + dis;
        
    end
          
    %---------------------------------------------------------------
%     sigmaR = [2*X(1:lAB,1:lAB), zeroBlock(1:lAB,1:lAB);
%               zeroBlock(1:lAB,1:lAB), 2*P(1:lAB,1:lAB)];
%     entPara = struct('isBlockZero', 0, 'renyi', renyi, 'type', type,...
%                      'partition', [lA, lA]);
%     Ent = getEntBoson_(sigmaR, entPara);
    %---------------------------------------------------------------
    
    entPara = struct('isBlockZero', 1, 'renyi', renyi, 'type', type,...
                     'partition', [LA, LA, dis]);
    Ent = getEntBoson_({X(1:lTotal,1:lTotal), ...
                        P(1:lTotal,1:lTotal)}, entPara);
    
    Res = updateRes_(Res,Ent, type, iVar,renyi);

    disp(iVar)
    
end
%% Save the data

Res.L = L;
Res.varAll = varAll;
Res.dis = dis;
Res.LA = LA;
Res.eta = eta;

save('default.mat','Res')

