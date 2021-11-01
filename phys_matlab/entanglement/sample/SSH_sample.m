%--------------------------------------------------------------------------
% This is the sample code for SSH model
%
%   Author: Yuhan Liu (yuhanliu@uchicago.edu) 08/25/2021
%   
%   Use the correlation method. 
%   - In the non-trivial phase, S = 2log(2) while \mathcal{E} = log(2).
%   - Zero mode in the entanglement spectrum for non-trivial phase.
%   - At critical value u = 0, S = c/3*log(sin((2*LA)*pi/N))+c1
%
%   Expr == 1: scaling on u to observe the phase transition
%   Expr == 2: scaling on LA at u=0 to extract c.
%              The difference RE - MI = c*log(2)/3. (c = 1)
%   Expr == 3: scaling on eta(cross ratio) 
%
%--------------------------------------------------------------------------
%% Parameter

Expr = 2;
isPureVerify = 1;

if Expr == 1
% For scaling on u

    % variable is u
    % L is number of unitcells. LA is number of unitcells in region A
    varAll = -1:0.02:1;    
    L = 150;
    LA = 20; LB = 20;
    type = 2;
    dis = 0;
   
    % This is used to verify S^{(n)} - R^{(n)}/2 =0 for pure state
    % test = RenyiA - RenyiRefEnt/2; 
    % test = xi_AB-RenyiRefEnt/2;
    %-----------------------------
    if isPureVerify == 1
        varAll = -1:0.5:1; 
        L = 80;
        LA = 40; LB = 40;
        type = [2,3,4,5];
        renyi = 1/2;
    end
    %------------------------------
     
elseif Expr == 2
% For scaling on LA at the critical point u = 0

    u = 0;
    L = 40;
    % variable is LA, which is number of unitecells in region A
%     varAll = L/2-1:(-1):1; 
    varAll = 1;

    % 3 for RE, 4 for MI
    type = [2,3,4];
    dis = 0;
          
    % If we don't want to compute renyi entropy, just comment out the
    % following line.
    renyi = 1/2; 

elseif Expr == 3
    % scale over the cross ratio eta
    
    L = 3000;
    u = 0;
    % variable is the distance between two intervals
    varAll = 40:(-1):0;
    LAB = 500;
    LA = LAB/2;
    type = [3,4];
    renyi = 1/2;
    eta = LA * LA ./((varAll+LA).*(varAll+LA));
    
end

dk = 2*pi/L;
% offset = dk/2 would change to APBC
offset = 0; 
k = dk:dk:2*pi; k = k-offset;
r = 1:1:L;

%% Compute

varPt = length(varAll);
% xiS_AB = cell(1,varPt);
% ES_AB = cell(1,varPt);

Res = generateRes_(type,varPt,renyi);

for ii=1:varPt
    
    if Expr == 1
        
        t1 = (1-varAll(ii))/2;
        t2 = (1+varAll(ii))/2;  
        LTotal = LA+LB;
        rSub = r(1:LTotal);
        
    elseif Expr == 2
        
        t1 = 1/2; t2 = 1/2;
        LA = varAll(ii);
        LB = LA;
        
        LTotal = LA+LB;
        rSub = r(1:LTotal);
        
    else
        dis = varAll(ii);
        t1 = 1/2; t2 = 1/2;
        LTotal = LA*2+dis;
        LB = LA;
        rSub = r(1:LTotal);
              
    end
    
    if (ii == 1) || (Expr == 1)
        % 2 comes from two atoms in one unitcell
        Corr=zeros(2*LTotal,2*LTotal);

        for i=1:L
            kTmp = k(i);
            [Ak,Bk] = SSH_findeig_(t1,t2,kTmp);
            wf = [Ak;Bk];
            prod_t = exp(1i*(kTmp*rSub))';
            Corr = Corr+kron((prod_t*prod_t'),conj(wf*wf'));
        end

        Corr = round(Corr/L,8);
    end
    
    corr = Corr(1:2*LTotal,1:2*LTotal);

    sigmay = [0,-1i;1i,0];
    Gamma = kron((corr-corr.'),eye(2))...
        +kron((eye(2*LTotal)-corr-corr.'),sigmay);   
%%
    % For region A+B
    if exist('renyi','var')
        entPara = struct('type',type,'partition',[4*LA,4*LB,4*dis],...
            'renyi',renyi);
    else
        entPara = struct('type',type,'partition',[4*LA,4*LB,4*dis]);
    end
    
    Ent = getEnt_(Gamma, entPara);
    Res = updateRes_(Res,Ent, type, ii,renyi);
   
    fprintf("%d  \n",ii);
end

%% save the data
Res.varAll = varAll;
Res.L = L;
Res.LAB = LAB;
Res.u = u;
save('default.mat','Res')
