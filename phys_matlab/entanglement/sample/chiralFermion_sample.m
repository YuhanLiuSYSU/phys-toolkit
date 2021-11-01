%--------------------------------------------------------------------------
% Chiral fermion model.
%
% This is used to benchmark the reflected entropy (2003.09546)
%
%   There is only one band, the hamiltonian is
%   H = -i/2*\sum_j [\psi_j^\dag \psi_{j+1}-\psi_{j+1}^\dag \psi_j]
%     = \sum_k sin(k) c_k^\dg c_k.
%       The state we are interested is the half filled state (fill k=0 to 
%       k=pi)
%
%   Notice that because of the no-go theorem, c = 1 instead of c = 1/2. 
%       (fermion doubling)
%--------------------------------------------------------------------------
%   - From Expr 1, we benchmark the reflected entropy for general cross
%       ratio. We find that 3(RE-MI)/log(2) has discontinuity from 2 to 1
%       when the cross ratio \eta -> 1.
%
%   - From Expr 2, we extract c = 1 from R = c/3*log(LA*LB/(LA+LB)) + c1 
%       The difference between RE and MI is c/3*log(2) when disp = 0
%       The result is the same for offsetScale = 0 and 0.5 (PBC and APBC)
%%

Expr = 2;
isPlot = 0;

if Expr == 1
% Study the scaling with disp (eta)

    L = 1600;
    % Variable is distance between A and B region
    varAll = 0:1:60;
    offsetScale = 0.5;
    LAB = 300;
    type = [2,3,4];
    
    LA = LAB/2;
    LB = LA;
    eta = LA * LA ./((varAll+LA).*(varAll+LA));
    varPt=length(varAll);
    
    renyi = 2;

elseif Expr == 2
% Adjacent interval. disp = 0, eta = 1.
% Study the scaling with L_A
    
    L = 200;
    dis = 0;
    offsetScale = 0.3;
    % variable is the length of region A 
    varAll = 1:1:L/2-1;
    % 3 for Reflected entropy, 4 for mutual information
    type = [2,3,4]; 
    renyi = 1/2;
    
    varPt=length(varAll);
    
end
%%

dk = 2*pi/L;
k = dk:dk:2*pi;
offset = dk*offsetScale;
k = k + offset;
r = 1:1:L;
bandNum = 1; 

Res = generateRes_(type,varPt,renyi);

for ii=1:varPt

    if Expr == 1
        dis = varAll(ii);
        
    else
        LA = varAll(ii);
        LAB = 2*LA;
        LB = LA;
        
    end
    %%
            
    % total length is length of A,B, plus the displacement
    LT = LAB + dis;
    rSub = r(1:LT);
    corr = zeros(bandNum*LT, bandNum*LT);
    
    for i=1:L/2
        % fill k=0 to k=pi states
        
        k_tmp=k(i);
        prod_t=exp(1i*(k_tmp*rSub))';
        corr=corr+(prod_t*prod_t');
    end

    corr=round(corr/L,8);
    
    %%
    sigmay = [0,-1i;1i,0];
    Gamma = kron((corr-corr.'),eye(2))+kron((eye(bandNum*LT)-corr-corr.'),...
        sigmay);
    
    %%
      
    entPara = struct('type',type,'renyi',renyi,...
        'partition',[2*bandNum*LA, 2*bandNum*LB, 2*bandNum*dis]);
    
    Ent = getEnt_(Gamma, entPara);    
    Res = updateRes_(Res,Ent, type, ii,renyi);
    
    fprintf("%d  \n",ii);
end

%% Save the data
% to use SSH_plot and match convention

Res.L = L;
Res.varAll = varAll;
Res.dis = dis;
Res.LAB = LAB;
Res.eta = eta;
save('default.mat','Res')

