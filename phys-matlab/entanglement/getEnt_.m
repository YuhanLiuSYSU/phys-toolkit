% This code compute various entanglement quatities using correlation
% method. It works for fermionic system. For bosonic system, see
% getEntBoson_.m
% 
%
%   Author: Yuhan Liu (yuhanliu@uchicago) 2021
%
%   Input: GammaR - correlation matrix in the subregion 
%          optional input(varagin) - compute type; 
%          subregion A and subregion B for tripartition.
%           Typical input is (GammaR, 2, [4*lA, 4*lB]) for SSH
%
%   Output: Ent - a field data.

%%
function [Ent] = getEnt_(GammaR, entPara)


% varargin: type : 1 -- only compute EE
%                  2 -- compute EE, negativity
%                  3 -- compute EE, reflected entropy
%                  4 -- compute EE, mutual information
%           way of partition 

% switch nargin
%     case 1               % only has GammaR
%         type = 1;        % only compute EE
%     case 2
%         type = varargin{1};
%     case 3
%         type = varargin{1};
%         partition = varargin{2};
%         
%         if length(partition)==2
%             NA = partition(1);
%             NB = partition(2);
%         else
%             
%             % d is the distance between region A and region B
%             % the length of GammaR is NA+NB+d
%             NA = partition(1);
%             NB = partition(2);
%             d = partition(3);
%         end     
%                    
% end

%------------------------------------------------------------------
% new version (09/25/2021): use entPara instead of varargin
%             (10/10/2021): include the non-Hermitian treatment
%------------------------------------------------------------------

d = 0; % distance between region A and region B


if isfield(entPara,'type')
	type = entPara.type;
else
    type = 1;
end

if isfield(entPara,'partition')
    partition = entPara.partition;
    
    if length(partition)==2
        NA = partition(1);
        NB = partition(2);
        
    else
        
        % d is the distance between region A and region B
        % the length of GammaR is NA+NB+d
        NA = partition(1);
        NB = partition(2);
        d = partition(3);
    end     
end


if isfield(entPara,'isNonHerm')
	isNonHerm = entPara.isNonHerm;
else
    isNonHerm = 0;
end 


if isfield(entPara,'renyi')
    renyi = entPara.renyi;
end

%%
if isNonHerm == 0
    GammaR = 1/2*(GammaR+GammaR'); % GammaR should be Hermitian
end

if any(type > 1) 
    
    G11 = GammaR(1:NA, 1:NA);
    G12 = GammaR(1:NA, d+NA+1:d+NA+NB);
    G21 = GammaR(d+NA+1:d+NA+NB, 1:NA);
    G22 = GammaR(d+NA+1:d+NA+NB, d+NA+1:d+NA+NB);

    if d > 0
        GammaR = [G11, G12; G21, G22];
    end
end

%% EE for 1\cup 2
% EE is always computed 

[oV,eigGammaR] = eig(GammaR);
eigGammaR = diag(eigGammaR);

Ent.S = getS_(eigGammaR, isNonHerm);
Ent.ES = eigGammaR;

if exist('renyi','var')
        
    Renyi = 1/(1-renyi)*getSR_(renyi,eigGammaR,isNonHerm);
    Ent.Renyi = Renyi;
end

% ES = sort(real(log(2./(oD+1)-1))); 
    
%% Negativity 
if any(type == 2) 
        
    Gp = [-G11,1i*G12;1i*G21,G22];
    Gm = [-G11,-1i*G12;-1i*G21,G22];
    Id = eye(length(Gm));
    Gc = Id-(Id-Gm)/(Id+Gp*Gm)*(Id-Gp);
    
    if isNonHerm == 0
        Gc = 1/2*(Gc+Gc');
    end
    
    [cV,cD] = eig(Gc);
    
    
    if isNonHerm == 0
        
        cD = real(diag(cD));
        R1 = getSR_(0.5, cD,isNonHerm);
        R2 = getSR_(2, eigGammaR,isNonHerm);
        xi = R1+R2/2; 
        
    else
        
        cD = diag(cD);
        % The following three lines are still under debug...
        % Use with care
        
        % ??? WE SHOULDN'T TAKE THE REAL PART OF cD???
        % TODO: It feels like c=-2 is only the coefficient when we take the 
        % real part. This should be examined carefully...
        R1 = getSR_(0.5, cD,isNonHerm);
        R2 = getSR_(2, eigGammaR,isNonHerm);
        xi = R1-R2/2;
              
    end 
    
    xiS = sort(cD); % Negativity spectrum

    Ent.xi = xi;
    Ent.xiS = xiS;
    
    %% spectrum of rho^{T_A}
    
    [eigxipV,eigxipD] = eig(Gp); 
    xipS = diag(eigxipD);
    
    Cont = cell(1,4);
    Cont{1} = cV; Cont{2} = cD; Cont{3} = oV; Cont{4} = eigGammaR;

    Ent.xipS = xipS;
    Ent.eigxipV = eigxipV;
    Ent.Cont = Cont;
      
end 

%% Reflected entropy
if any(type == 3)

    if isNonHerm == 0
        [O,tmp,gamma] = decompSchur_(1i*GammaR);

        isDecomp = sum(sum(abs(O.'*tmp*O - 1i*GammaR)));
        if isDecomp > 10^(-6)
            disp(isDecomp);
            disp('ERROR: The Schur decomposition is wrong!'); 
        end
        
        % O.'*tmp*O = 1i*GammaR
        Mtilde = O.'*(diag(sqrt(1-gamma.^2)))*O;
    
    else
        
        %--------------------------------
        % TODO: FINISH THIS PART.
        %--------------------------------
        
        [Q,D,Sigma] = decompYoula_(1i*GammaR); % GammaR = Q.'*Sigma*Q
        
        gamma = sqrt(-D);
        Mtilde = Q.'*(diag(sqrt(1-gamma.^2)))*Q;
        
        
%        [U,D] = myEig(GammaR);
%        Mtilde = O.'*(diag(sqrt(1-gamma.^2)))*O;

        %--------------------------------

    end
    
    
    MTFD = [GammaR(1:NA,1:NA), -1i*Mtilde(1:NA,1:NA);...
            1i*Mtilde(1:NA,1:NA), -GammaR(1:NA,1:NA)];
  
    eigMTFD = eig(MTFD);
        
    refEnt = getS_(eigMTFD,isNonHerm);    
    Ent.refEnt = refEnt;
    
    if exist('renyi','var')
                
        RenyiRefEnt = 1/(1-renyi)*getSR_(renyi, eigMTFD,isNonHerm);
        Ent.RenyiRefEnt = RenyiRefEnt;
        
    end

end

%% Mutual information
if any(type == 4)
    
    eigG11 = eig(G11);
    eigG22 = eig(G22);
    
    SA = getS_(eigG11,isNonHerm);
    SB = getS_(eigG22,isNonHerm);
    
    Ent.MI = SA + SB - Ent.S;

    RenyiA = 1/(1-renyi)*getSR_(renyi,eigG11,isNonHerm);
    RenyiB = 1/(1-renyi)*getSR_(renyi,eigG22,isNonHerm);
    
    Ent.MIRenyi = RenyiA  + RenyiB - Renyi;
    
end

%% Examine the region A. Useful for pure state limit
if any(type == 5)
    % This is used to verify S^{(n)} = S_R^{(n)}/2 for pure state
    
    eigG11 = eig(G11);
    
    SA = getS_(eigG11,isNonHerm);
    Ent.SA = SA;
 
        
    if exist('renyi','var')
                
        RenyiA = 1/(1-renyi)*getSR_(renyi,eigG11,isNonHerm);       
        Ent.RenyiA = RenyiA;
        
    end
    
end

    
end


function [U,D] = myEig(Gamma)

    % Deal with the case where Gamma is anti-symmetric, but not
    % pure-imaginary
    
    [U,D] = eig(Gamma);
    [D, ind] = sort(real(diag(D)));
    U = U(:, ind);

end


function [EE] = getS_(gamma, isNonHerm)
    % gamma is eig(Gamma)
   
    if isNonHerm == 0
        % For the hermitian system, the eigenvalues should be real   
        gamma = real(gamma);

        % eta takes values from 0 to 1
        eta = (gamma+1)/2;
        etaReg = eta(eta > 10^(-9));
        etaReg = etaReg(etaReg < 1-10^(-9));
        
    else
        
        eta = (gamma+1)/2;
        etaReg = eta(abs(eta) > 10^(-9));
        etaReg = etaReg(abs(etaReg-1) > 10^(-9));

    end

    % note the modification of abs()
    % This makes the non-hermitian second critical point correct.  
    EE = -etaReg'*log(abs(etaReg));
   
end


function [R] = getSR_(renyi, oD, isNonHerm)
% This code is for computing the renyi entropy. This works for general 
% cases, both hermitian and non-hermitian.
    
    eC = (oD+1)/2;
    
    % Using a smaller cutoff 10^(-7) makes the result more accurate for the
    % renyi entropy n = 1/2, n = 1/3, etc.
    cutoff = 10^(-7);
    if isNonHerm == 0
        eC = eC(eC > cutoff); eC = eC((1-eC) > cutoff);
    else
        eC = eC(abs(eC) > cutoff); eC = eC(abs(eC-1) > cutoff);
    end
    
    
    
    % note the division over 2 at the end.
    R = log(prod(eC.*abs(eC).^(renyi-1)...
        +(1-eC).*abs(1-eC).^(renyi-1)))/2; 
    
    %debug
%     R = log(prod(eC.^renyi+(1-eC).^renyi))/2;
    
    % old version:
    %--------------------------------------------------------
%     SRenyi = 1/(1-renyi)...
%         *log(prod(abs(abs(eC).^renyi-abs(1-eC).^renyi)))/2; 
    %--------------------------------------------------------


end