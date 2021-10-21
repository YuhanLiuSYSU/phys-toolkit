% This is the sample code for non-Hermitian SSH model
%
%   Author: Yuhan Liu (yuhanliu@uchicago.edu) 10/01/2021
%   
%   - Use the correlation method. The expression for renyi entropy, reflected
%   entropy, and logarithmic negativity should be modified for
%   non-Hermitian case. See the getEnt_ file.

%   - NOTE: The modification for reflected entropy is in python file (not
%   shown here). Re-writing the "left/right eigenvector solver" in Matlab is
%   too time-consuming so I am not going to update this part. I will move
%   all the codes to python later.

%%
L = 40;
% IsExamWhole is to verify \mathcal{E}=S^{(1/2)} at pure state case
isExamWhole = 0;

if isExamWhole == 0
    subAB = (2:2:L-2)';
else
    subAB = (20:20:L)';
end
% subAB = (10:10:50)'; % debug

shift = 0;
%------------choose the offset------------%
offset = 0.0;
%offset = pi/(L);
%offset = 0.0000001;
%------------choose the model-------------%
flagm = 1;

%w = 1;v = 1;u = -1;
%w = 0.8;v = 0.7;u = 0.5;

w = 1.300; v = 1.8+10^(-8); u = 0.5;    % non-Hermitian first critical point
isNonHerm = 1;

%w = 2;v = 2.5;u = 0;flag = 2    %    Lee's model 

% w = 1.8+10^(-8);v = 1.3;u = 0.5;    % non-Hermitian second critical point
% isNonHerm = 1;

% w = 1.0000000;v = 1.0;u = 0;          % Hermitian critical point
% isNonHerm = 0;

%w = 1;v = 0.1;u = 0.5;      % 2 mid-gap states, PT topological phase
%w = 0.1;v = 1;u = 0.5;      % 4 mid-gap states, trivial phase
%w = 1.20;v = 1.8000000001;u = 0.6;
%-----------------------------------------%

renyi = 1/2;
type = [2,4,5];

%%
k = (1:L)*2*pi/L; 
subM = max(subAB);

Res = generateRes_(type,length(subAB)-shift,renyi);

% Eta = zeros(subM*2,length(subAB)-shift);
%%
corr = zeros(2*subM,2*subM);

for ik = 1:L
    k_t = k(ik)+offset;
    [W] = SSH_findv_(k_t,u,v,w,flagm);
    if abs(w*exp(-1i*k_t)+v) >= u
        prod_t = exp(1i*(k_t*(1:subM)))';
        corr = corr+kron((prod_t*prod_t'),W);
        
    end
end

corr = corr/L;
sigmay = [0,-1i;1i,0];
Gamma = kron((corr-corr.'),eye(2))+kron((eye(2*subM)-corr-corr.'),sigmay); 
%%

for ii = 1:length(subAB)-shift
    LAB = subAB(ii);    
    LA = LAB/2; LB = LA;
    
    if exist('renyi','var')
        
        if isExamWhole == 0
            entPara = struct('type',type,'partition',[4*LA,4*LB],...
                'isNonHerm',isNonHerm,'renyi',renyi);
        else
            % Examine the case where A\cup B is the whole system
            entPara = struct('type',type,'partition',[4*LA,4*(L-LA)],...
                'isNonHerm',1,'renyi',renyi);
        end
                
    else
        entPara = struct('type',type,'partition',[4*LA,4*LB],...
            'isNonHerm',isNonHerm);
    end

    
    if shift == 0
        
        if isExamWhole == 0
            GammaR = Gamma(1:4*LAB,1:4*LAB);
        else
            GammaR = Gamma(1:4*L,1:4*L); 
        end

    else
        GammaR = Gamma(1+2:4*LAB+2,1+2:4*LAB+2);
    end
    Ent = getEnt_(GammaR, entPara);
    Res = updateRes_(Res,Ent, type, ii,renyi);
       
end

save('default.mat','Res')
varAll = subAB/2;
