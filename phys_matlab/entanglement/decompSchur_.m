function[QS,TS,Lambda] = decompSchur_(K)

% This is used in fstring and fvertex. 
% Block diagonalize anti-symmetric REAL matrix to canonical form

% Output: TS is in the block form [0,lambda;-lambda,0].
%         Using the output, K = QS.'*TS*QS;
%         Lambda is non-negative. Example: 
%           Lambda = [lambda_1,lambda_1,lambda_2,lambda_2,...].'
%

isReal = sum(sum(abs(imag(K))));
if isReal > 10^(-6)
    disp(isReal)
    disp('ERROR: (Schur) The K-matrix is not REAL!'); 
    
end

K = real(K);

isSkewSym = sum(sum(abs(K+K.')));
if isSkewSym > 10^(-6)
    disp(isSkewSym)
    disp('ERROR: (Schur) The K-matrix is not anti-symmetric!'); 
   
end

K = (K - K.')/2;
   

[Q,T] = schur(K);
NN = length(Q);
M = eye(NN);
Lambda = zeros(NN,1);

for i=1:NN/2
    
    if T(2*i-1,2*i)<T(2*i,2*i-1)
       M(2*i-1:2*i,2*i-1:2*i)=[0,1;1,0]; 
    end
    
    Lambda(2*i-1)=abs(T(2*i-1,2*i));
    Lambda(2*i)=Lambda(2*i-1);
end

TS = M*T*M;
QS = Q*M;

QS = QS.';
% K = QS.'*TS*QS;

end