function[Q,D,Sigma] = decompYoula_(C)

% ------------------------------------
% YOULA DECOMPOSITION
% 
% For C being a complex anti-symmetric matrix, C = conj(U)*Sigma*U', 
% where U is unitary, and Sigma is block diagonal.
%      Sigma = [0,  lambda]
%              [-lambda, 0], with lambda being real positive-definite.
%
% Ref: A normal form for a matrix 
%      under the unitary congruence group
% -------------------------------------
% To play with it, do:
%      n = 60;
%      rm = rand(n,n) + 1i*rand(n,n);
%      C = rm-rm.';
% -------------------------------------
% The old name of this function is skew_decomp.m

%%
N = length(C);
ccBar = conj(C)*C;
[V,D] = schur(ccBar); %D is real diagonal, -lambda^2. D = V'*ccBar*V(~=V);
[D0,V] = sortD(D,V);
%%
D = real(diag(D0));
U = zeros(N,N);

for i=1:N/2
    
    seed = 1;    
    lambda = sqrt(-D(2*i-1));
    
    if i>1
        if abs(lambda-sqrt(-D(2*i-3)))<10^(-12)
            % degenerate
            seed = seed+1;
        end
    end
    
    % C*a = \lambda*conj(b), conj(C)*conj(b) = -\lambda*a    
    % C\bar{C}*a = -\lambda^2*a
    
    a = V(:,2*i-1);
    b = conj(C*a/lambda);
    
    %------------------------------
    % SANITY CHECK BOARD:
    % C*a-lambda*conj(b) %should equal to 0
    % C*b+lambda*conj(a)
    % norm(ccbar*b+lambda^2*b)
    % norm(ccbar*a+lambda^2*a)
    %------------------------------
    
    % Let x, y satisfy orthogonal relation
    
    if abs(a'*b)<10^(-9) % if they are already orthogonal, do nothing
        mu=0;
    else
        coeff = [a'*b (a'*a-b'*b) -conj(a'*b)];
        mu = roots(coeff); mu = mu(1);
    end
    
    x = a + mu*b; x = x/norm(x);
    y = b-conj(mu)*a; y = y/norm(y);
    
    U(:,2*i-1:2*i)=[x,y];
    
    if seed==2
        % disp("degenerate");
        % See the appendix of unitaryCongruenceTrans.pdf
        
        x1 = U(:,2*i-3); y1 = U(:,2*i-2); x2 = x; y2 = y;
        x1p = (x1+y2); x1p = x1p/norm(x1p);
        y1p = (y1-x2); y1p = y1p/norm(y1p);
        
        p = (conj(x2'*y1)-1)/((x2'*y1)-1);
        x2p = (x2+p*y1); x2p = x2p/norm(x2p);
        y2p = (y2-conj(p)*x1);y2p = y2p/norm(y2p);
        U(:,2*i-3:2*i) = [x1p,y1p,x2p,y2p];
    end
    
    %disp(i)
    
end

%makes the upper corner to be positive


M = kron(eye(N/2), [0,1;1,0]);
U = U*M;

Sigma=U.'*C*U; 
Q = U';
% change convention. C = conj(U)*Sigma*U' = Q.'*Sigma*Q
end


%%
function[Ds,Vs]=sortD(D,V)

    Dv = real(diag(D));
    lD = length(Dv);
    [~,ind] = sort(Dv);
    sortM = zeros(lD,lD);
    
    for i=1:lD
        sortM(i,ind(i)) = 1;
    end
    
    Ds = sortM*D*sortM.';
    Vs = V*sortM.';
    
end
