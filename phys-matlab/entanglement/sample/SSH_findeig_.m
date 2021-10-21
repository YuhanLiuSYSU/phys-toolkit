function [ak1,bk1] = SSH_findeig_(t1,t2,k)

H=[0,t1+t2*exp(1i*k);t1+t2*exp(-1i*k),0];
[V,D] = eig(H);
[D,myI] = sort(diag(D));
V = V(:, myI);
ak1=V(1,1);
bk1=V(2,1);

end

