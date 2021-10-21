function [chernNb] = Chern_std()
% Compute the Chern number from lattice model using the projector method.

%% Parameter

filledBand = [1];
model = 2;
N = 200;

if model == 1
    % BHZ model.
    % When u = 1.8, C = 1; when u = 2.2, C = 0.
    param.u = 1.8;
    
elseif model == 2
    % Hofstadter model.
    % When [p,q,t] = [1,4,1], the lowest band has C = 1.
    % When [p,q,t] = [1,6,1], the lowest two bands have C = 1.
    
    param.p = 1;
    param.q = 6;
    param.t = 1;
end

%% Compute the Chern number

dk = 2*pi/N;
chernNb = 0;


for i = 1:N
    
    for j = 1:N
        tkx = dk*i;
        tky = dk*j;
        
        P = getProjector(getH(tkx,tky,param,model), filledBand);
        Px = getProjector(getH(tkx+dk,tky,param,model), filledBand);
        Py = getProjector(getH(tkx,tky+dk,param,model), filledBand);
        
        PPx = (Px-P);
        PPy = (Py-P);
        chernNb = chernNb+trace(P*(PPx*PPy-PPy*PPx));
        
    end
end

chernNb = chernNb*(-1i)/(2*pi);

if model == 2
    % The Brullouin zone is q times smaller
    chernNb = chernNb/param.q; 
end

end


%%
function [P] = getProjector(h,filledBand)

    [V,D] = eig(h);
    [~,permutation] = sort(diag(D));
    %D = D(permutation,permutation);
    V = V(:,permutation);
    gr = V(:, filledBand);
    P = gr*(gr)';

end

function [h] = getH(kx,ky,param,model)

    s1 = [0,1;1,0];
    s2 = [0,-1i;1i,0];
    s3 = [1,0;0,-1];
    
    if model == 1
        % BHZ model
        u = param.u;
        
        % give the same result
        %h=(-sin(kx)*s1-sin(ky)*s2)+(u+cos(kx)+cos(ky))*s3; 
        % give the same result
        %h=(sin(kx)*s1+sin(ky)*s2)+(u-cos(kx)-cos(ky))*s3; 
        h = (sin(kx)*s1+sin(ky)*s2)+(u+cos(kx)+cos(ky))*s3;
        
    elseif model == 2
        % Hofstadter model
        % The unitcell is enlarged by q times.
        % B = p/q*2*pi
        
        p = param.p;
        q = param.q;
        t = param.t;
        
        k0 = 2*pi/q;
        vec = 0:1:q-1;
        
        offDiagMtrx = diag(ones(q-p,1),p)+diag(ones(p,1),-q+p);
        h = diag(-2*t*cos(kx+vec*k0))...
            -t*exp(1i*ky)*offDiagMtrx - t*exp(-1i*ky)*offDiagMtrx.';
                
    end

end

