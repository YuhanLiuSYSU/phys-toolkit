function [Res] = updateRes_(Res,Ent,type,pt,varargin)
%updateRes_ Update the entanglement quantities
%   

if nargin == 5
    renyi = varargin{1};
end

%%
Res.S_AB(pt) = Ent.S;

if exist('renyi','var')
    Res.Renyi(pt) = Ent.Renyi; 
end 

if any(type == 2)
    Res.xi_AB(pt) = Ent.xi;
end
    
 
if any(type == 3)
    Res.RE(pt) = Ent.refEnt; 
    if exist('renyi','var')
        Res.RenyiRefEnt(pt) = Ent.RenyiRefEnt; 
    end
end

if any(type == 4) 
    Res.MI(pt) = Ent.MI; 
    if exist('renyi','var')
        Res.MIRenyi(pt) = Ent.MIRenyi;
    end
end


if any(type == 5)
    Res.SA(pt) = Ent.SA;
    Res.RenyiA(pt) = Ent.RenyiA;        
end

end

