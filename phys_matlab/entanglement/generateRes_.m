function [Res] = generateRes_(type,varPt,varargin)
%generateRES_ Generate the entanglement quantities according to input type
%   

if nargin == 3
    renyi = varargin{1};
end

%%

Res.S_AB = zeros(1, varPt);

if exist('renyi','var')
    Res.Renyi = zeros(1,varPt); 
    Res.renyi = renyi;
end

if any(type == 2)
   Res.xi_AB = zeros(1, varPt); 
end

if any(type == 3)
    Res.RE = zeros(1,varPt);   
    if exist('renyi','var')
        Res.RenyiRefEnt = zeros(1,varPt); 
    end
    
end
    
if any(type ==4)
    Res.MI = zeros(1,varPt);
    if exist('renyi','var')
        Res.MIRenyi = zeros(1,varPt);
    end
end

if any(type == 5)
   Res.SA = zeros(1, varPt);
   if exist('renyi','var')
       Res.RenyiA = zeros(1,varPt); 
   end   
   
end


end

