classdef EntData
 % Entanglement data class
 %-------------------------------------------------
 % TODO: save without making a copy...
 %-------------------------------------------------
 %  EXAMPLES OF USAGES:
 %-------------------------------------------------
 %     - CREATE A NEW ENTDATA
 %          myResult = EntData(ResTmp);
 %     or:
 %          myResult = EntData(0,SA,Nega,ES);
 %
 %     - SAVE THE ENTDATA
 %          myResult.saveEnt;
 %
 %     - DELETE THE ENTDATA
 %          myResult.deleteEnt;
 %
 %     - UPDATE SOME VARIABLES
 %          myResult.updataVar(SA,Nega,RE,MI);
 %------------------------------------------------
 
    properties
        
        model = '';
        runName;
        isCombine;
        NSite;
        Epsilon;
        SA;
        RE;
        MI;
        Nega;
        purpose;
        result;
        matName;
        
        % These are for the level spacing
        ES = 0;
        NS = 0;
        RhoTA = 0;
        XD = 0; % XD is redundant. Just for history reason.
        XV = 0;
        RenyiEnt = 0;
        renyi = 0;
        
        % These are for lattice calculation
        phix;
        phiy;
        
        myName_;
        
    end
        
    methods
        
               
        function obj = EntData(inField, varargin)
        % Constructor. 
        % - If varargin is emty, the obj is constructed from inField.
        % - If inField == 0, construct from varagin
        %---------------------------
        % EXAMPLE:
        %   SA = 1;
        %   Nega = 1;
        %   Res = EntData(0,SA,Nega);
        %---------------------------
        
            if inField == 0
                
             % the first one is the inField
                for ii = 1:nargin-1
                    varname = inputname(ii+1);
                    eval(['obj.',varname,' = varargin{ii};']);

                end
                
            else
                
                % just for historical purpose. do not use this for future
                % data
                
                obj.runName = inField.runName;
                obj.isCombine = inField.isCombine;
                obj.NSite = inField.NSite;
                obj.Epsilon = inField.Epsilon;
                obj.SA = inField.SA;
                obj.Nega = inField.Nega;

                for ii = 1:nargin
                    varname = inputname(ii);
                    eval(['Res.',varname,' = varargin{ii};']);

                end

                if isfield(inField,'purpose')
                   obj.purpose = inField.purpose;              
                end

                if isfield(inField,'model')
                   obj.model = inField.model;
                end

                if isfield(inField,'ES')
                   obj.ES = inField.ES; 
                end

                if isfield(inField,'NS')
                   obj.NS = inField.NS; 
                end

                if isfield(inField,'RhoTA')
                   obj.RhoTA = inField.RhoTA; 
                end

                if isfield(inField,'XV')
                    obj.XV = inField.XV;
                end

                if isfield(inField,'XD')
                    obj.XD = inField.XD;
                end

            end
          
                       
            obj.matName = ['Run-', obj.runName,'-model-',num2str(obj.model),'.mat'];
        
            disp(' --- Initialize an object...');
        end
        
        
       
        function saveEnt(inObj)
            % Save results
            
            % inputname(1) gets the name of the input object
            inObj.myName_ = inputname(1);
            eval([inObj.myName_ ' = inObj;']);
                        
            if ~isfile(inObj.matName)       
                save(inObj.matName, inObj.myName_ );
                disp(' --- Create new file...');
        
            else
                load(inObj.matName);
    
                if ~isempty(who(matfile(inObj.matName), inObj.myName_ ))
                    error(' --- !! Please rename...'); 
                else
                save(inObj.matName, inObj.myName_ , '-append');
                disp(' --- Append new data to existing file...');
                end 
            end
            
        end
        
        function deleteEnt(inObj)
            
            inObj.myName_ = inputname(1);
            matName_ = inObj.matName;
            load(matName_);
            clear(inObj.myName_);
            clear('inObj');
            save(matName_);
                       
        end
        
        function inObj = appendEnt(inObj, addField)
            % Append new results
            % Example: Result = Result.appendEnt(ResTmp);
            
            inObj.myName_ = inputname(1);
            
            inObj.NSite = [inObj.NSite, addField.NSite];
            inObj.SA = [inObj.SA; addField.SA];
            inObj.Nega = [inObj.Nega; addField.Nega];

            eval([inObj.myName_ ' = inObj;']);
            save(inObj.matName, inObj.myName_ , '-append');
            disp(' --- Append done.')
                          
        end
        
        function inObj = updateVar(inObj, varargin)
           % Update some variable.
           % Example: RE_MI.updateVar(SA,Epsilon,Nega,RE,MI); 
           
            for ii = 1:nargin-1
                
                % The first input variable is "RE_MI" 
                % This is implicit. In the example above,
                % there are 6 input variables in total.
                varname = inputname(ii+1);
                eval(['inObj.',varname,' = varargin{ii};']);
                

            end
            
            
            % Show the new variables and confirm
            for ii = 1:nargin-1
                varname = inputname(ii+1);
                fprintf([' --- New var: ', varname]);
                disp(varargin{ii});
                
            end
            isSave = input(' --- Ready to save? [y/n] ', 's');
            
            if isSave == 'y'
                eval([inObj.myName_ ' = inObj;']);
                save(inObj.matName, inObj.myName_ , '-append');
            end
            
            disp(' --- Update done.')
            
           
        end
                
    end
        
end

function deleteVar(var)
% TODO...
end