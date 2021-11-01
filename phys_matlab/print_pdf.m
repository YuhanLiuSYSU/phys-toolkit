function [] = print_pdf(f,title)



if ~exist('title','var')||isempty(title)
  title = 'combine';
end

if isfile(fullfile(cd,[title,'.pdf']))
    
    isRewrite = input(' --- Rewrite the existing file? [y/n] ', 's');
            
    if isRewrite == 'y'
        set(f,'Units','Inches');
        pos = get(f,'Position');
        set(f,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',...
            [pos(3), pos(4)])
        print(f,title,'-dpdf','-r600')
    else    
        error_msg = "please save with a different name...";
        disp(error_msg)
    
    end
    
else
    set(f,'Units','Inches');
    pos = get(f,'Position');
    set(f,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',...
        [pos(3), pos(4)])
    print(f,title,'-dpdf','-r600')
    
end



end