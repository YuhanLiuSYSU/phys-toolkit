%%
Data = Res;

save('default.mat','-struct','Data')
load('default.mat')

Expr = 2;

%% Plot entanglement entropy 

if Expr == 1

    figure
    plot(varAll,real(xi_AB/log(2)),'r')
    hold on
    plot(varAll,S_AB/log(2))

    dim = [.55 .6 .3 .3];
    annotation('textbox',dim,'String',...
        sprintf(['L = ', num2str(L),', LA = LB = ',num2str(LA)]),...
        'FitBoxToText','on');

    plotInfo = struct('xLabel','$u$',...
                      'yLabel','$S,\mathcal{E}/log(2)$',...
                      'title','Entanglement in SSH model'...
                      );

    plotStyle(plotInfo)

elseif Expr == 2
    
    isFitvonNeumann = 1;
    isFitNegativity = 1;
    isFitReflectedEnt = 0;
    isFitRenyiEntropy = 1;
    isFitRefRenyiEntropy = 0;

    if isFitvonNeumann == 1
        % Fit Entanglement entropy

        % 2*x because LTotal = 2*LA
        % From theory, c = 1
        fitfunS = fittype(@(c,c1,x) c/3*log(sin((2*x)*pi/L))+c1 );
        x0 = [1.,1.];
        fitCoeffiS = fit_(varAll',real(S_AB)',fitfunS,x0);

        plotInfo = struct('xLabel','$L_{A}$','yLabel','$S$',...                  
                      'title','Entanglement at critical point');

        plotStyle(plotInfo)
    end
    
    if isFitNegativity == 1
        % Fit negativity
        figure
        fitfunE = fittype(@(c,c1,x) c/4*...
            log(sin((x)*pi/L).*sin((x)*pi/L)./sin((2*x)*pi/L))+c1);
        x = varAll;
        y = real(xi_AB);
        fitCoeffiE = fit_(x',y',fitfunE,x0);

        plotInfo = struct('xLabel','$L_{A}$','yLabel','$\mathcal{E}$',...
                          'title','Negativity at critical point');                  
        plotStyle(plotInfo)
    end
    
    if isFitReflectedEnt == 1
        % Fit reflected entropy
        figure
        fitfunR = fittype(@(c,c1,x) c/3*...
            log(sin((x)*pi/L).*sin((x)*pi/L)./sin((2*x)*pi/L))+c1/log(2));
        
        fitfunR = fittype(@(c,c1,x) c/3*...
            log(sin((x)*pi/L).*sin((x)*pi/L)./(sin((2*x+dis)*pi/L)))+c1/log(2));
        x0 = [1.,1.];
        fitCoeffiR = fit_(varAll',RE',fitfunR,x0);

        plotInfo = struct('xLabel','$L_A$','yLabel','$R$',...  
                          'title','SSH-Reflected entropy');
        plotStyle(plotInfo)
        plot(varAll, real(MI), '.-');
        plot(varAll, real(RE-MI), '.-'); % c/3*log(2)
    end
    
    if isFitRenyiEntropy == 1
        % Fit Renyi entropy. 2*x because LTotal = 2*LA
        % From theory, c = 1
        
        figure
        
        fitfunRenyi = fittype(@(c,c1,x) c/6*(1+1/renyi)...
            *log(sin((2*x)*pi/L))+c1/log(2));
        x0 = [1.,1.];
        fitCoeffiRenyi = fit_(varAll',real(Renyi)',fitfunRenyi,x0);

        plotInfo = struct('xLabel','$L_{A}$','yLabel','$S_{Renyi}$',...                  
                      'title','Entanglement at critical point');

        plotStyle(plotInfo)
    
    end
    
    if isFitRefRenyiEntropy == 1
        % Fit Renyi entropy. 2*x because LTotal = 2*LA
        % From theory, c = 1
        
        figure
        
        fitfunRefRenyi = fittype(@(c,c1,x) c/6*(1+1/renyi)...
            *log(sin((x)*pi/L).*sin((x)*pi/L)./sin((2*x)*pi/L))+c1/log(2));
        x0 = [1.,1.];
        fitCoeffiRenyi = fit_(varAll',real(RenyiRefEnt)',...
                              fitfunRefRenyi,x0);

        plotInfo = struct('xLabel','$L_{A}$','yLabel','$R_{Renyi}$',...                  
                      'title','Entanglement at critical point');

        plotStyle(plotInfo)
    
    end
    
end


%% Plot entanglement spectrum 

figure
for ii=1:varPt
    
    if Expr == 1    
        ES = ES_AB{ii};
        NS = xiS_AB{ii};
        scatter(varAll(ii)*ones(4*LTotal,1), NS ,10,'b','filled')
    
    else
        ES = sort(real(log(2./(ES_AB{ii}+1)-1)));
        scatter(varAll(ii)*ones(8*varAll(ii),1), ES ,10,'b','filled')
        
    end
    hold on

end

if Expr == 1
    plotInfo = struct('xLabel','$u$','yLabel','NS',...                 
                  'title','Negativity spectrum');
else
    plotInfo = struct('xLabel','$L_A$','yLabel','ES',...  
                  'range',[min(varAll), max(varAll), -15, 15],...
                  'title','Entanglement spectrum');
    
end

plotStyle(plotInfo)

%% Plot RenyiRefEnt

f = figure;
f.Position = [200 50 650 610];
labelTotal = ["$\mathrm{(a)}$","$\mathrm{(b)}$",...
              "$\mathrm{(c)}$","$\mathrm{(d)}$"];

factor = (renyi+1)/(6*renyi);

xData = varAll;
yData = {3*real(RE-MI)/log(2),...
        real(RenyiRefEnt - MIRenyi)/(log(2)*factor),...
        real(xi_AB - RenyiRefEnt/2)/log(2),...
        real(xi_AB - MIRenyi/2)/log(2)};
    
titleTotal = ["$3(R_{A:B}-I_{A:B})/\log(2)$",...
              "$\frac{6n}{n+1}(R^{(n)}-I^{(n)})/\log(2)$",...
              "$(\mathcal{E}-R_{A:B}^{(1/2)}/2)/\log(2)$",...
              "$(\mathcal{E}-\frac{3}{4}I_{A:B})/\log(2)$"];

for i = 1:4
    
    subplot(2,2,i)
    plot(xData, yData{i},'.-')
    plotInfo = struct('xLabel','$L_A$','yLabel','',...
                      'title',titleTotal(1),...
                      'addText',labelTotal(1),...
                      'labelPosition',1.1...
                      );
    plotStyle(plotInfo)   
    
end

print_pdf(f,'reflectedEntropy-harmonic-chain')

%% R^{(n)}-I^{(n)}

% Data = {t2_result1,t2_result2,t2_result3,t2_result4};
% Data = {t3_result1,t3_result2,t3_result3,t3_result4};
Data = {cross_ratio1,cross_ratio2,cross_ratio3,cross_ratio4};
% Data = {cross_ratio_renyi1,cross_ratio_renyi2,...
%         cross_ratio_renyi3,cross_ratio_renyi4};

myLegend = cell(length(Data),1);

for i = 1:length(Data)    
    Res = Data{i};
    
%     xData = Res.eta;
    LA = Res.LAB/2; 
%     LA = Res.LA;
    xData = LA * LA ./((Res.varAll+LA).*(Res.varAll+LA));
    
%     yData = 3*real(Res.RE-Res.MI)/log(2);
%     yLabel = '$3(R-I)/\ln 2$';
%     isPositive = 1;
    
%     factor = (Res.renyi+1)/(6*Res.renyi);
%     yData = abs(Res.RenyiRefEnt-Res.MIRenyi)/(log(2)*factor);
%     yLabel = '$\frac{6n}{n+1}(R^{(n)}-I^{(n)})/\ln 2,\quad n=2$';
%     isPositive = 1;

    yData = real(Res.xi_AB - 1/2*Res.RenyiRefEnt)/log(2);
    yData = real(Res.xi_AB./Res.RenyiRefEnt);
%     yData = real(Res.xi_AB);
%     yData = real(Res.RenyiRefEnt);
    yLabel = '$(\mathcal{E}-0.5 R^{(1/2)})/\ln 2$';
    isPositive = 0;

    %----------------------------------------------------------%
    
    plot(xData, yData, '.-')
    if isPositive == 1
        plotInfo = struct('xLabel','$x$','yLabel','',...  
                          'range',[0.5,max(xData),0,max(yData)*1.1],...
                          'title',yLabel);
    else
        plotInfo = struct('xLabel','$x$','yLabel','',...  
                          'range',[0.5,max(xData),min(yData)*1.1,0],...
                          'title',yLabel);
    end
    
%     plotStyle(plotInfo)
    hold on
    myLegend{i} = ['$L=',num2str(Res.L),', L_{AB} = ',num2str(Res.LAB),'$'];
%     myLegend{i} = ['$L=',num2str(Res.L),', L_{A} = ',num2str(Res.LA),'$'];

end

legend(myLegend,'interpreter','latex','FontSize',10)

    