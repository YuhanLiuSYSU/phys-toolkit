%First Chern Number of one dimensional potoniccrystal
clear
tic;

Nx=1;%size of the lattice
Nb=50;%mesh 
ep=(2*pi)/(Nx*Nb);%the absolute value of displacement
Cherntemp=zeros(Nb*Nb*Nx,1);

num=6;


for nnn=1:Nb*Nb*Nx
    
    
    p3=floor(nnn/Nb)+1;
    p1=mod(nnn-1,Nb)+1;
    kxl=Nx*(p1-1)*2*pi/(Nx*Nb);
    kyl=(p3-1)*2*pi/(Nx*Nb);
    
    mye=[ep 0;0 ep;ep ep];
      
    H1=BHZ(kxl,kyl);
    H2=BHZ(kxl+mye(1,1),kyl+mye(1,2));
    H3=BHZ(kxl+mye(2,1),kyl+mye(2,2));
    H4=BHZ(kxl+mye(3,1),kyl+mye(3,2));        


    [v1,d1]=eig(H1);
    [v2,d2]=eig(H2);
    [v3,d3]=eig(H3);
    [v4,d4]=eig(H4);


    state1= v1(:,num);
    state2= v2(:,num);
    state3= v3(:,num);
    state4= v4(:,num);

    U1=dot(state1,state2)/norm(dot(state1,state2));
    U2=dot(state2,state4)/norm(dot(state2,state4));
    U3=dot(state3,state4)/norm(dot(state3,state4));
    U4=dot(state1,state3)/norm(dot(state1,state3));

        
    Cherntemp(nnn)=log(U1*U2/(U3*U4))/(2*pi*1i);%Chern Curvature
                %Chern1=Chern1+Cherntemp(p1,p3);        

end

sum(sum(Cherntemp))%First Chern Number

toc;


X(Nb)=0;Y(Nb)=0;Z(Nb,Nb)=0;
for nnn=1:Nb*Nb*Nx
    p2=floor((nnn-1)/Nb)+1;
    p1=mod(nnn-1,Nb)+1;
    k1=Nx*(p1-1)*2*pi/(Nx*Nb);
    k2=(p2-1)*2*pi/(Nx*Nb);
    X(p1)=k1;Y(p2)=k2;
    Z(p1,p2)=real(Cherntemp(p1+(p2-1)*Nb))/(ep^2);
end


gcf=figure('visible','off');
surf(X,Y,Z)
print_pdf(gcf,'chern')