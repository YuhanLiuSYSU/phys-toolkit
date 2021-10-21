function [W] = SSH_findv_(k,u,v,w,flag)

if flag==1
    %% Using projector P, which is equivalent to C
%     vk=w*exp(-1i*k)+v;
%     avk=abs(vk);
%     Hk=[1i*u,vk;conj(vk),-1i*u];Ek=sqrt(avk^2-u^2);
%     W=1/2*(eye(2)-Hk/Ek);
    %%
    
    vk=w*exp(-1i*k)+v;
    avk=abs(vk);
    ephi=sqrt((u+avk)/(u-avk));
    ephi2=sqrt(ephi);
    cphi2=(ephi2+1/ephi2)/2;
    sphi2=(ephi2-1/ephi2)/(2*1i);
    
    vR=[-vk/avk*sphi2;cphi2];
    vL=[-vk/avk*conj(sphi2);conj(cphi2)];
    W=vL*vR';
    
elseif flag==2
    %% temporary code for CHL's model
    gamma=u;
    sx=[0,1;1,0];sy=[0,-1i;1i,0];sz=[1,0;0,-1];
    Ek=sqrt((v-w*cos(k))^2+(gamma*sin(k))^2-(v-w)^2);
    Hk=(v-w*cos(k))*sx+gamma*sin(k)*sy+1i*(v-w)*sz;
    W=1/2*(eye(2)-Hk/Ek);
    
end
end

