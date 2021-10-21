function [K4m] = BHZ(k1,k2)

    mu=2;J=1.5*mu;b=1.5*mu;a=4*mu;
    omega=mu/0.07;Delta0=mu;
    sigma1=[0 1;1 0];
    sigma2=[0 -1i;1i 0];
    sigma3=[1 0;0 -1];
    i2=[1 0;0 1];
    c1=a*sin(k1);
    c2=a*sin(k2);
    c3=(mu-J)-2*b*(2-cos(k1)-cos(k2))+J*cos(k1)*cos(k2);
    K=c1*sigma1+c2*sigma2+c3*sigma3;

    K4m=zeros(6,6);
    K4m(1:2,1:2)=K+omega*i2;
    K4m(3:4,3:4)=K;
    K4m(5:6,5:6)=K-omega*i2;
    K4m(1:2,3:4)=Delta0*sigma3;
    K4m(3:4,5:6)=Delta0*sigma3;
    K4m(3:4,1:2)=Delta0*sigma3;
    K4m(5:6,3:4)=Delta0*sigma3;

end

