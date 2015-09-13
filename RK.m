function x1 = RK(x,u,other,A,B,h);
K1 = Sys(x,u,other,A,B);
K2 = Sys(x+h*K1/2,u,other,A,B);
K3 = Sys(x+h*K2/2,u,other,A,B);
K4 = Sys(x+h*K3,u,other,A,B);
x1 = x+(K1+2*K2+2*K3+K4)*h/6;
end