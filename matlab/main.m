%% Clear
clc;
clear all;
close all;
%% Inverted Pendulum System ������ģ��
%  x' = Ax+Bu
%  y  = Cx+Du
params1; % ϵͳ��������һ���������ļ�'params1.m'��
A = [0  1                    0                    0
     0  -(Il+ml*ll^2)*bl/pl  (ml^2*gl*ll^2)/pl    0
     0  0                    0                    1
     0  -(ml*ll*bl)/pl       ml*gl*ll*(Ml+ml)/pl  0];
B = [0
     (Il+ml*ll^2)/pl
     0
     ml*ll/pl];
C = [1 0 0 0
     0 1 0 0];
x0 = [0.98;0;0.2;0];  % Initial state

n1 = size(A,1);
n2 = size(B,2);

% Solve the Riccati equation��to get controller gain 'K'
% J = x'Qx + u'Ru
% or use function: 'place' to obtain 'K'
Q = C'*C;
R = eye(n2);
K = -lqr(A,B,Q,R)

% Observability
sys_ss = ss(A,B,C,0);
ob = obsv(sys_ss);
observability = rank(ob)

% Observer gain 'L'
p = [-13 -12 -11 -10];
L = place(A',C',p)'
x = x0;
xhat = 0*x0;
h = 0.01; ts = 1000;
%% Run
for k = 1:ts
    u(:,k) = K*xhat(:,k);
    x(:,k+1) = RK(x(:,k),u(:,k),zeros(n1,1),A,B,h);  % ���������������ϵͳ��ɢ����"RK"��һ���������ļ�����
    y(:,k) = C*x(:,k);  % ��ɢϵͳ�ͺܼ��ˣ�ֱ�ӵ����Ϳ�����
	yhat(:,k) = C*xhat(:,k);
	xhat(:,k+1) = RK(xhat(:,k),u(:,k),L*(y(:,k)-yhat(:,k)),A,B,h);  % ע��L����˸�h����������ϵͳ��ɢ��
end
%% Figures
t = [1:ts];
t1 = t*h;
for i = 1:n1
    figure(i)  % plot all the states 'x_i'
    plot(t1,x(i,t),t1,xhat(i,t),'LineWidth',2);hold on;grid;
    xlabel('Time');eval(['ylabel(''State x_' num2str(i) '(t)'');']);
end

figure(i+1)  % plot the output 'y'
plot(t1,y(:,t),'LineWidth',2);grid;
xlabel('Time');ylabel('y(t)');