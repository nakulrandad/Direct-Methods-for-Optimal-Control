%--------------------------------------------------------------------------
% double_integrator_multiple_shooting.m
% This file solves the double integrator bang-bang problem by
% multiple-shooting method using Casadi
%--------------------------------------------------------------------------
% Primary Contributor: Nakul Randad, Indian Institute of Technology Bombay
%--------------------------------------------------------------------------
% CasADi v3.5.5
import casadi.*

close all
% clc

%% Setting up the problem
N = 100; % number of control intervals
opti = casadi.Opti(); % Optimization problem

%% Declare decision variables
X = opti.variable(2,N+1); % state trajectory
pos = X(1,:);
speed = X(2,:);
U = opti.variable(1,N);   % control trajectory (throttle)
T = opti.variable();        % final time

%% Set up the objective
opti.minimize(T); % minimize time

%% System dynamics
f = @(x,u) [x(2);u]; % dx/dt = f(x,u)

%% Numerical integration and constraint to make zero gap
dt = T/N; % length of a control interval
for k=1:N % loop over control intervals
   % Runge-Kutta 4 integration
   k1 = f(X(:,k),         U(:,k));
   k2 = f(X(:,k)+dt/2*k1, U(:,k));
   k3 = f(X(:,k)+dt/2*k2, U(:,k));
   k4 = f(X(:,k)+dt*k3,   U(:,k));
   x_next = X(:,k) + dt/6*(k1+2*k2+2*k3+k4);
   opti.subject_to(X(:,k+1)==x_next); % close the gaps
end

%% Path constraints
u_lim = 1;
opti.subject_to(-u_lim<=U);
opti.subject_to(U<=u_lim);
opti.subject_to(T>0); % Time must be positive

%% Boundary conditions
opti.subject_to(pos(1)==10);   % start position
opti.subject_to(speed(1)==0); 
opti.subject_to(pos(N+1)==0); % finish line
opti.subject_to(speed(N+1)==0);

%% Initial guess
opti.set_initial(speed, 0);
opti.set_initial(pos, 0);
opti.set_initial(T, 1);

%% Solver set up
opti.solver('ipopt'); % set numerical backend
tic
sol = opti.solve();   % actual solve
toc

%% Post-processing
time = linspace(0, sol.value(T), N+1)';

t = tiledlayout(2,1);
t.Padding = 'compact';
t.TileSpacing = 'compact';
title(t, ['Optimal time taken is ', num2str(time(end)), ' secs.']);

nexttile
hold on
grid on
plot(time,sol.value(pos),'b','LineWidth',1);
plot(time,sol.value(speed),'Color',[0, 0.5, 0],'LineWidth',1);
legend('Pos','Vel');
xlabel("Time (in sec)");
xlim([0,time(end)]);
hold off

nexttile
grid on
stairs(time(1:end-1),sol.value(U),'r--','LineWidth',1);
legend('u');
ylabel("Acceleration");
xlabel("Time (in sec)");
xlim([0,time(end)]);
ylim([-u_lim,u_lim]*1.2)

% To print the figure
% print('./results/optimal_sol_multiple_shooting','-dpng')