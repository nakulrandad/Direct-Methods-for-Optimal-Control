%--------------------------------------------------------------------------
% Bryson_Denham_multiple_shooting.m
% This file solves the Bryson-Denham problem by
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
T = 1;        % final time
limit = 1/9;     % position limit
time = linspace(0, T, N+1)'; % time horizon

%% Set up the objective
L = U.^2/2; % integrand
dt = T/N; % length of a control interval
cost = [];
obj = 0;
for i = 1:N
    obj = obj + L(i)*dt;
    cost = [cost;obj];
end
opti.minimize(obj); % minimize objective

%% System dynamics
f = @(x,u) [x(2);u]; % dx/dt = f(x,u)

%% Numerical integration and constraint to make zero gap
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
opti.subject_to(pos<=limit);           % position limit

%% Boundary conditions
opti.subject_to(pos(1)==0);   % start at position 0
opti.subject_to(speed(1)==1); 
opti.subject_to(pos(N+1)==0); % finish line at position 0
opti.subject_to(speed(N+1)==-1);

%% Initial guess
opti.set_initial(speed, 1);
opti.set_initial(pos, 0);

%% Solver set up
opti.solver('ipopt'); % set numerical backend
tic
sol = opti.solve();   % actual solve
toc

%% Post-processing
t = tiledlayout(2,2);
t.Padding = 'compact';
t.TileSpacing = 'compact';

nexttile
hold on
plot(time,sol.value(pos),'b','LineWidth',1);
yline(limit,'k--','LineWidth',1);
hold off
ylim([-inf, limit*1.05])
ylabel('Position');
xlabel('Time [s]');
legend('x','$x < \frac{1}{9}$','Interpreter','latex','Location', 'South');

nexttile
plot(time,sol.value(speed),'Color',[0, 0.5, 0],'LineWidth',1);
ylabel('Speed');
xlabel('Time [s]');
legend('v','Location', 'South')

nexttile
stairs(time(1:end-1),sol.value(U),'r','LineWidth',1);
xlabel('Time [s]');
ylabel('Thrust');
legend('u','Location', 'South');

nexttile
plot(time(1:end-1),sol.value(cost),'--','LineWidth',1);
xlabel('Time [s]');
ylabel('Objective');
legend('$\frac{1}{2} \int u^2$','Interpreter','latex','Location', 'South');

% To print the figure
% print('./results/optimal_sol_multiple_shooting','-dpng')