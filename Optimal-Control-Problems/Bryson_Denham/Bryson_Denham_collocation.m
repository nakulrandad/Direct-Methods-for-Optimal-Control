%--------------------------------------------------------------------------
% Bryson_Denham_collocation.m
% This file solves the Bryson-Denham problem by
% direct collocation method using Casadi
%--------------------------------------------------------------------------
% Primary Contributor: Nakul Randad, Indian Institute of Technology Bombay
%--------------------------------------------------------------------------
% CasADi v3.5.5
import casadi.*

close all
% clc

%% Set up collocation points 
% Degree of interpolating polynomial
d = 3;

% Get collocation points
tau = collocation_points(d, 'legendre');

% Collocation linear maps
[C,D,B] = collocation_coeff(tau);

%% Declare decision variables
T = 1;             % Time horizon
limit = 1/9;       % position limit

% Declare model variables
x1 = SX.sym('x1'); % position
x2 = SX.sym('x2'); % velocity
x = [x1; x2];
u = SX.sym('u');

%% Dynamics and Objective
% Model equations
xdot = [x2; u];

% Objective term
L = u^2/2;

% Continuous time dynamics
f = Function('f', {x, u}, {xdot, L});

%% Control discretization
N = 100; % number of control intervals
h = T/N;

%% Set up solver
% Start with an empty NLP

opti = Opti();
J = 0;

% Initial conditions
Xk = opti.variable(2);
opti.subject_to(Xk==[0; 1]);
opti.set_initial(Xk, [0; 1]);

% Collect all states/controls
Xs = {Xk};
Us = {};
cost = [];

% Formulate the NLP
for k=0:N-1
   % New NLP variable for the control
   Uk = opti.variable();
   Us{end+1} = Uk;
   opti.set_initial(Uk, 0);

   % Decision variables for helper states at each collocation point
   Xc = opti.variable(2, d);
   opti.subject_to(Xc(1,:) <= limit); % inequality constraint on state
   opti.set_initial(Xc, repmat([0;0],1,d));

   % Evaluate ODE right-hand-side at all helper states
   [ode, quad] = f(Xc, Uk);

   % Add contribution to quadrature function
   int_L = quad*B*h;
   J = J + int_L;
   cost = [cost; J];

   % Get interpolating points of collocation polynomial
   Z = [Xk Xc];

   % Get slope of interpolating polynomial (normalized)
   Pidot = Z*C;
   % Match with ODE right-hand-side 
   opti.subject_to(Pidot == h*ode);

   % State at end of collocation interval
   Xk_end = Z*D;

   % New decision variable for state at end of interval
   Xk = opti.variable(2);
   Xs{end+1} = Xk;
   opti.subject_to(Xk(1) <= limit);% inequality constraint on state
   opti.set_initial(Xk, [0;0]);

   % Continuity constraints
   opti.subject_to(Xk_end==Xk)
end

% Boundary conditions
opti.subject_to(Xs{end}(1)==0);
opti.subject_to(Xs{end}(2)==-1);

%% Optimisation solver

Xs = [Xs{:}];
Us = [Us{:}];

opti.minimize(J); % minimise the objective function

opti.solver('ipopt'); % backend NLP solver

tic
sol = opti.solve(); % Solve actual problem
toc

x_opt = sol.value(Xs);
u_opt = sol.value(Us);
cost_opt = sol.value(cost);

%% Post-processing
time = linspace(0, T, N+1);
t = tiledlayout(2,2);
t.Padding = 'compact';
t.TileSpacing = 'compact';

nexttile
hold on
plot(time,x_opt(1,:),'b','LineWidth',1);
yline(limit,'k--','LineWidth',1);
hold off
ylim([-inf, limit*1.05])
ylabel('Position');
xlabel('Time [s]');
legend('x','$x < \frac{1}{9}$','Interpreter','latex','Location', 'South');

nexttile
plot(time, x_opt(2,:),'Color',[0, 0.5, 0],'LineWidth',1);
ylabel('Speed');
xlabel('Time [s]');
legend('v','Location', 'South')

nexttile
plot(time, [u_opt nan],'r','LineWidth',1);
xlabel('Time [s]');
ylabel('Thrust');
legend('u','Location', 'South');

nexttile
plot(time,[cost_opt; nan],'--','LineWidth',1);
xlabel('Time [s]');
ylabel('Objective');
legend('$\frac{1}{2} \int u^2$','Interpreter','latex','Location', 'South');

% To print the figure
% print('./results/optimal_sol_collocation','-dpng')