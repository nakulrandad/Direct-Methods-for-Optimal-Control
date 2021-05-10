%--------------------------------------------------------------------------
% double_integrator_collocation.m
% This file solves the double integrator bang-bang problem by
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
x1 = SX.sym('x1'); % position
x2 = SX.sym('x2'); % velocity
x = [x1; x2];
u = SX.sym('u'); % control

%% Dynamics and Objective
% Model equations
xdot = [x2; u];

% Continuous time dynamics
f = Function('f', {x, u}, {xdot});

%% Discretization
N = 100; % number of control intervals

%% Set up solver
% Start with an empty NLP
opti = Opti();

T = opti.variable();        % Time horizon
opti.set_initial(T, 1);     % Inital guess
opti.subject_to(T>0);       % Time must be positive
h = T/N;

% Initial conditions
Xk = opti.variable(2);
opti.subject_to(Xk==[10; 0]);
opti.set_initial(Xk, [10; 0]);

% Collect all states/controls
Xs = {Xk};
Us = {};
J = 0;

% Formulate the NLP
for k=1:N
   % New NLP variable for the control
   Uk = opti.variable();
   Us{end+1} = Uk;
   u_lim = 1;
   opti.subject_to(-u_lim<=Uk);
   opti.subject_to(Uk<=u_lim);
   opti.set_initial(Uk, 0);

   % Decision variables for helper states at each collocation point
   Xc = opti.variable(2, d);
   opti.set_initial(Xc, repmat([0;0],1,d));

   % Evaluate ODE right-hand-side at all helper states
   [ode] = f(Xc, Uk);
   
   % Add contribution to quadrature function
   int_L = ones(1,3)*B*h;
   J = J + int_L;

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
   opti.set_initial(Xk, [0;0]);

   % Continuity constraints
   opti.subject_to(Xk_end==Xk)
end

% Boundary conditions at finish line
opti.subject_to(Xs{end}(1)==0);
opti.subject_to(Xs{end}(2)==0);

%% Optimisation solver
Xs = [Xs{:}];
Us = [Us{:}];

opti.minimize(T);

opti.solver('ipopt');

tic
sol = opti.solve();
toc

x_opt = sol.value(Xs);
u_opt = sol.value(Us);

%% Post-processing
time = linspace(0, sol.value(T), N+1)';

t = tiledlayout(2,1);
t.Padding = 'compact';
t.TileSpacing = 'compact';
title(t, ['Optimal time taken is ', num2str(time(end)), ' secs.']);

nexttile
hold on
grid on
plot(time,x_opt(1,:),'b','LineWidth',1);
plot(time,x_opt(2,:),'Color',[0, 0.5, 0],'LineWidth',1);
legend('Pos','Vel');
xlabel("Time (in sec)");
xlim([0,time(end)]);
hold off

nexttile
grid on
plot(time, [u_opt nan],'r--','LineWidth',1);
legend('u');
ylabel("Acceleration");
xlabel("Time (in sec)");
xlim([0,time(end)]);
ylim([-u_lim,u_lim]*1.2)

% To print the figure
% print(".\results\optimal_sol_collocation",'-dpng')