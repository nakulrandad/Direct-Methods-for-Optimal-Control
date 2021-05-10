%--------------------------------------------------------------------------
% double_integrator_single_shooting.m
% This file solves the double integrator bang-bang problem using a
% single-step method using fmincon
% (employs the trapezodial rule with composite trapezoidal quadrature)
%--------------------------------------------------------------------------
% Primary Contributor: Nakul Randad, Indian Institute of Technology Bombay
%--------------------------------------------------------------------------

soln = method();

% actual implementation
function soln = method
    % problem parameters
    p.ns = 2; p.nu = 1; % number of states and controls
    p.t0 = 0; % time starts at '0'
    p.y10 = 10; p.y1f = 0; p.y20 = 0; p.y2f = 0; % boundary conditions
    p.umax = 1; % maximum absolute force
    
    % direct transcription parameters
%     p.nt = 200; % number of node points
    p.nt = 100; % number of node points
    x0 = zeros(p.nt*(p.ns+p.nu)+1,1); % initial guess (all zeros)
    p.tfi = p.nt*(p.ns+p.nu)+1;
    x0(p.tfi) = 1; % final time guess
    p.tf = x0(p.tfi);
    p.t = linspace(p.t0,p.tf,p.nt)'; % time horizon
    p.h = diff(p.t); % step size
    
    % discretized variable indices in x = [y1,y2,u];
    p.y1i = 1:p.nt; p.y2i = p.nt+1:2*p.nt; p.ui = 2*p.nt+1:3*p.nt;
    
    options = optimoptions(@fmincon,'display','iter','MaxFunEvals',1e5...
                           ,'MaxIterations',1e2,'ConstraintTolerance',1.0000e-08); % options
                       
    % solve the problem
    x = fmincon(@(x) objective(x,p),x0,[],[],[],[],[],[],@(x) constraints(x,p),options);
    
    % obtain the optimal solution
    y1 = x(p.y1i); y2 = x(p.y2i); u = x(p.ui); p.tf = x(p.tfi); % extract
    p.t = linspace(p.t0,p.tf,p.nt)';
    soln.y1 = y1; soln.y2 = y2; soln.u = u; soln.tf = p.tf; soln.p = p;
    
    % plot
    showPlot(y1,y2,u,p)
end

% objective function
function f = objective(x,p)
    f = x(p.tfi) - p.t0;  % calculate objective i.e. minimise time
end

% constraint function
function [c,ceq] = constraints(x,p)
    y1 = x(p.y1i); y2 = x(p.y2i); u = x(p.ui); % extract
    Y = [y1,y2]; F = [y2,u]; % create matrices (p.nt x p.ns)
    ceq1 = y1(1) - p.y10; % initial state conditions
    ceq2 = y2(1) - p.y20;
    ceq3 = y1(end) - p.y1f; % final state conditions
    ceq4 = y2(end) - p.y2f;
    
    % update step size
    p.tf = x(p.tfi);
    p.t = linspace(p.t0,p.tf,p.nt)'; % time horizon
    p.h = diff(p.t); % step size
    
    % integrate using trapezoidal quadrature
    ceq5 = Y(2:p.nt,1) - Y(1:p.nt-1,1) - p.h/2.*( F(1:p.nt-1,1) + F(2:p.nt,1) );
    ceq6 = Y(2:p.nt,2) - Y(1:p.nt-1,2) - p.h/2.*( F(1:p.nt-1,2) + F(2:p.nt,2) );
    c1 = x(p.tfi)-p.t0;
    c2 = u-p.umax;
    c3 = -u-p.umax;
    
    c = [c1;c2;c3]; ceq = [ceq1;ceq2;ceq3;ceq4;ceq5;ceq6]; % combine constraints
end

% plotting function
function showPlot(y1,y2,u,p)
    close all
    
    t = tiledlayout(2,1);
    t.Padding = 'compact';
    t.TileSpacing = 'compact';
%     title(t, ['Optimal time taken is ', num2str(p.t(end)), ' secs.']);

    nexttile
    hold on
    grid on
    plot(p.t,y1,'b','LineWidth',1);
    plot(p.t,y2,'Color',[0, 0.5, 0],'LineWidth',1);
    legend('Pos','Vel');
    xlabel("Time (in sec)");
    xlim([0,p.t(end)]);
    hold off

    nexttile
    grid on
    stairs(p.t,u,'r--','LineWidth',1);
    legend('u');
    ylabel("Acceleration");
    xlabel("Time (in sec)");
    xlim([0,p.t(end)]);
    ylim([-p.umax,p.umax]*1.2)
    
    % To print the figure
%     print('./results/optimal_sol_single_shooting','-dpng')
    
end