%--------------------------------------------------------------------------
% Bryson_Denham_single_shooting.m
% This file solves the Bryson-Denham problem by single-shooting method
% using fmincon
% (employs the trapezodial rule with composite trapezoidal quadrature)
%--------------------------------------------------------------------------
% Primary Contributor: Nakul Randad, Indian Institute of Technology Bombay
%--------------------------------------------------------------------------

soln = method();

% actual implementation
function soln = method
    % problem parameters
    p.ns = 2; p.nu = 1; % number of states and controls
    p.t0 = 0; p.tf = 1; % time horizon
    p.y10 = 0; p.y1f = 0; p.y20 = 1; p.y2f = -1; % boundary conditions
    p.l = 1/9;
    
    % direct transcription parameters
%     p.nt = 20; % number of node points
    p.nt = 100; % number of node points
    p.t = linspace(p.t0,p.tf,p.nt)'; % time horizon
    p.h = diff(p.t); % step size
    
    % discretized variable indices in x = [y1,y2,u];
    p.y1i = 1:p.nt; p.y2i = p.nt+1:2*p.nt; p.ui = 2*p.nt+1:3*p.nt;
    x0 = zeros(p.nt*(p.ns+p.nu),1); % initial guess (all zeros)
    options = optimoptions(@fmincon,'display','iter','MaxFunEvals',1e5); % options
    
    % solve the problem
    tic
    x = fmincon(@(x) objective(x,p),x0,[],[],[],[],[],[],@(x) constraints(x,p),options);
    toc
    
    % obtain the optimal solution
    y1 = x(p.y1i); y2 = x(p.y2i); u = x(p.ui); % extract
    soln.y1 = y1; soln.y2 = y2; soln.u = u; soln.p = p;
    
    % plots
    showPlot(y1,y2,u,p)
end

% objective function
function f = objective(x,p)
    u = x(p.ui); % extract
    L = u.^2/2; % integrand
    f = trapz(p.t,L); % calculate objective
end

% constraint function
function [c,ceq] = constraints(x,p)
    y1 = x(p.y1i); y2 = x(p.y2i); u = x(p.ui); % extract
    Y = [y1,y2]; F = [y2,u]; % create matrices (p.nt x p.ns)
    ceq1 = y1(1) - p.y10; % initial state conditions
    ceq2 = y2(1) - p.y20;
    ceq3 = y1(end) - p.y1f; % final state conditions
    ceq4 = y2(end) - p.y2f;
    
    % integrate using trapezoidal quadrature
    ceq5 = Y(2:p.nt,1) - Y(1:p.nt-1,1) - p.h/2.*( F(1:p.nt-1,1) + F(2:p.nt,1) );
    ceq6 = Y(2:p.nt,2) - Y(1:p.nt-1,2) - p.h/2.*( F(1:p.nt-1,2) + F(2:p.nt,2) );
    c1 = y1 - p.l; % path constraints
    
    c = c1; ceq = [ceq1;ceq2;ceq3;ceq4;ceq5;ceq6]; % combine constraints
end

% plotting function
function showPlot(y1,y2,u,p)
    close all
    t = tiledlayout(2,2);
    t.Padding = 'compact';
    t.TileSpacing = 'compact';
    
    nexttile
    hold on
    plot(p.t,y1,'b','LineWidth',1);
    yline(p.l,'k--','LineWidth',1);
    scatter(p.t0,p.y10,[],'g','filled');
    scatter(p.tf,p.y1f,[],'r','filled');
    hold off
    ylim([-inf, p.l*1.05])
    ylabel('Position');
    xlabel('Time [s]');
    legend('$y1$','$y1 < \frac{1}{9}$',"$y1_0$","$y1_f$",'Interpreter','latex','Location', 'South');    
    
    nexttile
    hold on
    plot(p.t,y2,'Color',[0, 0.5, 0],'LineWidth',1);
    scatter(p.t0,p.y20,[],'g','filled');
    scatter(p.tf,p.y2f,[],'r','filled');
    hold off
    ylabel('Speed');
    xlabel('Time [s]');
    legend('$y2$',"$y2_0$","$y2_f$",'Interpreter','latex','Location', 'South')

    nexttile(3,[1,2])
    stairs(p.t,u,'r','LineWidth',1);
    xlabel('Time [s]');
    ylabel('Thrust');
    legend('u','Location', 'South');

    % To print the figure
%     print('./results/optimal_sol_single_shooting','-dpng')
end