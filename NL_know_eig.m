% NL_EIG.m - Nonlinear Dynamical System Definition
%
% Defines a two-dimensional nonlinear dynamical system:
% dx/dt = [∇Ψ(x)]^(-1) * diag([-1, 2.5]) * Ψ(x)
%
% where the principal eigenfunctions are:
% ψ₁(x) = x₁² + 2x₂ + x₂³
% ψ₂(x) = x₁ + sin(x₂) + x₁³
%
% and the associated eigenvalues are:
% λ₁ = -1, λ₂ = 2.5

% Main script execution
% Clear workspace and command window
clear; clc;

% Define the nonlinear dynamical system
dynamics = defineNonlinearSystem();

% Visualize the vector field in domain [-2,2] x [-2,2] with step size 0.05
domain = [-2, 2, -2, 2]; % [xmin, xmax, ymin, ymax]
step_size = 0.1;
visualizeVectorField(dynamics, domain, step_size);

% Example trajectory simulation
% x0 = [0.5; 0.5]; % Initial condition
% tspan = [0, 10]; % Time span
%
% % Solve the ODE
% [t, x] = ode45(dynamics, tspan, x0);
%
% % Plot the trajectory on the vector field
% hold on;
% plot(x(:, 1), x(:, 2), 'r-', 'LineWidth', 2);
% plot(x0(1), x0(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
% % legend('Vector field', '', 'Trajectory', 'Initial point', 'Location', 'best');

% Display system information
fprintf('Nonlinear Dynamical System Analysis:\n');
fprintf('Eigenvalues: λ₁ = -1, λ₂ = 2.5\n');
% fprintf('Initial condition: x₀ = [%.1f, %.1f]\n', x0(1), x0(2));
% fprintf('System exhibits saddle-point behavior: convergent in ψ₁ direction, divergent in ψ₂ direction\n');

% Save results to workspace
% t_nl = t;
% x_nl = x;
dynamics_func = dynamics;

% fprintf('\nVariables saved to workspace: t_nl, x_nl, dynamics_func\n');

%% Generate data and apply ResDMD analysis
fprintf('\nGenerating trajectory data for ResDMD analysis...\n');
[X_data, Y_data] = generateTrajectoryData(dynamics);

% visualize the generated data
figure;
plot(X_data(1, :), X_data(2, :), 'r.');
hold on;
plot(Y_data(1, :), Y_data(2, :), 'b.');
axis equal;

fprintf('Applying ResDMD algorithm...\n');
analyzeWithResDMD(X_data, Y_data);

function dynamics = defineNonlinearSystem()
    % DEFINENONLINEARSYSTEM Returns the nonlinear dynamical system as a function handle
    %
    % Returns:
    %   dynamics - Function handle @(t,x) that can be used with ODE solvers
    %              where x is a 2x1 state vector [x1; x2]

    % Define system parameters
    lambda1 = -1;
    lambda2 = 2.5;

    % Define principal eigenfunctions Ψ(x)
    psi = @(x) [x(1) ^ 2 + 2 * x(2) + x(2) ^ 3; % ψ₁(x)
                x(1) + sin(x(2)) + x(1) ^ 3]; % ψ₂(x)

    % Compute gradient matrix ∇Ψ(x)
    syms x1 x2
    psi_sym = [x1 ^ 2 + 2 * x2 + x2 ^ 3; x1 + x1 ^ 3];
    nabla_psi_sym = jacobian(psi_sym, [x1, x2]);
    nabla_psi = matlabFunction(nabla_psi_sym, 'Vars', [x1, x2]);
    nabla_psi = @(x) nabla_psi(x(1), x(2));

    % Define diagonal eigenvalue matrix
    Lambda = diag([lambda1, lambda2]);

    % Define the dynamical system: dx/dt = [∇Ψ(x)]^(-1) * Λ * Ψ(x)
    dynamics = @(t, x) nabla_psi(x) \ (Lambda * psi(x));
end

function visualizeVectorField(dynamics, domain, step_size)
    % VISUALIZEVECTORFIELD Plots the vector field of the dynamical system using streamlines
    %
    % Inputs:
    %   dynamics  - Function handle for the dynamical system
    %   domain    - [xmin, xmax, ymin, ymax] domain bounds
    %   step_size - Grid step size for vector field computation

    % Extract domain bounds
    xmin = domain(1); xmax = domain(2);
    ymin = domain(3); ymax = domain(4);

    % Create grid for vector field visualization
    [X, Y] = meshgrid(xmin:step_size:xmax, ymin:step_size:ymax);

    % Initialize velocity components
    U = zeros(size(X));
    V = zeros(size(Y));

    % Compute velocity field
    for i = 1:size(X, 1)

        for j = 1:size(X, 2)

            dxdt = dynamics(0, [X(i, j); Y(i, j)]);
            U(i, j) = dxdt(1);
            V(i, j) = dxdt(2);

        end

    end

    % Create figure for vector field
    figure('Name', 'Nonlinear System Vector Field', 'Position', [100, 100, 900, 700]);

    % Create streamlines
    % Define starting points for streamlines
    startx = linspace(xmin, xmax, 20);
    starty = linspace(ymin, ymax, 20);
    [StartX, StartY] = meshgrid(startx, starty);

    % Plot streamlines
    streamline(X, Y, U, V, StartX, StartY);
    hold on;

    % Format plot
    xlabel('x₁');
    ylabel('x₂');
    title('Vector Field of Nonlinear Dynamical System (Streamlines)');
    grid on;
    axis equal;
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);

    % Add contour lines for velocity magnitude
    speed = sqrt(U .^ 2 + V .^ 2);
    contour(X, Y, speed, 8, 'LineWidth', 0.5, 'LineColor', [0.8 0.8 0.8], 'LineStyle', '--');

    fprintf('Vector field visualized with streamlines in domain [%.1f, %.1f] × [%.1f, %.1f]\n', ...
        xmin, xmax, ymin, ymax);
    fprintf('Grid step size: %.3f\n', step_size);
end

function [X, Y] = generateTrajectoryData(dynamics)
    % GENERATETRAJECTORYDATA Generates trajectory data for ResDMD analysis
    %
    % Inputs:
    %   dynamics - Function handle for the dynamical system
    %
    % Outputs:
    %   X - Current state data matrix (2 × M)
    %   Y - Next state data matrix (2 × M)

    rng(1); % For reproducible results

    % Set parameters
    M1 = 2000; % Number of initial conditions
    M2 = 20; % Number of time steps per trajectory
    delta_t = 0.01; % Time step

    % ODE solver options
    options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);

    % Initialize data matrices
    X = [];
    Y = [];

    fprintf('Generating %d trajectories with %d time steps each...\n', M1, M2);

    for jj = 1:M1
        % Random initial condition in domain [-2, 2] × [-2, 2]
        Y0 = (rand(2, 1) - 0.5) * 3;

        try
            % Solve ODE for trajectory
            [~, Y1] = ode45(dynamics, [0, 0.000001, (1:(3 + M2)) * delta_t], Y0, options);
            Y1 = Y1';

            % Collect data pairs (current state, next state)
            X = [X, Y1(:, [1, 3:M2 + 1])];
            Y = [Y, Y1(:, 3:M2 + 2)];
        catch
            % Skip trajectories that cause numerical issues
            fprintf('Skipping trajectory %d due to numerical issues\n', jj);
        end

        % Progress indicator
        if mod(jj, 100) == 0
            fprintf('Completed %d/%d trajectories\n', jj, M1);
        end

    end

    M = size(X, 2);
    fprintf('Generated %d data points\n', M);
end

function analyzeWithResDMD(X, Y)
    % ANALYZEWITHRESDOMD Applies ResDMD algorithm and visualizes results
    %
    % Inputs:
    %   X - Current state data matrix (2 × M)
    %   Y - Next state data matrix (2 × M)

    % Add path to main_routines if not already added
    if ~exist('KoopPseudoSpecQR', 'file')
        addpath('main_routines');
    end

    % Set parameters
    N = 100; % Number of basis functions
    PHI = @(r) exp(-r); % Radial basis function

    % Grid for pseudospectrum
    x_pts = -3:0.08:3;
    y_pts = -3:0.08:3;
    v = (10 .^ (-3:0.2:0));

    M = size(X, 2);

    % Scaling for radial function
    d = mean(vecnorm(X - mean(X')'));

    % Find centers using k-means
    fprintf('Computing k-means clustering for %d centers...\n', N);
    [~, C] = kmeans([X'; Y'], N);

    % Build feature matrices
    fprintf('Building feature matrices...\n');
    PX = zeros(M, N);
    PY = zeros(M, N);

    for j = 1:N
        R = sqrt((X(1, :) - C(j, 1)) .^ 2 + (X(2, :) - C(j, 2)) .^ 2);
        PX(:, j) = PHI(R(:) / d);
        R = sqrt((Y(1, :) - C(j, 1)) .^ 2 + (Y(2, :) - C(j, 2)) .^ 2);
        PY(:, j) = PHI(R(:) / d);
    end

    %% Apply ResDMD algorithm (residuals computed after EDMD)
    fprintf('Computing Koopman operator approximation...\n');
    K = PX \ PY;
    [V, LAM] = eig(K, 'vector');
    % save the eigenvalues and eigenvectors to workspace
    assignin('base', 'LAM', LAM);
    assignin('base', 'V', V);
    res = (vecnorm(PY * V - PX * V * diag(LAM)) ./ vecnorm(PX * V))'; % residuals

    %% Visualize eigenvalues with residuals
    figure('Name', 'ResDMD Eigenvalues', 'Position', [200, 200, 800, 600]);
    scatter(real(LAM), imag(LAM), 250, res, '.');
    colormap turbo;
    colorbar;
    title('ResDMD Eigenvalues colored by Residuals');
    xlabel('Real(\lambda)');
    ylabel('Imag(\lambda)');
    ax = gca;
    ax.FontSize = 14;
    axis equal tight;
    grid on;

    %% Compute and visualize pseudospectrum
    fprintf('Computing pseudospectrum...\n');
    z_pts = kron(x_pts, ones(length(y_pts), 1)) + 1i * kron(ones(1, length(x_pts)), y_pts(:));
    z_pts = z_pts(:);

    % Check if parallel computing is available
    if license('test', 'Distrib_Computing_Toolbox')
        parallel_option = 'on';
    else
        parallel_option = 'off';
    end

    try
        RES = KoopPseudoSpecQR(PX, PY, 1 / M, z_pts, 'Parallel', parallel_option);
        RES = reshape(RES, length(y_pts), length(x_pts));

        %% Plot pseudospectrum
        figure('Name', 'ResDMD Pseudospectrum', 'Position', [300, 300, 900, 700]);

        % Create meshgrid for plotting
        [X_grid, Y_grid] = meshgrid(x_pts, y_pts);

        % Plot pseudospectrum
        contourf(X_grid, Y_grid, log10(max(min(v), real(RES))), log10(v));
        hold on;

        % Plot symmetric part
        contourf(X_grid, -Y_grid, log10(max(min(v), real(RES))), log10(v));

        % Format plot
        cbh = colorbar;
        cbh.Ticks = log10([0.001, 0.01, 0.1, 1]);
        cbh.TickLabels = [0.001, 0.01, 0.1, 1];
        clim([log10(min(v)), 0]);
        colormap gray;

        xlabel('Re(\lambda)', 'Interpreter', 'latex', 'FontSize', 18);
        ylabel('Im(\lambda)', 'Interpreter', 'latex', 'FontSize', 18);
        title(sprintf('Pseudospectrum $\\mathrm{Sp}_\\epsilon(\\mathcal{K})$, $N=%d$', N), ...
            'Interpreter', 'latex', 'FontSize', 18);

        ax = gca;
        ax.FontSize = 16;
        axis equal tight;
        axis([x_pts(1), x_pts(end), -y_pts(end), y_pts(end)]);
        grid on;

        fprintf('ResDMD analysis completed successfully!\n');

    catch ME
        fprintf('Error computing pseudospectrum: %s\n', ME.message);
        fprintf('Make sure KoopPseudoSpecQR function is available in main_routines/\n');
    end

    % Save results to workspace
    assignin('base', 'X_data', X);
    assignin('base', 'Y_data', Y);
    assignin('base', 'eigenvalues', LAM);
    assignin('base', 'residuals', res);
    assignin('base', 'koopman_matrix', K);

    fprintf('\nResDMD results saved to workspace: X_data, Y_data, eigenvalues, residuals, koopman_matrix\n');
end

% End of script
