% load_and_plot.m -- reproduce the PINN's CFD-style contour plots in MATLAB,
% no toolboxes required (contourf, colormap-with-a-custom-array, and
% subplot are all base MATLAB).
%
% Reads results/matlab/nozzle_surrogate.mat, written by src/export.py's
% export_mat(). Arrays are all (n_dt x n_x) grids: X (axial position [m]),
% DT (throat diameter [m], repeated along each row), V/P (PINN surrogate,
% SI units), V_exact/P_exact (closed-form Bernoulli solution, SI), and D
% (local duct diameter [m] -- used to reconstruct the duct's actual shape
% for the contour plots below, the same way src/visualize.py's
% plot_colored_field() does on the Python side).

S = load('../../results/matlab/nozzle_surrogate.mat');   % adjust path if run elsewhere
x = S.X(1, :);            % axial stations [m] -- identical across every row
dt_vals = S.DT(:, 1);     % throat diameters swept [m]
n_dt = numel(dt_vals);
n_y = 60;                 % radial resolution for the duct-shape contour plots

% Hand-built blue-white-red diverging colormap (no Image Processing / other
% toolbox needed -- colormap() accepts any N-by-3 RGB array).
half = 32;
b2w = [linspace(0, 1, half)', linspace(0, 1, half)', ones(half, 1)];
w2r = [ones(half, 1), linspace(1, 0, half)', linspace(1, 0, half)'];
redblue = [b2w; w2r];

% =====================================================================
% Figure 1: CFD-style filled contours in the ACTUAL duct shape --
% the ANSYS-style view. Two throat diameters side by side (narrowest and
% widest in the sweep), pressure on top, velocity on the bottom -- same
% 2x2 layout as results/figures/colored_cfd_fields.png on the Python side.
% =====================================================================
figure('Name', 'PINN nozzle surrogate -- CFD-style contours', 'Position', [50 30 1300 850]);

dt_idx = [1, n_dt];
col_labels = {'narrowest D_t', 'widest D_t'};

for col = 1:2
    i = dt_idx(col);
    Yhalf = linspace(-1, 1, n_y)';           % n_y x 1, spans the duct's local half-width
    Ygrid = Yhalf * (S.D(i, :) / 2);         % n_y x n_x -- the duct's actual radial extent at each x
    Xgrid = repmat(x, n_y, 1);               % n_y x n_x

    % --- pressure, filled contour, duct wall outlined in black ---
    subplot(2, 2, col);
    Cp = repmat(S.P(i, :) / 1000, n_y, 1);   % kPa, same value across y (plug flow -- quasi-1D has no radial variation)
    contourf(Xgrid, Ygrid, Cp, 60, 'LineColor', 'none');
    hold on;
    plot(x, S.D(i, :) / 2, 'k-', 'LineWidth', 2);
    plot(x, -S.D(i, :) / 2, 'k-', 'LineWidth', 2);
    colormap(gca, redblue); colorbar; axis equal tight;
    xlabel('x [m]'); ylabel('y [m]');
    title(sprintf('Pressure [kPa] -- D_t = %.3f m (%s)', dt_vals(i), col_labels{col}));

    % --- velocity, same treatment, rainbow colormap (classic CFD-tool look) ---
    subplot(2, 2, col + 2);
    Cv = repmat(S.V(i, :), n_y, 1);          % m/s
    contourf(Xgrid, Ygrid, Cv, 60, 'LineColor', 'none');
    hold on;
    plot(x, S.D(i, :) / 2, 'k-', 'LineWidth', 2);
    plot(x, -S.D(i, :) / 2, 'k-', 'LineWidth', 2);
    colormap(gca, jet); colorbar; axis equal tight;
    xlabel('x [m]'); ylabel('y [m]');
    title(sprintf('Velocity [m/s] -- D_t = %.3f m (%s)', dt_vals(i), col_labels{col}));
end

% =====================================================================
% Figure 2: design-space sweep -- throat quantities vs Dt (PINN vs exact),
% and rel-error filled contours over the WHOLE (x, Dt) design space.
% =====================================================================
figure('Name', 'PINN nozzle surrogate -- design-space sweep', 'Position', [80 60 1150 800]);

i_throat = round(size(S.X, 2) / 2);   % column nearest the throat (x = L/2)

subplot(2, 2, 1);
plot(dt_vals, S.V(:, i_throat), 'o-', 'LineWidth', 1.5); hold on;
plot(dt_vals, S.V_exact(:, i_throat), 'k--', 'LineWidth', 1.5);
xlabel('D_t [m]'); ylabel('throat velocity [m/s]');
legend('PINN surrogate', 'exact (D_{in}/D_t)^2 V_{in}', 'Location', 'northwest');
title('Throat velocity vs throat diameter'); grid on;

subplot(2, 2, 2);
plot(dt_vals, S.P(:, i_throat) / 1000, 'o-', 'LineWidth', 1.5); hold on;
plot(dt_vals, S.P_exact(:, i_throat) / 1000, 'k--', 'LineWidth', 1.5);
xlabel('D_t [m]'); ylabel('throat pressure [kPa]');
legend('PINN surrogate', 'exact Bernoulli', 'Location', 'southwest');
title('Throat pressure vs throat diameter'); grid on;

subplot(2, 2, 3);
errV = abs(S.V - S.V_exact) ./ max(abs(S.V_exact), 1e-12) * 100;
contourf(x, dt_vals, errV, 40, 'LineColor', 'none');
colormap(gca, redblue); colorbar;
xlabel('x [m]'); ylabel('D_t [m]'); title('|V - V_{exact}| / |V_{exact}| [%]');

subplot(2, 2, 4);
errP = abs(S.P - S.P_exact) ./ max(abs(S.P_exact), 1e-12) * 100;
contourf(x, dt_vals, errP, 40, 'LineColor', 'none');
colormap(gca, redblue); colorbar;
xlabel('x [m]'); ylabel('D_t [m]'); title('|P - P_{exact}| / |P_{exact}| [%]');

disp('Done. Figure 1 = CFD-style filled contours in the duct shape (ANSYS-style view).');
disp('Figure 2 = design-space sweeps (throat quantities vs Dt) and relative-error maps.');
