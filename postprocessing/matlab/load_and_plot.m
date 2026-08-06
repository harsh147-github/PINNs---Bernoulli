% load_and_plot.m
% Pure MATLAB (no toolboxes) — loads results/matlab/nozzle_surrogate.mat and
% reproduces the key figures from README.md Section 10, without needing Python.
%
% Usage (from the repo root, after running scripts/run_parametric_sweep.py):
%   >> cd postprocessing/matlab
%   >> load_and_plot

data = load('../../results/matlab/nozzle_surrogate.mat');
X = data.X; DT = data.DT; V = data.V; P = data.P; D = data.D;
V_exact = data.V_exact; P_exact = data.P_exact;

[n_dt, n_x] = size(X);

% --- Figure (a): V(x), p(x) PINN vs analytical for a few throat diameters ---
figure('Name', 'Fields vs analytical');
idx_to_plot = round(linspace(1, n_dt, min(4, n_dt)));

subplot(1, 2, 1); hold on;
for k = idx_to_plot
    plot(X(k, :), V(k, :), '-', 'DisplayName', sprintf('PINN, Dt=%.3f m', DT(k, 1)));
    plot(X(k, :), V_exact(k, :), 'k--', 'HandleVisibility', 'off');
end
xlabel('x [m]'); ylabel('V [m/s]'); title('Velocity (dashed = analytical)');
legend('Location', 'best'); grid on;

subplot(1, 2, 2); hold on;
for k = idx_to_plot
    plot(X(k, :), P(k, :), '-', 'DisplayName', sprintf('PINN, Dt=%.3f m', DT(k, 1)));
    plot(X(k, :), P_exact(k, :), 'k--', 'HandleVisibility', 'off');
end
xlabel('x [m]'); ylabel('p [Pa]'); title('Pressure (dashed = analytical)');
legend('Location', 'best'); grid on;

% --- Figure (b): throat values vs Dt sweep ---
[~, throat_col] = min(abs(X(1, :) - X(1, round(n_x/2))));
throat_V = V(:, throat_col);
throat_P = P(:, throat_col);
throat_V_exact = V_exact(:, throat_col);
throat_P_exact = P_exact(:, throat_col);
Dt_vals = DT(:, 1);

figure('Name', 'Throat sweep');
subplot(1, 2, 1);
plot(Dt_vals, throat_V, 'o-', Dt_vals, throat_V_exact, 'kx--');
xlabel('Dt [m]'); ylabel('Throat V [m/s]'); legend('PINN', 'Analytical'); grid on;
subplot(1, 2, 2);
plot(Dt_vals, throat_P, 'o-', Dt_vals, throat_P_exact, 'kx--');
xlabel('Dt [m]'); ylabel('Throat p [Pa]'); legend('PINN', 'Analytical'); grid on;

% --- Figure (c): pointwise error map |P - P_exact| over (x, Dt) ---
figure('Name', 'Pressure error map');
err = abs(P - P_exact);
pcolor(X, DT, err); shading interp; colorbar;
xlabel('x [m]'); ylabel('Dt [m]'); title('|p_{PINN} - p_{exact}| [Pa]');

disp('Done. Three figures produced: fields vs analytical, throat sweep, error map.');
