% load_and_plot.m — reproduce the key PINN figures in MATLAB (no toolboxes needed).
% Reads results/matlab/nozzle_surrogate.mat written by src/export.py.
% Arrays: X, DT (grids), V, P (PINN, SI), V_exact, P_exact (analytical), D (geometry).

S = load('../../results/matlab/nozzle_surrogate.mat');  % adjust path if run elsewhere

% --- (1) colored pressure field, smallest throat in the sweep ---
figure('Position', [50 50 1100 800]);
subplot(2,2,1);
contourf(S.X, S.X*0, S.P/1000, 60, 'LineColor', 'none'); % collapse y: use pcolor instead
pcolor(S.X, (S.D/2).*linspace(-1,1,2)'.*ones(size(S.X)), [S.P; S.P]/1000); shading interp;
title('Pressure field [kPa], D_t sweep row 1'); xlabel('x [m]'); colorbar;

% --- (2) throat velocity vs throat diameter: PINN vs exact ---
i_throat = round(size(S.X,2)/2);
subplot(2,2,2);
plot(S.DT(:,i_throat), S.V(:,i_throat), 'o-', 'LineWidth', 1.5); hold on;
plot(S.DT(:,i_throat), S.V_exact(:,i_throat), '--', 'LineWidth', 1.5);
xlabel('D_t [m]'); ylabel('throat velocity [m/s]');
legend('PINN surrogate', 'exact (D_{in}/D_t)^2 V_{in}', 'Location', 'northwest');
title('Throat velocity vs throat diameter'); grid on;

% --- (3) throat pressure vs throat diameter ---
subplot(2,2,3);
plot(S.DT(:,i_throat), S.P(:,i_throat)/1000, 'o-', 'LineWidth', 1.5); hold on;
plot(S.DT(:,i_throat), S.P_exact(:,i_throat)/1000, '--', 'LineWidth', 1.5);
xlabel('D_t [m]'); ylabel('throat pressure [kPa]');
legend('PINN surrogate', 'exact Bernoulli', 'Location', 'northeast');
title('Throat pressure vs throat diameter'); grid on;

% --- (4) relative error map across the whole design space ---
subplot(2,2,4);
err = abs(S.V - S.V_exact) ./ max(abs(S.V_exact), 1e-12) * 100;
contourf(S.X, S.DT(:,1)*0 + S.DT(:,1), err, 40, 'LineColor', 'none');
xlabel('x [m]'); ylabel('D_t [m]'); title('|V - V_{exact}| / |V_{exact}| [%]'); colorbar;
