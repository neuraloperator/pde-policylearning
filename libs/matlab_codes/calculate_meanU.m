function meanU = calculate_meanU(ym, U)
    % Calculate the average velocity profile in the x-direction
    mean_velocity_profile = mean(mean(U(:, 2:end-1, :), 3), 1);
    
    % Create the y-values for the trapezoidal integration
    y_values = [0; ym; 2];
    
    % Perform the trapezoidal integration
    meanU = trapz(y_values, [0, mean_velocity_profile, 0]) / 2;
end