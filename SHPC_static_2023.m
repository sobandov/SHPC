% This MATLAB script aims to find the optimal angle for a compensation wave to achieve the best hologram reconstruction. 
% It implements a 'fast' version of the algorithm.

% Authors: Sofia Obando-Vasquez(1), Ana Doblas (2) and Carlos trujillo (1)
% Date: 08-04-2023
% (1) Universidad EAFIT, Medellin, Colombia
% (2) The University of Memphis, Memphis, TN, United States

% Inputs:
% path: path of the folder where the holograms are
% file: name of the hologram
% ext: extension of the file
% dx,dy: pixel pitch of the digital sensor in each direction
% lambda: wavelenght of the register 
% region: for the spatial, one must select in which region of the Fourier 
% spectrum want to perform the spatial filter. Asigne a value of 1-4
% according to the cartessian regions of a plane. 
% save: a boolean parameter for saving or not the Amplitude and phase
% information of the reconstruction as .bmp images

% Outputs: 
% phase: the phase reconstruction of the hologram
%
%

%%
% Clear workspace, close all figures, and clear command window
clear all
close all
clc

% Define file path and filename
path = 'C:\Users\sofio\OneDrive - Universidad EAFIT\EAFIT - Maestría\MAESTRIA\PUBLICACIONES\ROI fast method - review\RESULTADOS\PHASE ACCURACY\';
file = 'holo_star_FB_25um';
ext = '.TIFF';

% Define dx and dy (pixel pitch in um)
dx = 2.4;
dy = 2.4; 

% Define wavelength 
lambda = 0.532;

% Define the Cartessian region for the spatial filter 
region = 1;

% Concatenate file path and extension
filename = strcat(path,file);
filename = strcat(filename,ext);

% Call phase_rec function
phase = phase_rec(filename, dx, dy, lambda, region, path, file, 'false');

figure(1)
imagesc(phase), colormap gray
%% functions

% Main function to perform phase reconstruction
function  phase = phase_rec(filename, dx, dy, lambda, region, path, file, save)

% Read hologram image and convert it to double precision
holo=double(imread(filename));

% If the image is RGB, use only one channel (assume grayscale image)
holo = holo(:,:,1);

% Get the size of the hologram
[N,M] = size(holo);
% Create a meshgrid for the hologram
[m,n] = meshgrid(-M/2:M/2-1,-N/2:N/2-1);

% Calculate the Fourier Transform of the hologram and shift the zero-frequency component to the center
ft_holo = fftshift(fft2(fftshift(holo)));

% Initialize a filter with zeros
filter = zeros(N,M);

% Create a filter mask for the desired region
if region==1
    filter(1:round(N/2-(N*0.1)),round(M/2+(M*0.1)):M) = 1; % 1nd quadrant
elseif region==2
    filter(1:round(N/2-(N*0.1)),1:round(M/2-(M*0.1))) = 1;  % 2nd quadrant
elseif region==3
    filter(round(N/2+(N*0.1)):N,1:round(M/2-(M*0.1))) = 1; % 3nd quadrant
else
    filter(round(N/2+(N*0.1)):N,round(M/2+(M*0.1)):M) = 1; % 4nd quadrant
end


% Apply the filter to the Fourier Transform of the hologram
ft_filtered_holo = ft_holo .* filter;
filtered_spect = log(abs(ft_filtered_holo).^2);
% Find the maximum value in the filtered spectrum
[~,idx] = max(filtered_spect(:));


% Define wavenumber
k = 2 * pi / lambda;

% Calculate the center frequencies for fx and fy
fx_0 = M/2;
fy_0 = N/2;

% Get the maximum values of fx and fy
[fy_max,fx_max] = ind2sub([N,M],idx);

% Define the step size for the search
step = 0.5;

% Initialize variables for the search
j = 0;

% Calculate the Inverse Fourier Transform of the filtered hologram
holo_rec = fftshift(ifft2(fftshift(ft_filtered_holo)));

% Define the search range (G)
G = 3;
% Initialize flag for the search loop
fin = 0;

% Set initial values for fx and fy
fx = fx_max;
fy = fy_max;

% Initialize temporary search range
G_temp = G;

tic
% Loop to find the optimal fx and fy values
while fin == 0
  i = 0;
  j = j + 1;
  
  % Initialize the maximum sum (for thresholding)
  suma_maxima=0;
  
  % Nested loops for searching in the range of fx and fy
  for fy_tmp = fy-step*G_temp:step:fy+step*G_temp
    for  fx_tmp = fx-step*G_temp:step:fx+step*G_temp
      i = i + 1;
      
      % Calculate the metric for the current fx and fy
      suma = metric(holo_rec, fx_0, fy_0, fx_tmp, fy_tmp, lambda, M, N , dx, dy, m, n, k);
      
      % Update maximum sum and corresponding fx and fy if 
      % current sum is greater than the previous maximum
      if (suma > suma_maxima)
        x_max_out = fx_tmp;
        y_max_out = fy_tmp;
        suma_maxima=suma;
      end         
    end
  end
  
  % Update the temporary search range
  G_temp = G_temp - 1;
  
  % Check if the optimal values are found, set the flag to exit the loop
  if x_max_out == fx && y_max_out == fy
    fin = 1;        
  end
  
  % Update fx and fy for the next iteration
  fx = x_max_out;
  fy = y_max_out;  
end
toc
% Calculate the angles for the compensation wave
theta_x = asin((fx_0 - x_max_out) * lambda / (M * dx));
theta_y = asin((fy_0 - y_max_out) * lambda / (N * dy));

% Calculate the reference wave
ref = exp(1i * k * (sin(theta_x) * m * dx + sin(theta_y) * n * dy));

% Apply the reference wave to the hologram reconstruction
holo_rec2 = holo_rec .* (ref);
Ampl = abs(holo_rec2);
phase = angle(holo_rec2);


% Normalize the phase and convert it to uint8
phase = mat2gray(phase);
phase = uint8(phase * 255);

% Normalize the amplitude and convert it to uint8
Ampl = mat2gray(Ampl);
Ampl = uint8(Ampl * 255);

if save == true
    % Save the phase image
    filename2 = strcat(path, file);
    imwrite(phase,gray(256),strcat(filename2,'Phase_SHPC_output.bmp'))

    % Save the phase image
    filename2 = strcat(path, file);
    imwrite(Ampl,gray(256),strcat(filename2,'Amplitude_SHPC_output.bmp'))
end

end

% Function to calculate the metric for the current fx and fy
function suma = metric(holo_rec, fx_0, fy_0, fx_tmp, fy_tmp, lambda, M, N , dx, dy, m, n, k)

   % Calculate the angles for the compensation wave
   theta_x = asin((fx_0 - fx_tmp) * lambda / (M * dx));
   theta_y = asin((fy_0 - fy_tmp) * lambda / (N * dy));

   % Calculate the reference wave
   ref = exp(1i * k * (sin(theta_x) * m * dx + sin(theta_y) * n * dy));

   % Apply the reference wave to the hologram reconstruction
   holo_rec2 = holo_rec .* ref;
        
   % Calculate the phase of the hologram reconstruction
   phase = angle(holo_rec2);

   % Normalize the phase and convert it to uint8
   phase = mat2gray(phase);
   phase = uint8(phase * 255);

   % Threshold the phase image
   BW = imbinarize(phase, 0.1);

   % Calculate the sum of all elements in the resulting binary image
   suma = sum(sum(BW));
end
