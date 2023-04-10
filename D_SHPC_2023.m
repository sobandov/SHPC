% This MATLAB script aims to find the optimal angle for a compensation wave
% to achieve the best hologram reconstruction. 
% It implements a 'dynamic' version of the algorithm.

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

% Outputs: 
% phase: the phase reconstruction of the hologram
% Compensation times as a .mat file
% Compensation video with the recosntruction of each frame

%%
% Clear workspace, close all figures, and clear command window
clc;
close all;
clear all;

%% Loading the images

% Define file path (input and output) and extension
path = 'C:\Users\sofio\OneDrive - Universidad EAFIT\EAFIT - Maestría\MAESTRIA\PUBLICACIONES\ROI fast method - review\MUESTRAS\MUESTRAS SEMEN\40X\';
ext = '.bmp';
path_OUTPUT = 'C:\Users\sofio\OneDrive - Universidad EAFIT\EAFIT - Maestría\MAESTRIA\PUBLICACIONES\ROI fast method - review\MUESTRAS\RESULTADOS\DRF\Esp 40x G 3 s 0.5\';

% Find total number of images of the selected extension in the folder
TotIm=dir([path '/*.bmp']);
NumIm = size(TotIm,1);

% If numeration of images start in a number different from 1
ImInicial = 350;
% Calculate the total images when ImInicial!=1
Total = (NumIm - ImInicial);

% Define the name of the first image
FileName = strcat(path,'holo_ESP_1',ext);
% Read hologram image 
Image0 = imread(FileName); 
% Get the size of the hologram
[N,M] = size(Image0);

% Zero's matrices that will contain all the frames of the holographic video
Images = zeros(N,M,Total);
Comp = zeros(N,M, Total);

% For loop for loading all the holographic video frames
j = ImInicial;
for i = 1:1:Total
    % Construction of the name of the file to be loaded
    file = num2str(j);
    FileName = strcat(path,'holo_ESP_',file,ext);
    % Read hologram image and convert it to double precision 
    img = double(imread(FileName));
    % Assignment of the hologram to the previously allocated matrix
    Images(:,:,i) = img;
    j=j+1;
end

%%
% Create a meshgrid for the hologram
[m,n] = meshgrid(-M/2:M/2-1,-N/2:N/2-1);

% Define wavelength 
lambda = 0.633;

% Define wavenumber
k = 2 * pi / lambda;

% Define dx and dy (pixel pitch in um)
dx = 3.75;
dy = 3.75;

% Calculate the center frequencies for fx and fy
fx_0 = M/2;
fy_0 = N/2;

% Zero's vector to save measured reconstruction times
V = zeros(1,Total);

% Define the Cartessian region for the spatial filter 
region = 1;
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

%% Compensation for the first Frame

% Define the search range (G)
G =3;
% Initialize counting time
tic

file = num2str(ImInicial);
% Call phase_rec function
[x_max_out, y_max_out, holo_rec,suma_maxima] = phase_rec(Images(:,:,1),N,M, lambda, dx, dy, G, fx_0, fy_0,  k,m,n, filter);

% Calculate the angles for the compensation wave
theta_x = asin((fx_0 - x_max_out) * lambda / (M * dx));
theta_y = asin((fy_0 - y_max_out) * lambda / (N * dy));
% Calculate the reference wave
ref = exp(1i * k * (sin(theta_x) * m * dx + sin(theta_y) * n * dy)); 
% Apply the reference wave to the hologram reconstruction
holo_rec2 = holo_rec .* ref;

phase = angle(holo_rec2);
% Normalize the phase and convert it to uint8
phase = mat2gray(phase);
phase = uint8(phase * 255);

% Save the phase image
path = path_OUTPUT;
filename2 = strcat(path, file);
imwrite(phase,gray(256),strcat(filename2,'D-SHPC_output.bmp'))

% Save the compensation tiem of the first frame
V(1) = toc;
%% Compensation for the second frame and on

% Define the search range (G)
G = 0.4; 
% Update fx and fy for the next iteration
fx = x_max_out;
fy = y_max_out;

% Initialize counter for the loop
j = ImInicial+1;

% Initialize counting time

for i= 2:1:Total
    tic

    file = num2str(j);
    % Call phase_rec_2 function
    [x_max_out_1, y_max_out_1, holo_rec]= phase_rec_2(Images(:,:,i),N,M, lambda, dx, dy, G, fx, fy,fx_0,fy_0, k,m,n, filter);
   
    % Calculate the angles for the compensation wave
    theta_x = asin((fx_0 - x_max_out_1) * lambda / (M * dx));
    theta_y = asin((fy_0 - y_max_out_1) * lambda / (N * dy));
    
    % Calculate the reference wave
    ref = exp(1i * k * (sin(theta_x) * m * dx + sin(theta_y) * n * dy));
   
    % Apply the reference wave to the hologram reconstruction
    holo_rec2 = holo_rec .* ref;
    phase = angle(holo_rec2);
    % Normalize the phase and convert it to uint8
    phase = mat2gray(phase);
    phase = uint8(phase * 255);

    % Save the phase image
    path = path_OUTPUT;
    filename2 = strcat(path, file);
    imwrite(phase,gray(256),strcat(filename2,'D-SHPC_output.bmp'))

    % Save the phase image for construct the recosntruction video
    Comp(:,:,i) = phase;

    % Update fx and fy for the next iteration
    fx = x_max_out_1;
    fy = y_max_out_1;  

    % Update compensation time
    V(i)=toc;

    j=j+1;
end

%% Save compensation video

% Video saving paht
FileNameVideo = strcat(path_OUTPUT,'video_Esp_DRF.avi');
v = VideoWriter(FileNameVideo);
% number of FPS according to the compensation times
v.FrameRate = round(1/mean(V)); 

open(v)
% Generate the compensation video 
for i=1:1:Total
    % Read the compensated image
    phase = Comp(:,:,i);

    % Normalize the phase and convert it to uint8
    phase = mat2gray(phase);
    phase = uint8(phase * 255);

    % Save the video
    writeVideo(v,phase)
end 
close(v)

%% Plot and save the compensation times

% Save the compensation times as a .mat file
FileNameTimes = strcat(path_OUTPUT,'times_ESP_40x_DRF.mat');
save(FileNameTimes, 'V');

% Plot the compensation times
Xplot = (1:Total);
figure(2), plot(Xplot, V)
title('Tiempos de compensacion por cuadro de video')
xlabel('Cuadro del video')
ylabel('Tiempo de compensacion (s)')


%% functions

function [x_max_out, y_max_out, holo_rec,suma_maxima]= phase_rec(Images,N,M, lambda, dx, dy, G, fx_0, fy_0, k,m,n, filter)
holo=Images;

% Calculate the Fourier Transform of the hologram and shift the zero-frequency component to the center
ft_holo=fftshift(fft2(fftshift(holo)));
% Apply the filter to the Fourier Transform of the hologram
ft_filtered_holo = ft_holo .* filter;
filtered_spect = log(abs(ft_filtered_holo).^2);
% Find the maximum value in the filtered spectrum
[~,idx] = max(filtered_spect(:));
% Get the maximum values of fx and fy
[fy_max,fx_max] = ind2sub([N,M],idx);

% Define the step size for the search
step = 0.5;

% Initialize variables for the search
j = 0;

% Calculate the Inverse Fourier Transform of the filtered hologram
holo_rec = fftshift(ifft2(fftshift(ft_filtered_holo)));

% Initialize variables for the search
fin =0;

% Set initial values for fx and fy
fx = fx_max;
fy = fy_max;

% Initialize temporary search range
G_temp = G;

% Loop to find the optimal fx and fy values
while fin == 0
    i = 0;
    j = j + 1;

    % Initialize the maximum sum (for thresholding)
    suma_maxima=0; 

    % Nested loops for searching in the range of fx and fy
    for fy_tmp = fy-step*G_temp:step:fy+step*G_temp
        for fx_tmp = fx-step*G_temp:step:fx+step*G_temp
            i = i + 1;

            % Calculate the metric for the current fx and fy
            suma = metric(holo_rec, fx_0, fy_0, fx_tmp, fy_tmp, lambda, M, N , dx, dy, k, m,n);

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
end


function [x_max_out, y_max_out, holo_rec]= phase_rec_2(Images,N,M, lambda, dx, dy, G, fx, fy,fx_0,fy_0,k,m,n, filter)

holo=Images;
% Calculate the Fourier Transform of the hologram and shift the zero-frequency component to the center
ft_holo=fftshift(fft2(fftshift(holo)));
% Apply the filter to the Fourier Transform of the hologram
ft_filtered_holo = ft_holo .* filter;

% Define the step size for the search
step = 0.2;

% Calculate the Inverse Fourier Transform of the filtered hologram
holo_rec = fftshift(ifft2(fftshift(ft_filtered_holo)));

% Initialize flag for the search loop
fin =0;
% Define the search range (G)
G_temp = G;
% Loop to find the optimal fx and fy values
while fin == 0
    % Initialize the maximum sum (for thresholding)
    suma_maxima=0; 
     
    % Nested loops for searching in the range of fx and fy
    for fy_tmp = fy:step:fy+step*G_temp
        for fx_tmp = fx:step:fx+step*G_temp

            % Calculate the metric for the current fx and fy
            suma = metric(holo_rec, fx_0, fy_0, fx_tmp, fy_tmp, lambda, M, N , dx, dy, k,m,n);
            
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
    G_temp = G_temp - 0.1;

    % Check if the optimal values are found, set the flag to exit the loop
    if x_max_out == fx && y_max_out == fy
        fin = 1;
    end

     % Update fx and fy for the next iteration
    fx = x_max_out;
    fy = y_max_out ;
end
end

% Function to calculate the metric for the current fx and fy
function suma = metric(holo_rec, fx_0, fy_0, fx_tmp, fy_tmp, lambda, M, N , dx, dy, k, m,n)

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
   suma=sum(sum(BW)); 
end
