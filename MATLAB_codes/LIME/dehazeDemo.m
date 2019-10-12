

clc;
close all;
clear all;
addpath(genpath('./'))

% Input image
filename = 'sam';
I = imread([filename, '.bmp']);

%% Do the job

wsz = 15; % window size
A = globalLight(imresize(I,0.2), wsz); % estimate global air light

% make sure the boundary constraints
ts = Boundcon(I, A, 30, 300);

% depth map refinement 
lambda = .15;
sigma = 2; 
strategy = 3;

t = spedupS(ts,lambda,sigma,strategy);
res = deHaze(I, t, A, 0.85); 

% show the results
figure, imshow(I)
figure, imshow(res);

