%%%%%%% input:   image, parameters
%%%%%%% output:  estimated illumination: L, estimated reflectance: R, 
%%%%%%%          relative errors: epsilon_R, epsilon_L

%%%%%%% The calculation is in the HSV color space


%clc;
%clear all;

function X = demoFn(filename)
    img = double(imread(filename)); 
    
    if size(img,3)>1
        HSV = rgb2hsv(img);   % RGB space to HSV  space
        S = HSV(:,:,3);       % V layer
    else
        S = img;              % gray image
    end
    
    
    c_1 = 0.01; c_2 = 0.1; lambda = 1;     % set parameters
    
    epsilon_stop = 1e-3;  % stopping criteria
    
    [ R, L, epsilon_R, epsilon_L ] = processing( S, c_1, c_2, lambda, epsilon_stop );
    
    
    %%% Gamma correction
    gamma = 2.2;
    L_gamma = 255 * ((L/255).^(1/gamma));
    enhanced_V = R .* L_gamma;
    HSV(:,:,3) = enhanced_V;
    enhanced_result = hsv2rgb(HSV);  
    
    
    %figure,
    %subplot(2,2,1),imshow(uint8(img)), title('input image');
    %subplot(2,2,2),imshow(uint8(enhanced_result)),title('Gamma correction');
    %subplot(2,2,3),imshow(uint8(L)), title('estimated illumination');
    %subplot(2,2,4),imshow(R), title('estimated reflectance');
    X = uint8(enhanced_result);
end