%% Initialization
clear all;
clc;

%% Parameters setting
angRes = 5;                 % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor = 2;                 % SR factor
downRatio = 1/factor;
src_data_path = './datasets/'; 
src_datasets = dir(src_data_path);
src_datasets(1:2) = [];
num_datasets = length(src_datasets); 


%% Test data generation
for index_dataset = 1 : num_datasets 
    idx_save = 0;
    name_dataset = src_datasets(index_dataset).name;
    src_sub_dataset = [src_data_path, name_dataset, '/test/'];
    scenes = dir(src_sub_dataset);
    scenes(1:2) = [];
    num_scene = length(scenes); 
    
    for index_scene = 1 : num_scene 
        % Load LF image
        idx_scene_save = 0;
        name_scene = scenes(index_scene).name;
        fprintf('Generating test data of Scene_%s in Dataset %s......\t\t', name_scene, src_datasets(index_dataset).name);
        data_path = [src_sub_dataset, name_scene];
        data = load(data_path);
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        while mod(H, 4) ~= 0
            H = H - 1;
        end
        while mod(W, 4) ~= 0
            W = W - 1;
        end
        
        % Extract central angRes*angRes views
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), 1:H, 1:W, 1:3); % Extract central angRes*angRes views
        [U, V, H, W, ~] = size(LF);
    
        % Convert to YCbCr
        idx_save = idx_save + 1;
        idx_scene_save = idx_scene_save + 1;
        Hr_SAI_y = single(zeros(U * H, V * W)); 
        Lr_SAI_y = single(zeros(U * H * (1/factor), V * W * (1/factor)));  
        Sr_SAI_cbcr = single(zeros(U * H, V * W, 2));
    
        for u = 1 : U
            for v = 1 : V
                x = (u-1)*H+1;
                y = (v-1)*W+1;
                %                                       e.g. HCI/origami.mat.h5
                temp_Hr_rgb = double(squeeze(LF(u, v, :, :, :)));   %[512, 512, 3]
                temp_Hr_ycbcr = rgb2ycbcr(temp_Hr_rgb);             %[512, 512, 3]
                tmp = ycbcr2rgb(temp_Hr_ycbcr);                     %[512, 512, 3]
                Hr_SAI_y(x:u*H, y:v*W) = single(temp_Hr_ycbcr(:,:,1));          

                temp_Hr_y = squeeze(temp_Hr_ycbcr(:,:,1));
				
                %temp_Lr_y = convert420_A(temp_Hr_y,factor);   %420A down    (option)
                %temp_Lr_y = convert420_D(temp_Hr_y,factor);   %420D down    (option)
                temp_Lr_y = imresize(temp_Hr_y, downRatio);    %bicubic down (option)
				
                Lr_SAI_y((u-1)*H*(1/factor)+1 : u*H*(1/factor), (v-1)*W*(1/factor)+1:v*W*(1/factor)) = single(temp_Lr_y);

                tmp_Hr_cbcr = temp_Hr_ycbcr(:,:,2:3);
                tmp_Lr_cbcr = imresize(tmp_Hr_cbcr, 1/factor);
                tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, factor);
                Sr_SAI_cbcr(x:u*H, y:v*W, :) = tmp_Sr_cbcr;
            end
        end 
                                    %SR_%SR_420A_%SR_420D_
        SavePath = ['./data_for_test/SR_', num2str(angRes), 'x' , num2str(angRes), '_' ,num2str(factor), 'x/', name_dataset,'/' ];%8/15 _y
        if exist(SavePath, 'dir')==0
            mkdir(SavePath);
        end

        SavePath_H5 = [SavePath, name_scene,'.h5'];
        
        h5create(SavePath_H5, '/Hr_SAI_y', size(Hr_SAI_y), 'Datatype', 'single');
        h5write(SavePath_H5, '/Hr_SAI_y', single(Hr_SAI_y), [1,1], size(Hr_SAI_y));
        
        h5create(SavePath_H5, '/Lr_SAI_y', size(Lr_SAI_y), 'Datatype', 'single');
        h5write(SavePath_H5, '/Lr_SAI_y', single(Lr_SAI_y), [1,1], size(Lr_SAI_y));
        
        h5create(SavePath_H5, '/Sr_SAI_cbcr', size(Sr_SAI_cbcr), 'Datatype', 'single');
        h5write(SavePath_H5, '/Sr_SAI_cbcr', single(Sr_SAI_cbcr), [1,1,1], size(Sr_SAI_cbcr));

        fprintf([num2str(idx_scene_save), ' test samples have been generated\n']);
    end
end

%%%
function  output = mir(x, edge)
    if (x < 0)
        output = -x;
    elseif (x >= edge)
        output = edge - (x - edge + 2);
    elseif (x == 0)
        output = 1;
    else
        output = x;
    end
end

%%%clip , bound
function output = clip(x, min, max)
    output = x;
    if x < min
        output = min;
    elseif x > max
        output = max;
    end
end

function lf_DS = convert420_A(hr, scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hr [h,w,1,ah,aw] --> lf [h/scale,w/scale,1,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = size(hr,1);
    W = size(hr,2);
    
    lf_DS = zeros(ceil(H/2), ceil(W/2), size(hr,3), size(hr,4),'single');
    %lf_DS = zeros(floor(H/2), floor(W/2), size(hr,3), size(hr,4),'single');
    for v = 1:size(hr, 3)
        for u = 1:size(hr, 4)
            for i = 1:floor(H/2)
                for j = 1:floor(W/2)
                    lf_DS(i,j,v,u) = hr(i*2,j*2,v,u)/4 + hr(mir(i*2-1, H),j*2,v,u)/4 + hr(i*2,mir(j*2-1, W),v,u)/4 + hr(mir(i*2-1, H), mir(j*2-1, W),v,u)/4;
                end
            end               
        end
    end  
    if (scale==4)
        lf_DS = convert420_A(lf_DS,2);
    end
end



function lf_DS = convert420_D(hr, scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	12/6
% hr [h,w,1,ah,aw] --> lf [h/scale,w/scale,1,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = size(hr,1);
    W = size(hr,2);
    
    lf_DS = zeros(ceil(H/2), ceil(W/2), size(hr,3), size(hr,4),'single');
    for v = 1:size(hr, 3)
        for u = 1:size(hr, 4)
            for i = 1:floor(H/2)
                for j = 1:floor(W/2)
                    lf_DS(i,j,v,u) = hr(mir(i*2-1, H), mir(j*2-1,W), v, u);
                end
            end               
        end
    end  
    if (scale==4)
        lf_DS = convert420_D(lf_DS,2);
    end
end
