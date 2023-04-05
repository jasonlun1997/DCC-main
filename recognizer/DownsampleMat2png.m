clear; close all;
%% >>>>> turn downsample mat file to downsample png file<<<<<<<<

src_data_path = '../datasets/';
src_datasets = dir(src_data_path);
src_datasets(1:2) = [];
num_datasets = length(src_datasets);
angRes = 5;
factor = 2; %2  %4
patchsize = factor*32;  
stride = patchsize/2; 
src_dataset_for = 'test'; % training %test

%%mkdir 
SavePath_noncenter = ['./recognizer/dataset/dataset_',num2str(factor),'x_',src_dataset_for,'-position/noncenter'];
if exist(SavePath_noncenter, 'dir')==0
    mkdir(SavePath_noncenter);
end
SavePath_center = ['./recognizer/dataset/dataset_',num2str(factor),'x_',src_dataset_for,'-position/center'];
if exist(SavePath_center, 'dir')==0
    mkdir(SavePath_center);
end
SavePath_420A = ['./recognizer/dataset/dataset_',num2str(factor),'x_',src_dataset_for,'-scheme/420A'];
if exist(SavePath_420A, 'dir')==0
    mkdir(SavePath_420A);
end
SavePath_bicubic = ['./recognizer/dataset/dataset_',num2str(factor),'x_',src_dataset_for,'-scheme/bicubic'];
if exist(SavePath_bicubic, 'dir')==0
    mkdir(SavePath_bicubic);
end

%% classification data generation
for index_dataset = 1 : num_datasets 
    idx_save = 0;
    name_dataset = src_datasets(index_dataset).name;
    src_sub_dataset = [src_data_path, name_dataset, '/', src_dataset_for ,'/'];  %training     %test
    scenes = dir(src_sub_dataset);
    scenes(1:2) = [];
    num_scene = length(scenes); 
    
    for index_scene = 1 : num_scene
        % Load LF image
        idx_scene_save = 0;
        name_scene = scenes(index_scene).name;
        data_path = [src_sub_dataset, name_scene];
        data = load(data_path);
        LF = data.LF;
        [U, V, ~, ~, ~] = size(LF);
        % Extract central angRes*angRes views
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, 1:3); % Extract central angRes*angRes views
        [U, V, H, W, ~] = size(LF);%[5, 5, 512, 512, 3]            
        H = floor(H/4) * 4;%H - mod(H, 4);  %   Divisible 4 and multiple 4
        W = floor(W/4) * 4;%W - mod(W, 4);
        for u = 1 : U
            for v = 1 : V
                temp_Hr_rgb = double(squeeze(LF(u, v, 1:H ,  1:W, :)));   %[64, 64, 3]
                temp_Hr_ycbcr = rgb2ycbcr(temp_Hr_rgb);             %[64, 64, 3]
                tmp = ycbcr2rgb(temp_Hr_ycbcr);                     %[64, 64, 3]
                temp_Hr_y = squeeze(temp_Hr_ycbcr(:,:,1)); %need more training may need (:,:,1)/ (:,:,2)/ (:,:,3) separately execute
                imshow(temp_Hr_y);
                down = ["bicubic" "420A" "420D"];
                for down_i = 1:3    %3
                    if (strcmp(down(down_i),"420A"))
                        patch_Sr_y = convert420_A(temp_Hr_y,factor);   %420A down   
                        imgname = [SavePath_center,'/',char(down(down_i)),'_', name_dataset,'_',name_scene,'_', num2str(u,'%01d'),'_', num2str(v,'%01d'),'_cr.png'];
                        imgname2 = [SavePath_420A,'/',char(down(down_i)),'_', name_dataset,'_',name_scene,'_', num2str(u,'%01d'),'_', num2str(v,'%01d'),'_cr.png'];
                        pause(0.005);
                        imwrite(patch_Sr_y,imgname,'png');
                        imwrite(patch_Sr_y,imgname2,'png');
                    
                    elseif (strcmp(down(down_i),"420D"))
                        patch_Sr_y = convert420_D(temp_Hr_y,factor);   %420D down      
                        imgname = [SavePath_noncenter,'/',char(down(down_i)),'_', name_dataset,'_',name_scene,'_', num2str(u,'%01d'),'_', num2str(v,'%01d'),'_CR.png'];
                        pause(0.005);
                        imwrite(patch_Sr_y,imgname,'png');

                    elseif (strcmp(down(down_i),"bicubic"))
                        patch_Sr_y = imresize(temp_Hr_y, [floor(H/factor), floor(W/factor)]);   %bicubic down
                        imgname = [SavePath_center,'/',char(down(down_i)),'_', name_dataset,'_',name_scene,'_', num2str(u,'%01d'),'_', num2str(v,'%01d'),'_cr.png'];
                        imgname2 = [SavePath_bicubic,'/',char(down(down_i)),'_', name_dataset,'_',name_scene,'_', num2str(u,'%01d'),'_', num2str(v,'%01d'),'_cr.png'];
                        pause(0.005);
                        imwrite(patch_Sr_y,imgname,'png');
                        imwrite(patch_Sr_y,imgname2,'png');
                    end      
                    idx_scene_save = idx_scene_save + 1;

                end %for down_i = 1:3
            end
        end %for u = 1 : U
    fprintf([num2str(idx_scene_save), src_dataset_for,' samples have been generated\n']);
    end %for index_scene = 1 : num_scene
end


%% functions
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


function lf_DS = convert420_A(hr, scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hr [h,w,1,ah,aw] --> lf [h/scale,w/scale,1,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = size(hr,1);
    W = size(hr,2);
    
    lf_DS = zeros(ceil(H/2), ceil(W/2),'single');
    %lf_DS = zeros(floor(H/2), floor(W/2), size(hr,3), size(hr,4),'single');
    for i = 1:ceil(H/2)
        for j = 1:ceil(W/2)
            lf_DS(i,j) = hr(i*2,j*2)/4 + hr(mir(i*2-1, H),j*2)/4 + hr(i*2,mir(j*2-1, W))/4 + hr(mir(i*2-1, H), mir(j*2-1, W))/4;
        end
    end               

    if (scale==4)
        lf_DS = convert420_A(lf_DS,2);
    end
end



% 
function lf_DS = convert420_D(hr, scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	12/6
% hr [h,w,1,ah,aw] --> lf [h/scale,w/scale,1,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = size(hr,1);
    W = size(hr,2);
    
    lf_DS = zeros(ceil(H/2), ceil(W/2), 'single');
    %lf_DS = zeros(floor(H/scale), floor(W/scale), size(hr,3), size(hr,4),'uint8');
    for i = 1:ceil(H/2)
        for j = 1:ceil(W/2)
            lf_DS(i,j) = hr(mir(i*2-1, H), mir(j*2-1,W));
        end
    end                       
    if (scale==4)
        lf_DS = convert420_D(lf_DS,2);
    end
end
%