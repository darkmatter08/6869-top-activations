% do proj_setup.m
%% Get activations for all imgs
imgs = dir('/home/osboxes/Documents/top_weights/t_imgs/*.jpg');
allres = struct;

for img = imgs'
   fullpath = strcat(img.folder, '/', img.name);
   disp(fullpath)
   r = getres(fullpath, net);
   nojpg = strrep(img.name, '.jpg', '');
   allres.(strcat('i', nojpg)) = r;
end

%% Layer Indicies
nlayers = size(net.layers);
conv_indicies = [];
for i = 1:nlayers(2)
   if strcmp(net.layers{i}.type, 'conv') && (i < 14)
       conv_indicies = [conv_indicies; i];
   end
end

%% Get top N activations per layer having fed forward 1 image
all_nfilters = containers.Map;
imgs_fields = fieldnames(allres);
all_act_dets = struct;
for img = 1:numel(imgs_fields)
    res = allres.(imgs_fields{img});
    fprintf('Calculating top activations for img %s:\n', imgs_fields{img});
    activation_determinants = struct;

    for conv_i = 1:numel(conv_indicies)
        layer_i = conv_indicies(conv_i) + 1;
        dets = [];
        [vol_x vol_y nfilters] = size(res(layer_i).x);
        all_nfilters(int2str(conv_i)) = nfilters;
        for filter_i = 1:nfilters
            A = res(layer_i).x(:,:,filter_i);
            dets = [dets norm(A, 'fro')];
        end
        activation_determinants.(strcat('i', num2str(layer_i))) = dets;
    end
    all_act_dets.(imgs_fields{img}) = activation_determinants;
end

%% Get top N activations per layer
N = 10;
all_act_dets_fields = fieldnames(all_act_dets);
all_top10 = struct;
for act_det = 1:numel(all_act_dets_fields)
    activation_determinants = all_act_dets.(all_act_dets_fields{act_det});
    top10 = [];
    for conv_i = 1:numel(conv_indicies)
        layer_i = conv_indicies(conv_i) + 1;
        nodes = getfield(activation_determinants, strcat('i', num2str(layer_i)));
        [sorted_vals, sorted_idxs] = sort(nodes(:), 'descend');
        fprintf('Layer %d: Top %d max activation indexes\n', layer_i, N);
        for cur = 1:N
            fprintf('%d\n', sorted_idxs(cur));
        end
        top10 = [top10; sorted_idxs(1:10)];
    end
    all_top10.(all_act_dets_fields{act_det}) = top10;
end

%% Plot max activations

% all_top10_fields = fieldnames(all_top10);
% for top10_idx = 1:numel(all_top10_fields)
%     top10 = all_top10.(all_top10_fields{top10_idx});
%     layer_i_expanded = [];
%     for conv_i = 1:numel(conv_indicies)
%        layer_i_expanded = [layer_i_expanded; repmat(conv_indicies(conv_i) , N, 1)]; 
%     end
%     figure;
%     scatter(layer_i_expanded, top10);
%     fname = strrep(all_top10_fields{top10_idx}, '_', '\_');
%     title(strcat('Top10 Activation plot for :', ' ', fname));
%     % TODO add an indicator of nfilters per layer
% end

%close all

%% Set up nested map for next part
% layer -> ( node_idx -> count)
NCONVLAYER = 5;
layer_names = fieldnames(activation_determinants);
layer_map = containers.Map(...
    {1, 2, 3, 4, 5}, ... %     layer_names, ... %TODO: generalize
    {containers.Map, containers.Map, containers.Map, containers.Map, containers.Map} ...
);

for layer_i = 1:NCONVLAYER
   idx_to_count = layer_map(layer_i);
   nfilters_layer = all_nfilters(int2str(layer_i));
   for filt_i = 1:nfilters_layer
       idx_to_count(int2str(filt_i)) = 0;
   end
end

%% Populate nested map
all_top10_fields = fieldnames(all_top10);
for top10_idx = 1:numel(all_top10_fields)
    top10 = all_top10.(all_top10_fields{top10_idx});
    % each row contains N indicies, the most activated ones in that layer
    top10_by_layer = reshape(top10, N, [])';
    [r, c] = size(top10_by_layer);
    for layer_i = 1:r
        top10_acts = top10_by_layer(layer_i, :);
        for act_i = 1:c
            idx_to_count = layer_map(layer_i);
            dictkey = int2str(top10_acts(act_i));
            idx_to_count(dictkey) = idx_to_count(dictkey) + 1;
        end
    end
end

%% Per layer, which M nodes were activated most frequently?
M = 15;
topM = [];
for layer_i = 1:NCONVLAYER
   node_activation_counts = layer_map(layer_i);
   acts = cell2mat(values(node_activation_counts));
   
   [sorted_vals, sorted_idxs] = sort(acts(:), 'descend');
   fprintf('Layer %d: Top %d max activation indexes\n', layer_i, M);
   for cur = 1:M
       fprintf('idx: %d val: %d\n', sorted_idxs(cur), sorted_vals(cur));
   end
   topM = [topM ; sorted_idxs(1:M)];
end

%% Plot top M nodes per layer across all images
% x = layer_i
% y = activation idx

layer_i_expanded = [];
for conv_i = 1:numel(conv_indicies)
   layer_i_expanded = [layer_i_expanded; repmat(conv_indicies(conv_i) , M, 1)]; 
end
norm_factor = [];
for layer_i = 1:NCONVLAYER
    norm_factor = [norm_factor ; repmat(all_nfilters(int2str(layer_i)), M, 1)];
end
norm_factor = norm_factor ./ 2;
topM_norm = topM - norm_factor;


scatter(layer_i_expanded, topM_norm, [], 'r');
title('Top 15 activated nodes per layer');
set(gca,'XTickLabel',{' '});
set(gca,'YTickLabel',{' '});

% Draw connecting lines:
for layer_i = 1:NCONVLAYER-1
    expanded_rs = reshape(layer_i_expanded, M, [])';
    topM_norm_rs = reshape(topM_norm, M, [])';
    for i = 1:M
        for j = 1:M
            line([expanded_rs(layer_i, i) expanded_rs(layer_i+1, j)], ...
                 [topM_norm_rs(layer_i, i) topM_norm_rs(layer_i+1, j)], ...
                 'LineStyle', ':', ...
                 'LineWidth', 1 ...
            );
        end
    end
end