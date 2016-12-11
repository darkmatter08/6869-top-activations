% do proj_setup.m
%% Get activations for all imgs
imgs = dir('/home/osboxes/Documents/top_weights/imgs/*.jpg');
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
        for filter_i = 1:nfilters
            A = res(layer_i).x(:,:,filter_i);
            dets = [dets norm(A, 'fro')];
        end
        activation_determinants.(strcat('i', num2str(layer_i))) = dets;
    end
    all_act_dets.(imgs_fields{img}) = activation_determinants;
end

%% Get top N activations per layer
all_act_dets_fields = fieldnames(all_act_dets);
all_top10 = struct;
for act_det = 1:numel(all_act_dets_fields)
    activation_determinants = all_act_dets.(all_act_dets_fields{act_det});
    top10 = [];
    N = 10;
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
% x = layer_i
% y = activation idx

all_top10_fields = fieldnames(all_top10);
for top10_idx = 1:numel(all_top10_fields)
    top10 = all_top10.(all_top10_fields{top10_idx});
    layer_i_expanded = [];
    for conv_i = 1:numel(conv_indicies)
       layer_i_expanded = [layer_i_expanded; repmat(conv_indicies(conv_i) , N, 1)]; 
    end
    figure;
    scatter(layer_i_expanded, top10);
    fname = strrep(all_top10_fields{top10_idx}, '_', '\_');
    title(strcat('Top10 Activation plot for :', ' ', fname));
    % TODO add an indicator of nfilters per layer
end
%close all
    