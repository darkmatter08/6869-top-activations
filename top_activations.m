% do proj_setup.m
%% Layer Indicies
nlayers = size(net.layers);
conv_indicies = [];
for i = 1:nlayers(2)
   if strcmp(net.layers{i}.type, 'conv') && (i < 14)
       conv_indicies = [conv_indicies; i];
   end
end

%% Get top N activations per layer having fed forward 1 image
disp('Top Activations:')
activation_determinants = struct;

for conv_i = 1:numel(conv_indicies)
    layer_i = conv_indicies(conv_i) + 1; % TODO: check indicies!
    dets = [];
    [vol_x vol_y nfilters] = size(res(layer_i).x);
    for filter_i = 1:nfilters
        A = res(layer_i).x(:,:,filter_i);
        dets = [dets norm(A, 'fro')];
    end
    activation_determinants.(strcat('i', num2str(layer_i))) = dets;
end

%% Get top N activations per layer
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

%% Plot max activations
% x = layer_i
% y = activation idx
layer_i_expanded = [];
for conv_i = 1:numel(conv_indicies)
   layer_i_expanded = [layer_i_expanded; repmat(conv_indicies(conv_i) , N, 1)]; 
end
scatter(layer_i_expanded, top10);
% TODO add an indicator of nfilters per layer
