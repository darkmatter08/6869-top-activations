% do proj_setup.m
%% Layer Indicies
nlayers = size(net.layers);
conv_indicies = [];
for i = 1:nlayers(2)
   if strcmp(net.layers{i}.type, 'conv') && (i < 14)
       conv_indicies = [conv_indicies; i];
   end
end

%% Calculate determinants
layer_determinants = struct;

for conv_i = 1:numel(conv_indicies)
    layer_i = conv_indicies(conv_i);
    dets = [];
    [r_w r_h one nfilters] = size(net.layers{layer_i}.filters(:,:,1,:));
    for filter_i = 1:nfilters
        A = net.layers{layer_i}.filters(:,:,1,filter_i);
        dets = [dets norm(A, 'fro')];
    end
    layer_determinants.(strcat('i', num2str(layer_i))) = dets;
end

%% Remove FC layers, whose 'type' field is also 'conv'
% layer_determinants = rmfield(layer_determinants, 'i14');
% layer_determinants = rmfield(layer_determinants, 'i16');
% layer_determinants = rmfield(layer_determinants, 'i18');

%% Get top N weights per layer
for conv_i = 1:numel(conv_indicies)
    layer_i = conv_indicies(conv_i);
    nodes = getfield(layer_determinants, strcat('i', num2str(layer_i)));
    [val, idx] = max(nodes);
    disp(layer_i);
    disp(idx)
end
