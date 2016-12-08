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
        [a b] = size(net.layers{layer_i}.filters(:,:,1,filter_i));
        if a == b
            dets = [dets det(net.layers{layer_i}.filters(:,:,1,filter_i))];
        else % bad calc
            A = net.layers{layer_i}.filters(:,:,1,filter_i);
            dets = [dets det(A'*A)];
        end
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

%% Get top N activations per layer having fed forward 1 image
disp('Top Activations:')
activation_determinants = struct;

for conv_i = 1:numel(conv_indicies)
    layer_i = conv_indicies(conv_i) + 1; % TODO: check indicies!
    dets = [];
    [vol_x vol_y nfilters] = size(res(layer_i).x);
    for filter_i = 1:nfilters
        this_filter = res(layer_i).x(:,:,filter_i);
        [a b] = size(this_filter);
        if a == b
            dets = [dets det(this_filter)];
        else % bad calc
            dets = [dets det(this_filter'*this_filter)];
        end
    end
    activation_determinants.(strcat('i', num2str(layer_i))) = dets;
end

%% Get top N activations per layer
for conv_i = 1:numel(conv_indicies)
    layer_i = conv_indicies(conv_i) + 1;
    nodes = getfield(activation_determinants, strcat('i', num2str(layer_i)));
    [val, idx] = max(nodes);
    disp(layer_i)
    disp(idx)
end
