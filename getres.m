function [ res ] = getres( imgpath, net )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

im = imread(imgpath);
if size(im, 3) > 1, im = rgb2gray(im); end;
im = imresize(im, [32, 100]);

im = single(im);
s = std(im(:));
im = im - mean(im(:));
im = im / ((s + 0.0001) / 128.0);

%% CHAR
% net = load('/home/osboxes/Documents/model_release/charnet.mat');
net2=vl_simplenn_tidy(net);
net2.layers{19}.forward = @(l,prev,next)struct('x',reshape(prev.x,37,[]),'dzdx',[],'dzdw',[],'aux',[],'time',nan,'backwardTime',nan, 'stats', []);
res = vl_simplenn(net2, im);
s = '0123456789abcdefghijklmnopqrstuvwxyz ';
[~,pred] = max(res(end).x, [], 1);
fprintf('Predicted text: %s\n', s(pred));
end