%% Step 1. Copy the customized library
copyfile('./matlab', '../../matlab');

%% Step 2. Download the trained network
% standard cnn
network_path    = './network/cnn_sparse_view_dagnn_single_init_residual_input256_view60120240/';
network_name	= [network_path 'net-epoch-150.mat'];
network_url     = 'https://www.dropbox.com/s/3yt24xpng0jn5z1/net-epoch-150.mat?dl=1';

mkdir(network_path);
fprintf('downloading standard cnn from %s\n', network_url) ;
websave(network_name, network_url);

% u-net
network_path    = './network/cnn_sparse_view_dagnn_multi_init_residual_input256_view60120240/';
network_name	= [network_path 'net-epoch-150.mat'];
network_url     = 'https://www.dropbox.com/s/nap5nualurgdo29/net-epoch-150.mat?dl=1';

mkdir(network_path);
fprintf('downloading u-net from %s\n', network_url) ;
websave(network_name, network_url);

% tight-frame u-net
network_path    = './network/cnn_sparse_view_dagnn_tight_frame_init_residual_input256_view60120240/';
network_name	= [network_path 'net-epoch-150.mat'];
network_url     = 'https://www.dropbox.com/s/uhzuf3694v124lc/net-epoch-150.mat?dl=1';

mkdir(network_path);
fprintf('downloading tight-frame u-net from %s\n', network_url) ;
websave(network_name, network_url);