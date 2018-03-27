%% YOU HAVE TO EXCUTE, JUST ONE TIME.
install();

%%
clear ;
reset(gpuDevice(1));

restoredefaultpath();
run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

%%
networkType         = 'dagnn';

solver_handle       = [];

imageRange          = [0, 200];
batchSize           = 4;

gpus                = 1;
train               = struct('gpus', gpus);

lr                  = [-3, -5];
sz                  = [256, 256, 1];
wgt                 = 5e3;

views               = 60120240; % trained with 60, 120, and 240 view, simultenously.

netDir              = './network/';
modelPath           = @(epdir, ep) fullfile(epdir, sprintf('net-epoch-%d.mat', ep));

% NETWORK SETTING
networkMR           = 'cnn_sparse_view_dagnn_multi_init';
methodMR            = 'residual';
expDirMR            = [netDir networkMR '_' methodMR '_input' num2str(sz(1)) '_view' num2str(views)];
epochMR             = findLastCheckpoint(expDirMR);
[netMR, ~, statsMR]	= loadState(modelPath(expDirMR, epochMR));

networkSR           = 'cnn_sparse_view_dagnn_single_init';
methodSR            = 'residual';
expDirSR            = [netDir networkSR '_' methodSR '_input' num2str(sz(1)) '_view' num2str(views)];
epochSR             = findLastCheckpoint(expDirSR);
[netSR, ~, statsSR]	= loadState(modelPath(expDirSR, epochSR));

networkFR       	= 'cnn_sparse_view_dagnn_tight_frame_init';
methodFR           	= 'residual';
expDirFR            = [netDir networkFR '_' methodFR '_input' num2str(sz(1)) '_view' num2str(views)];
epochFR             = findLastCheckpoint(expDirFR);
[netFR, ~, statsFR]	= loadState(modelPath(expDirFR, epochFR));

%%%
rmlyr_set           = {'loss', 'psnr', 'mse'};

for ilyr = 1:length(rmlyr_set)
    lyr = rmlyr_set{ilyr};
    
    netMR.removeLayer(lyr);
    netSR.removeLayer(lyr);
    netFR.removeLayer(lyr);
end

vMR                      	= netMR.getVarIndex('regr') ;
vSR                      	= netSR.getVarIndex('regr') ;
vFR                      	= netFR.getVarIndex('regr') ;

netMR.vars(vMR).precious 	= true ;
netSR.vars(vSR).precious 	= true ;
netFR.vars(vFR).precious 	= true ;

%%

epochMIN	= min([epochMR, epochSR, epochFR]);

for i = 1:epochMIN
    objMR(i)	= statsMR.val(i).objective;
    objSR(i)	= statsSR.val(i).objective;
    objFR(i)	= statsFR.val(i).objective;
    
    psnrMR(i)	= statsMR.val(i).psnr;
    psnrSR(i)	= statsSR.val(i).psnr;
    psnrFR(i)	= statsFR.val(i).psnr;
    
    nmseMR(i)	= statsMR.val(i).mse;
    nmseSR(i)	= statsSR.val(i).mse;
    nmseFR(i)	= statsFR.val(i).mse;
    
end

%% ERROR PLOT
figure(1);
suptitle('ERROR');

subplot(131);
plot(objSR,  	'g-','LineWidth',2);	hold on;
plot(objMR,     'b:','LineWidth',2);
plot(objFR,     'r--','LineWidth',2);	hold off;
legend('Standard CNN', 'U-Net', 'Tight-frame U-Net', 'location', 'NorthEast');
ylabel('objective (validation)', 'FontSize', 15, 'FontWeight', 'bold');
xlabel('The number of epochs', 'FontSize', 15, 'FontWeight', 'bold');
% title('Objective (validation)');
grid on;
grid minor;
ylim([4, 10]);
xlim([1, 150]);

ax  = gca;
ax.FontSize     = 12;
ax.FontWeight 	= 'bold';

subplot(132);
plot(psnrSR,    'g-','LineWidth',2);	hold on;
plot(psnrMR, 	'b:','LineWidth',2);
plot(psnrFR,    'r--','LineWidth',2);	hold off;
legend('Standard CNN', 'U-Net', 'Tight-frame U-Net', 'location', 'NorthEast');
ylabel('PSNR [dB] (validation)', 'FontSize', 15, 'FontWeight', 'bold');
xlabel('The number of epochs', 'FontSize', 15, 'FontWeight', 'bold');
% title('PSNR (validation)');
grid on;
grid minor;
ylim([38, 42]);
xlim([1, 150]);

ax  = gca;
ax.FontSize     = 12;
ax.FontWeight 	= 'bold';

subplot(133);
plot(nmseSR,    'g-','LineWidth',2);       hold on;
plot(nmseMR,    'b:','LineWidth',2);
plot(nmseFR,   'r--','LineWidth',2);	hold off;
legend('Standard CNN', 'U-Net', 'Tight-frame U-Net', 'location', 'NorthEast');
ylabel('NMSE (validation)', 'FontSize', 15, 'FontWeight', 'bold');
xlabel('The number of epochs', 'FontSize', 15, 'FontWeight', 'bold');
% title('NMSE (validation)');
grid on;
grid minor;
ylim([0.5e-4, 2e-4]);
xlim([1, 150]);

ax  = gca;
ax.FontSize     = 12;
ax.FontWeight 	= 'bold';

return ;

%% NETWORK 
mode            = 'test';         % 'test' / 'normal'

netMR.mode      = mode;
netSR.mode      = mode;
netFR.mode      = mode;

%% RECONSTRUCTION OPTIONS
opts_IMG.wgt        = wgt;
opts_IMG.offset     = 0;
opts_IMG.imageSize  = [512, 512, 1];

opts_IMG.inputSize  = [512, 512, 1];
opts_IMG.kernalSize = [0, 0, 1];

opts_IMG.meanNorm	= false;
opts_IMG.varNorm	= false;
opts_IMG.batchSize  = batchSize;
opts_IMG.gpus       = gpus;
opts_IMG.method     = 'residual';
opts_IMG.size       = [512, 512, 1];
opts_IMG.set        = 1;

%% Fig5. for 'Framing U-Net via Deep Convolutional Framelets:Application to Sparse-view CT'
imdb                = load(['./data/imdb_60view.mat']);
% imdb                = load(['./data/imdb_90view.mat']);
% imdb                = load(['./data/imdb_120view.mat']);

dsr                 = imdb.images.dsr;
view                = 720/dsr;

labels              = max(iradon(imdb.images.p, imdb.images.theta, 512), 0);
images              = iradon(imdb.images.p(:,1:dsr:end), imdb.images.theta(1:dsr:end), 512);

%% STANDARD CNN
opts_IMG.vid    	= vSR;
recSR               = recon_cnn4img(netSR, images, opts_IMG);

%% U-NET
opts_IMG.vid     	= vMR;
recMR               = recon_cnn4img(netMR, images, opts_IMG);
        
%% Tight-frame U-NET
opts_IMG.vid    	= vFR;
recFR               = recon_cnn4img(netFR, images, opts_IMG);

%%
nmseIN              = nmse(images, labels);
nmseSR              = nmse(recSR, labels);
nmseMR              = nmse(recMR, labels);
nmseFR              = nmse(recFR, labels);

wnd                 = [0, 0.04];

figure(2); colormap gray;
subplot(231); imagesc(labels, wnd);	axis image off;	title({'Fig.5. Ground truth'});
subplot(232); imagesc(images, wnd);	axis image off;	title({[num2str(view) ' view'], ['NMSE : ' num2str(nmseIN, '%.4e')]});
subplot(234); imagesc(recSR, wnd);  axis image off;	title({'Standard CNN',          ['NMSE : ' num2str(nmseSR, '%.4e')]});
subplot(235); imagesc(recMR, wnd);  axis image off;	title({'U-NET',                 ['NMSE : ' num2str(nmseMR, '%.4e')]});
subplot(236); imagesc(recFR, wnd);	axis image off;	title({'Tight-frame U-NET',     ['NMSE : ' num2str(nmseFR, '%.4e')]});
