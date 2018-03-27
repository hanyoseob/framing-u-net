function rec = recon_cnn4img(net, data, opts)

if ~isempty(opts.gpus)
    net.move('gpu');
    net.vars(opts.vid).precious	= true ;
end

rec     = zeros(opts.size, 'single');
data    = data + opts.offset;
set     = opts.set;


for ival	= 1:1:length(set)
    
    iz          = set(ival);
    
    disp([num2str(iz) ' / ' num2str(set(end))]);
    
    data_       = single(squeeze(data(:,:,iz)));
    
    data_patch	= getBatchPatchVal(data_, opts);
    
    if opts.meanNorm
        means_patch_  	= mean(mean(mean(data_patch, 1), 2), 3);
    else
        means_patch_ 	= 0;
    end
    
    data_patch	= bsxfun(@minus, data_patch, means_patch_);
    
    if opts.varNorm
        vars_patch      = max(max(max(abs(data_patch), [], 1), [], 2), [], 3);
    else
        vars_patch      = 1;
    end
    
    data_patch	= bsxfun(@times, opts.wgt*data_patch, 1./vars_patch);
    
    %%
    nbatch      = size(data_patch, 4);
    batch_      = (1:opts.batchSize) - 1;
    
    rec_batch = single([]);
    
    for ibatch  = 1:opts.batchSize:nbatch
        batch                       = ibatch + batch_;
        batch(batch > nbatch)       = [];
        
        data_batch                  = data_patch(:,:,:,batch);
        
        if ~isempty(opts.gpus)
            data_batch	= gpuArray(data_batch);
        end

        net.eval({'input',data_batch}) ;
        rec_batch_  	= net.vars(opts.vid).value;
        
        if strcmp(opts.method, 'residual')
            rec_batch_ 	= data_batch - rec_batch_;
        end

        rec_batch(:,:,:,batch)      = gather(rec_batch_);
    end
    
    rec_batch                       = bsxfun(@times, rec_batch/opts.wgt, vars_patch);
    
    rec_            = getReconPatchVal(rec_batch, opts);
    rec(:,:,ival)	= bsxfun(@plus, rec_, means_patch_);
    
end

rec             = max(rec - opts.offset, 0);

net.reset();
net.move('cpu');

end