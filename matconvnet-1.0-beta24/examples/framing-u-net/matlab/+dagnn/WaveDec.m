classdef WaveDec < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = false
    opts = {'cuDNN'}
    
    wname = 'haar'  % Wavelet
    ker = 'LL'  % Low-Low
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      
      sz_in     = size(inputs{1});
      sz_out    = sz_in;
      sz_out(1) = sz_out(1)/2;
      sz_out(2) = sz_out(2)/2;
%       sz_out	= [sz_in(1)/2, sz_in(2)/2, sz_in(3), sz_in(4)];
      
      outputs{1} = vl_nnconv(...
        reshape(inputs{1}, sz_in(1), sz_in(2), 1, []), params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    
      outputs{1} = reshape(outputs{1}, sz_out);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      
      sz_in     = size(inputs{1});
      sz_out    = sz_in;
      sz_out(1) = sz_out(1)/2;
      sz_out(2) = sz_out(2)/2;
%       sz_out	= [sz_in(1)/2, sz_in(2)/2, sz_in(3), sz_in(4)];
      
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        reshape(inputs{1}, sz_in(1), sz_in(2), 1, []), params{1}, params{2}, ...
        reshape(derOutputs{1}, sz_out(1), sz_out(2), 1, []), ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    
      derInputs{1} = reshape(derInputs{1}, sz_in);
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
        Lo_D    = [0.7071, 0.7071];
        Hi_D    = [-0.7071, 0.7071];
        
        switch obj.ker
            case 'LL'
                kernel	= Lo_D'*Lo_D;
            case 'HL'
                kernel	= Hi_D'*Lo_D;
            case 'LH'
                kernel	= Lo_D'*Hi_D;
            case 'HH'
                kernel	= Hi_D'*Hi_D;
        end
        
        params{1}   = single(kernel);
        
%       % Xavier improved
%       sc = sqrt(2 / prod(obj.size(1:3))) ;
%       %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
%       params{1} = randn(obj.size,'single') * sc ;
%       if obj.hasBias
%         params{2} = zeros(obj.size(4),1,'single') ;
%       end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = WaveDec(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
      
      obj.wname = obj.wname;
      obj.ker = obj.ker;
    end
  end
end
