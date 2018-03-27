function fov = getFOV(param, wgt)

if nargin < 2
    wgt = 1;
end

szX_h       = param.dX*param.nX/2;
szY_h       = param.dY*param.nY/2;

radius      = fix(param.DSO*tan(param.dStepDctX*param.nNumDctX/2 - abs(param.dOffsetX)))*wgt;

[rr, cc]    = meshgrid(linspace(-szX_h, szX_h, param.nX), linspace(-szY_h, szY_h, param.nY));

fov         = sqrt(rr.^2 + cc.^2) < radius;

end