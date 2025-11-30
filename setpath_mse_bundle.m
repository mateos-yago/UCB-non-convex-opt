function setpath_mse_bundle()
%SETPATH_MSE_BUNDLE Add mse_bundle paths to MATLAB search path.
%
%   Call this from inside the `matlab` folder of the repo, e.g.:
%
%       cd path/to/mse_bundle/matlab
%       setpath_mse_bundle();
%
%   This will add:
%       - the current folder
%       - ./src
%       - ./src/+polyflow
%

    here = fileparts(mfilename('fullpath'));
    src  = fullfile(here, 'src');

    addpath(here);
    addpath(src);

    fprintf('mse_bundle paths added. You can now use the polyflow package.\n');
end
