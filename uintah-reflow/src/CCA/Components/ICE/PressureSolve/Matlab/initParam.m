function initParam()
%INITPARAM  Initialize parameter structure.
%   This file contains the options for running the driver programs. Set or
%   modify any parameters in here. the structure PARAM holds the
%   parameters and is a global variable for all functions.
%
%   See also: TESTDISC, TESTADAPTIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

param                       = [];

% Problem type, title, streams
param.problemType           = 'GaussianSource'; %'linear'; %'jump_quad'; %'diffusion_quad_quad';  %'sinsin'; % %'jump_quad'; %'diffusion_quad'; %
param.outputDir             = 'test'; %'sinsin_1level';
param.logFile               = 'testDisc.log';
param.outputType            = 'screen';

% Domain geometry
param.dim                   = 2;                                % Number of dimensions
param.domainSize            = repmat(1.0,[1 param.dim]);        % Domain is from [0.,0.] to [1.,1.]

% AMR hierarchy control
param.maxLevels             = 5;
param.maxPatches            = 5;
param.twoLevel              = 0;
param.twoLevelType          = 'centralHalf';
param.threeLevel            = 0;
param.threeLevelType        = 'leftHalf';

% Modules activation flags
param.fluxInterpOrder       = 1;
param.profile               = 0;
param.setupGrid             = 1;
param.solveSystem           = 1;
param.saveResults           = 1;
param.verboseLevel          = 1;
param.catchException        = 1;

% For testDisc: experiments are parameterized by the resolution.
param.numCellsRange         = 64; %2.^[2:1:5]; %2.^[2:1:12];
