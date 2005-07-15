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
param.problemType           = 'jump_quad'; %'diffusion_quad'; %
param.outputDir             = 'test'; %'sinsin_1level';
param.logFile               = 'testDisc.log';
param.outputType            = 'screen';

% AMR hierarchy control
param.twoLevel              = 1;
param.twoLevelType          = 'rightHalf'; %'nearXMinus'; %'centralHalf'; %
param.threeLevel            = 0;
param.threeLevelType        = 'leftHalf';

% Modules activation flags
param.setupGrid             = 1;
param.solveSystem           = 1;
param.plotResults           = 0;
param.saveResults           = 1;
param.verboseLevel          = 1;

% For testDisc: experiments are parameterized by the resolution.
param.numCellsRange         = 4;%2.^[2:1:5];
