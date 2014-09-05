% This script is a demonstration on how to use my boundary element code

% Load the geometries

load(fullfile('examples','geometry'));

% Create a new model
model.surface{1} = tankexactgridgeom;
model.surface{2} = cagegeom;

% compute the complete transfer matrix
Transfer = bemMatrixPP(model);


% Now load some data to play around with

dataset = 3;   % change the number to view a different data set

filename = fullfile('examples',sprintf('datafile-15may2002-%04d',dataset));

load(filename);

% Compute the forward solution for each time frame

Uforward = Transfer*Ucager(cagegeom.channels,:);

Uforwardmeasonly = Uforward(tankexactgridgeom.channels,:);

% Now plot the two solutions next to each other

timeframe = 130;

figure
bemPlotSurface(tankgeom,Uforwardmeasonly(:,timeframe),'colorbar');

figure
bemPlotSurface(tankgeom,Utankr(:,timeframe),'colorbar');

% Determine some statistics

RDM = errRDM(Utankr(:,timeframe),Uforwardmeasonly(:,timeframe))
MAG = errMAG(Utankr(:,timeframe),Uforwardmeasonly(:,timeframe))
