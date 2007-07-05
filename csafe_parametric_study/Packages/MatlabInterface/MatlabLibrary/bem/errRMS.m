function rms= errRMS(Umeas,Upred)
% FUNCTION rms = errRMS(Umeas,Upred)
%
% DESCRIPTION
% This fuunction computes the RMS error measurement bewteen two vectors
%
% INPUT
% Umeas    the measured vector
% Upred    the vector predicted by modelling
%
% OUTPUT
% rms      the rms number
%
% SEE ALSO errMAG errRDM errRDMS

rms = sqrt(mean((Upred-Umeas).^2));