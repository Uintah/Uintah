function mag = errMAG(Umeas,Upred)
% FUNCTION mag = errMAG(Umeas,Upred)
%
% DESCRIPTION
% This fuunction computes the MAG error measurement bewteen two vectors
%
% INPUT
% Umeas    the measured vector
% Upred    the vector predicted by modelling
%
% OUTPUT
% mag      the mag number
%
% SEE ALSO errRMS errRDM errRDMS

mag = sqrt(sum(Upred.^2)./sum(Umeas.^2));