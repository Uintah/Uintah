function rdms= errRDMS(Umeas,Upred)
% FUNCTION rdms = errRDMS(Umeas,Upred)
%
% DESCRIPTION
% This fuunction computes the RDM* error measurement bewteen two vectors
%
% INPUT
% Umeas    the measured vector
% Upred    the vector predicted by modelling
%
% OUTPUT
% rdm      the rdm number
%
% SEE ALSO errRMS errRDM errMAG


rdms = sqrt( sum(((Upred./(ones(size(Upred,1),1)*(sum(Upred.^2))))-(Umeas./(ones(size(Umeas,1),1)*(sum(Umeas.^2))))).^2) );
