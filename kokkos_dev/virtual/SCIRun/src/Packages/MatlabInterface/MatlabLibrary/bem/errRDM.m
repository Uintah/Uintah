function rdm = errRDM(Umeas,Upred)
% FUNCTION rdm = errRDM(Umeas,Upred)
%
% DESCRIPTION
% This fuunction computes thee RDM error measurement bewteen two vectors
%
% INPUT
% Umeas    the measured vector
% Upred    the vector predicted by modelling
%
% OUTPUT
% rdm      the rdm number
%
% SEE ALSO errRMS errRDMS errMAG

rdm = sqrt( sum((Upred-Umeas).^2)./sum(Umeas.^2));
