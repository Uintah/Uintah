% function: plot_nyquist_limit
% author: Tony Saad
% date: Sept 2012
%
% nyquist_limit: plots a line designating the nyquist limit

function plot_nyquist_limit(k_nyquist)
  loglog( [k_nyquist,k_nyquist],[1e-10,1e-2],'k--')
end

