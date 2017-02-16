% function: plot_energy_decay_all
% author:   Tony Saad
% date:     June, 2013
%
% plot_energy_decay_all: plots all data for the energy spectrum
% 
% calculation
% usage: plot_energy_decay_all('ke_wasatch_wale_32','32_wasatch_wale','Wale model, C_w = 0.7')
%
% 1. Run your turbulent flow simulation
% 2. Use lineextract to save your u, v, and w velocities at t = 0.28s and t
% = 0.66s as these correspond to the experimental datapoints
% 3. With lineextract, save your output as
% uvel/vvel/wvel_basename_t0.28s/t0.66s.txt
% for example: vvel_32_wasatch_csmag_t0.28s.txt is the lineextract for the
% y-velocity field. here, the basename is 32_wasatch_csmag.
% 4. Execute energy_spectrum_plot_all. Here, the output_filename designates
% the pdf filename of the exported figure, base_name denotes the
% base_name in the lineextract output filename, and figure_title
% is a string that corresponds to the figure's title. For example,
% energy_spectrum_plot_all('ke_wasatch_wale_32','32_wasatch_wale','Wale model, C_w = 0.7')
% will look for lineextract files with names:
% uvel_32_wasatch_wale_t0.28s.txt
% vvel_32_wasatch_wale_t0.28s.txt
% wvel_32_wasatch_wale_t0.28s.txt
% uvel_32_wasatch_wale_t0.66s.txt
% vvel_32_wasatch_wale_t0.66s.txt
% wvel_32_wasatch_wale_t0.66s.txt
% The output pdf figure will have the title ke_wasatch_wale_32.pdf and will
% have the title: Wale model, C_w = 0.7
%

function [] = plot_energy_decay_all(output_filename, modelname, N, figure_title)

  close all   % cleanup  
  plot_style  % set some plot style parameters

  % set figure properties
  set(gca,'Units',Plot_Units)
  set(gca,'Position',[Plot_X,Plot_Y,Plot_Width,Plot_Height])
  set(gcf,'DefaultLineLineWidth',Line_Width)

  % format axes
  set(gca,'FontName',Font_Name)
  set(gca,'FontSize',Key_Font_Size)
  %axis([0 0.7 0 0.1])
  ylabel('{\it E}, m^2/s^2','FontSize',Title_Font_Size)
  xlabel('{\it t}, s','FontSize',Title_Font_Size)
  title(figure_title);

  % plot the decay for the inviscid flow
  inviscid_fname = strcat('KEDecay-inviscid_',num2str(N), '.dat');
  plot_energy_decay_uda(inviscid_fname, N, 'r--');

  % plot the decay for the inviscid flow
  viscous_fname = strcat('KEDecay-viscous_',num2str(N), '.dat');
  plot_energy_decay_uda(viscous_fname, N, 'k-.');

  % plot the decay for the turbulence model
  filename=strcat('KEDecay-',modelname,'_',num2str(N), '.dat');
  plot_energy_decay_uda(filename, N, 'k-');
    
  %plot the cbc energy decay
  plot_energy_decay_cbc(N);

  legend('Inviscid','Viscous',modelname,'CBC data','Location','East')

  % print to pdf
  set(gcf,'Visible',Figure_Visibility);
  set(gcf,'PaperUnits',Paper_Units);
  set(gcf,'PaperSize',[Paper_Width (Paper_Height)]);
  set(gcf,'PaperPosition',[0.0 0.0 Paper_Width (Paper_Height)]);
  print(gcf,'-dpdf',['',output_filename]);

end