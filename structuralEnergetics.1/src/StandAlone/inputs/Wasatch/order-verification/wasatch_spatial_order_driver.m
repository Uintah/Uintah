%clear all
%clc
wasatch_calculate_spatial_order_of_accuracy('xmom','XVOL','SVOL','{\it x}-momentum','wasatch_spatial_order_xmom',1);
wasatch_calculate_spatial_order_of_accuracy('ymom','SVOL','YVOL','{\it y}-momentum','wasatch_spatial_order_ymom',1);

wasatch_calculate_spatial_order_of_accuracy('tauxx','SVOL','SVOL','\tau_{\it xx}','wasatch_spatial_order_tauxx');
wasatch_calculate_spatial_order_of_accuracy('tauyy','SVOL','SVOL','\tau_{\it yy}','wasatch_spatial_order_tauyy');

wasatch_calculate_spatial_order_of_accuracy('xconvx','SVOL','SVOL','xconvx','wasatch_spatial_order_xconvx');
wasatch_calculate_spatial_order_of_accuracy('yconvy','SVOL','SVOL','yconvy','wasatch_spatial_order_yconvy');
wasatch_calculate_spatial_order_of_accuracy('xconvy','XVOL','YVOL','xconvy','wasatch_spatial_order_xconvy'); 
wasatch_calculate_spatial_order_of_accuracy('yconvx','XVOL','YVOL','yconvx','wasatch_spatial_order_yconvx');

wasatch_calculate_spatial_order_of_accuracy('xmomrhspart','XVOL','SVOL','{\it x}-mom rhs partial','wasatch_spatial_order_xmom_part');
wasatch_calculate_spatial_order_of_accuracy('ymomrhspart','SVOL','YVOL','{\it y}-mom rhs partial','wasatch_spatial_order_ymom_part');

wasatch_calculate_spatial_order_of_accuracy('xmomrhsfull','XVOL','SVOL','{\it x}-mom rhs full','wasatch_spatial_order_xmom_full',1);
wasatch_calculate_spatial_order_of_accuracy('ymomrhsfull','SVOL','YVOL','{\it y}-mom rhs full','wasatch_spatial_order_ymom_full',1);

clear all;