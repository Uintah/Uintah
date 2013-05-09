 module variables_smpe

 use ewald_data, only : nfftx, nffty, nfftz, NFFT, &
                 order_spline_xx, order_spline_yy, order_spline_zz, &
                 h_cut_z

 implicit none

 real(8), allocatable :: tx(:),ty(:),tz(:) ! reduced coordinates
 real(8), allocatable :: qqq1_Re(:),qqq2_Re(:),qqq3_Re(:),qqq4_Re(:)
 real(8), allocatable :: qqq1_Im(:),qqq2_Im(:),qqq3_Im(:),qqq4_Im(:)

 real(8), allocatable :: spline2_REAL_dd_2_x(:,:), spline2_REAL_dd_2_y(:,:),spline2_REAL_dd_2_z(:,:) ! 2nd derivatives 
 real(8), allocatable :: spline2_REAL_dd_x(:,:), spline2_REAL_dd_y(:,:),spline2_REAL_dd_z(:,:) ! first derivatives
 real(8), allocatable :: spline2_REAL_pp_x(:,:), spline2_REAL_pp_y(:,:),spline2_REAL_pp_z(:,:) ! the splines 
 
 real(8), allocatable :: spline2_CMPLX_xx(:),spline2_CMPLX_yy(:),spline2_CMPLX_zz(:)

 real(8), allocatable :: ww1_Re(:),ww1_Im(:),ww2_Re(:),ww2_Im(:),ww3_Re(:),ww3_Im(:)
 integer, allocatable :: key1(:),key2(:),key3(:) 

 complex(8),allocatable ::  ww1(:),ww2(:),ww3(:)
 complex(8), allocatable :: qqq1(:),qqq2(:),qqq3(:),qqq4(:)

 real(8) reciprocal_zz, inv_rec_zz
 end module variables_smpe
