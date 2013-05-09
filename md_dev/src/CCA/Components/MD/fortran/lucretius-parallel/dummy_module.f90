
 module dummy_module
 implicit none

 public :: Do_Dummy_Coords
 public :: Do_Dummy_Forces

 contains

 
  Subroutine Do_Dummy_Coords
  use ALL_dummies_data
  use ALL_atoms_data, only : Natoms, xxx,yyy,zzz,is_dummy
  use boundaries, only : periodic_images
  implicit none
     real(8), allocatable :: x_12(:),y_12(:),z_12(:),dr_sq_12(:)
     real(8), allocatable :: x_13(:),y_13(:),z_13(:),dr_sq_13(:)
     real(8), allocatable :: x_23(:),y_23(:),z_23(:),dr_sq_23(:), dr_23(:)
     real(8), allocatable :: x12_a_x23(:),y12_a_y23(:),z12_a_z23(:),dr_sq_12_a_23(:)
     real(8), allocatable :: VV(:),  R12_dot_R13(:)
     real(8) aa,bb,cc, factor, cross_xx,cross_yy,cross_zz,rid_xx,rid_yy,rid_zz,i_vi
     integer i,j,k,ii,jj,kk, i1, iStyle, id
     
     allocate(x_12(Ndummies),y_12(Ndummies),z_12(Ndummies),dr_sq_12(Ndummies))
     allocate(x_13(Ndummies),y_13(Ndummies),z_13(Ndummies),dr_sq_13(Ndummies))
     allocate(x_23(Ndummies),y_23(Ndummies),z_23(Ndummies),dr_sq_23(Ndummies),dr_23(Ndummies))
     allocate(x12_a_x23(Ndummies),y12_a_y23(Ndummies),z12_a_z23(Ndummies),dr_sq_12_a_23(Ndummies))
     allocate(VV(Ndummies),R12_dot_R13(Ndummies))
 
 
     i1 = 0
     do i=1,Ndummies
       i1 = i1 + 1
       ii = all_dummy_connect_info(i,2)
       jj = all_dummy_connect_info(i,1)
       kk = all_dummy_connect_info(i,3)
       x_12(i1) = xxx(jj)-xxx(ii) ; y_12(i1) = yyy(jj)-yyy(ii) ; z_12(i1) = zzz(jj)-zzz(ii)
       x_13(i1) = xxx(kk)-xxx(ii) ; y_13(i1) = yyy(kk)-yyy(ii) ; z_13(i1) = zzz(kk)-zzz(ii)
       x_23(i1) = xxx(kk)-xxx(jj) ; y_23(i1) = yyy(kk)-yyy(jj) ; z_23(i1) = zzz(kk)-zzz(jj)
     enddo
     call periodic_images(x_12(1:i1),y_12(1:i1),z_12(1:i1))
     call periodic_images(x_13(1:i1),y_13(1:i1),z_13(1:i1))
     call periodic_images(x_23(1:i1),y_23(1:i1),z_23(1:i1))
     dr_sq_12(1:i1)=x_12(1:i1)*x_12(1:i1)+y_12(1:i1)*y_12(1:i1)+z_12(1:i1)*z_12(1:i1)
     dr_sq_13(1:i1)=x_13(1:i1)*x_13(1:i1)+y_13(1:i1)*y_13(1:i1)+z_13(1:i1)*z_13(1:i1)
     dr_sq_23(1:i1)=x_23(1:i1)*x_23(1:i1)+y_23(1:i1)*y_23(1:i1)+z_23(1:i1)*z_23(1:i1)
     dr_23(1:i1)   = dsqrt(dr_sq_23(1:i1))
     x12_a_x23(1:i1) = x_12(1:i1) + all_dummy_params(i1,1) * x_23(1:i1) !rij(kk)+aa*rjk(kk)
     y12_a_y23(1:i1) = y_12(1:i1) + all_dummy_params(i1,1) * y_23(1:i1)
     z12_a_z23(1:i1) = z_12(1:i1) + all_dummy_params(i1,1) * z_23(1:i1) 
     dr_sq_12_a_23(1:i1) = x12_a_x23(1:i1)*x12_a_x23(1:i1)+y12_a_y23(1:i1)*y12_a_y23(1:i1)+z12_a_z23(1:i1)*z12_a_z23(1:i1)
     
     R12_dot_R13(1:i1) = x_12(1:i1)*x_13(1:i1) + y_12(1:i1)*y_13(1:i1) + z_12(1:i1)*z_13(1:i1)
     VV(1:i1) = dsqrt(dr_sq_12(1:i1)*dr_sq_13(1:i1) - R12_dot_R13(1:i1)*R12_dot_R13(1:i1))

     i1 = 0
     do i = 1, Ndummies
       i1 = i1 + 1
       ii = all_dummy_connect_info(i,2)
       jj = all_dummy_connect_info(i,1)
       kk = all_dummy_connect_info(i,3)
       id = map_dummy_to_atom(i)
       iStyle = i_Style_dummy(i)
        bb = all_dummy_params(i,2)
        cc = all_dummy_params(i,3)

        factor=-bb/dsqrt(dr_sq_12_a_23(i1))
        rid_xx = factor * x12_a_x23(i1)  ;  
        rid_yy = factor * y12_a_y23(i1)  ; 
        rid_zz = factor * z12_a_z23(i1)  
        select case (iStyle)
         case (1,2)
        cross_xx= y_12(i)*z_13(i1) - y_13(i1)*z_12(i1)  
          cross_yy=-x_12(i)*z_13(i1) + x_13(i1)*z_12(i1) 
          cross_zz= x_12(i)*y_13(i1) - x_13(i1)*y_12(i1)
          i_vi = 1.0d0/VV(i1)
          rid_xx=rid_xx+(cc*i_vi)*cross_xx
          rid_yy=rid_yy+(cc*i_vi)*cross_yy
          rid_zz=rid_zz+(cc*i_vi)*cross_zz
          xxx(id) = xxx(ii) + rid_xx
          yyy(id) = yyy(ii) + rid_yy
          zzz(id) = zzz(ii) + rid_zz
         case (3)
          i_vi = 1.0d0/dr_23(i)
          rid_xx=rid_xx+(cc*i_vi)*x_23(i1)
          rid_yy=rid_yy+(cc*i_vi)*y_23(i1)
          rid_zz=rid_zz+(cc*i_vi)*z_23(i1)
          xxx(id) = xxx(ii) + rid_xx
          yyy(id) = yyy(ii) + rid_yy
          zzz(id) = zzz(ii) + rid_zz          
         case default
            print*, 'ERROR: in DoDommyCoords iStyle not implemented ', iStyle
            STOP
          end select
     enddo


!open(unit=14,file='fort.14',recl=1000)
!write(14,*) Natoms
!do i = 1, Natoms
!write(14,*) i,xxx(i),yyy(i),zzz(i),is_dummy(i)
!enddo
!close(14)
     
     deallocate(x_12,y_12,z_12,dr_sq_12)
     deallocate(x_13,y_13,z_13,dr_sq_13)
     deallocate(x_23,y_23,z_23,dr_sq_23,dr_23)
     deallocate(x12_a_x23,y12_a_y23,z12_a_z23,dr_sq_12_a_23)
     deallocate(VV,R12_dot_R13)
  
     
   end subroutine Do_Dummy_Coords
      
!----------------------------------------------------------------------
!----------------------------------------------------------------------
!----------------------------------------------------------------------  

 Subroutine Do_Dummy_Forces(stress_update,fdummy_xx,fdummy_yy,fdummy_zz)
  use ALL_dummies_data
  use ALL_atoms_data, only : Natoms, xxx,yyy,zzz,is_dummy
  use boundaries, only : periodic_images
  use stresses_data, only : stress_dummy,stress
  implicit none
     logical, intent(IN):: stress_update
     real(8), intent(INOUT) :: fdummy_xx(:),fdummy_yy(:),fdummy_zz(:)
     real(8), allocatable :: x_12(:),y_12(:),z_12(:),dr_sq_12(:)
     real(8), allocatable :: x_13(:),y_13(:),z_13(:),dr_sq_13(:)
     real(8), allocatable :: x_23(:),y_23(:),z_23(:),dr_sq_23(:), dr_23(:)
     real(8), allocatable :: x12_a_x23(:),y12_a_y23(:),z12_a_z23(:),dr_sq_12_a_23(:)
     real(8), allocatable :: VV(:),  R12_dot_R13(:), fct(:),Fd_xx(:),Fd_yy(:),Fd_zz(:)
     real(8), allocatable :: crs_xx(:), crs_yy(:), crs_zz(:),f1_xx(:), f1_yy(:),f1_zz(:)
     real(8), allocatable :: ri_xx(:),ri_yy(:),ri_zz(:),rj_xx(:),rj_yy(:),rj_zz(:),rk_xx(:),rk_yy(:),rk_zz(:)
     real(8) aa,bb,cc, factor, cross_xx,cross_yy,cross_zz
     real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,lstr_xx,lstr_xy,lstr_xz,lstr_yx,lstr_yy,lstr_yz,lstr_zx,lstr_zy,lstr_zz
     real(8) Fd_dot_rid, C1,C2,C3,cross_dot_Fd, Fd_dot_r23
     real(8) rid_xx,rjd_xx,rkd_xx,rid_yy,rjd_yy,rkd_yy,rid_zz,rjd_zz,rkd_zz
     real(8) fiout_xx,fiout_yy,fiout_zz,fjout_xx,fjout_yy,fjout_zz,fkout_xx,fkout_yy,fkout_zz
     real(8) x12,y12,z12,x13,y13,z13,x23,y23,z23,  r12_2, r13_2, i_vi,i_vi3, i_vi2, Fdx,Fdy,Fdz
     real(8) tmp1_xx,tmp1_yy,tmp1_zz,tmp2_xx,tmp2_yy,tmp2_zz,tmp3_xx,tmp3_yy,tmp3_zz,ftmp_xx,ftmp_yy,ftmp_zz
     integer i,j,k,ii,jj,kk, i1, iStyle, id

     allocate(x_12(Ndummies),y_12(Ndummies),z_12(Ndummies),dr_sq_12(Ndummies))
     allocate(x_13(Ndummies),y_13(Ndummies),z_13(Ndummies),dr_sq_13(Ndummies))
     allocate(x_23(Ndummies),y_23(Ndummies),z_23(Ndummies),dr_sq_23(Ndummies),dr_23(Ndummies))
     allocate(x12_a_x23(Ndummies),y12_a_y23(Ndummies),z12_a_z23(Ndummies),dr_sq_12_a_23(Ndummies))
     allocate(VV(Ndummies),R12_dot_R13(Ndummies),fct(Ndummies),Fd_xx(Ndummies),Fd_yy(Ndummies),Fd_zz(Ndummies))
     allocate(ri_xx(Ndummies),ri_yy(Ndummies),ri_zz(Ndummies),&
              rj_xx(Ndummies),rj_yy(Ndummies),rj_zz(Ndummies),&
              rk_xx(Ndummies),rk_yy(Ndummies),rk_zz(Ndummies))
     allocate(crs_xx(Ndummies),crs_yy(Ndummies),crs_zz(Ndummies),f1_xx(Ndummies),f1_yy(Ndummies),f1_zz(Ndummies))

     i1 = 0
     do i=1,Ndummies
       i1 = i1 + 1
       ii = all_dummy_connect_info(i,2)
       jj = all_dummy_connect_info(i,1)
       kk = all_dummy_connect_info(i,3)
       x_12(i1) = xxx(jj)-xxx(ii) ; y_12(i1) = yyy(jj)-yyy(ii) ; z_12(i1) = zzz(jj)-zzz(ii)
       x_13(i1) = xxx(kk)-xxx(ii) ; y_13(i1) = yyy(kk)-yyy(ii) ; z_13(i1) = zzz(kk)-zzz(ii)
       x_23(i1) = xxx(kk)-xxx(jj) ; y_23(i1) = yyy(kk)-yyy(jj) ; z_23(i1) = zzz(kk)-zzz(jj)
     enddo
     call periodic_images(x_12(1:i1),y_12(1:i1),z_12(1:i1))
     call periodic_images(x_13(1:i1),y_13(1:i1),z_13(1:i1))
     call periodic_images(x_23(1:i1),y_23(1:i1),z_23(1:i1))
     dr_sq_12(1:i1)=x_12(1:i1)*x_12(1:i1)+y_12(1:i1)*y_12(1:i1)+z_12(1:i1)*z_12(1:i1)
     dr_sq_13(1:i1)=x_13(1:i1)*x_13(1:i1)+y_13(1:i1)*y_13(1:i1)+z_13(1:i1)*z_13(1:i1)
     dr_sq_23(1:i1)=x_23(1:i1)*x_23(1:i1)+y_23(1:i1)*y_23(1:i1)+z_23(1:i1)*z_23(1:i1)
     dr_23(1:i1)   = dsqrt(dr_sq_23(1:i1))
     x12_a_x23(1:i1) = x_12(1:i1) + all_dummy_params(i1,1) * x_23(1:i1) !rij(kk)+aa*rjk(kk)
     y12_a_y23(1:i1) = y_12(1:i1) + all_dummy_params(i1,1) * y_23(1:i1)
     z12_a_z23(1:i1) = z_12(1:i1) + all_dummy_params(i1,1) * z_23(1:i1)
     dr_sq_12_a_23(1:i1) = x12_a_x23(1:i1)*x12_a_x23(1:i1)+y12_a_y23(1:i1)*y12_a_y23(1:i1)+z12_a_z23(1:i1)*z12_a_z23(1:i1)

     R12_dot_R13(1:i1) = x_12(1:i1)*x_13(1:i1) + y_12(1:i1)*y_13(1:i1) + z_12(1:i1)*z_13(1:i1)
     VV(1:i1) = dsqrt(dr_sq_12(1:i1)*dr_sq_13(1:i1) - R12_dot_R13(1:i1)*R12_dot_R13(1:i1))

     i1 = 0
     do i = 1, Ndummies
       i1 = i1 + 1
       ii = all_dummy_connect_info(i,2)
       jj = all_dummy_connect_info(i,1)
       kk = all_dummy_connect_info(i,3)
       id = map_dummy_to_atom(i)
       Fd_xx(i) = fdummy_xx(id)   ; Fd_yy(i) = fdummy_yy(id)   ; Fd_zz(i) = fdummy_zz(id)
       iStyle = i_Style_dummy(i)
        bb = all_dummy_params(i,2)
        cc = all_dummy_params(i,3)
        
        factor=-bb/dsqrt(dr_sq_12_a_23(i1))
        rid_xx = factor * x12_a_x23(i1)  ;
        rid_yy = factor * y12_a_y23(i1)  ;
        rid_zz = factor * z12_a_z23(i1)
        Fd_dot_rid = Fd_xx(i)*rid_xx + Fd_yy(i)*rid_yy + Fd_zz(i)*rid_zz    
        i_vi = Fd_dot_rid/(bb*bb)
        f1_xx(i) = i_vi * rid_xx 
        f1_yy(i) = i_vi * rid_yy 
        f1_zz(i) = i_vi * rid_zz   

        
                
        select case (iStyle)
         case (1,2)
          cross_xx= y_12(i1)*z_13(i1) - y_13(i1)*z_12(i1)
          cross_yy=-x_12(i1)*z_13(i1) + x_13(i1)*z_12(i1)
          cross_zz= x_12(i1)*y_13(i1) - x_13(i1)*y_12(i1)
          i_vi = 1.0d0/VV(i1)
          rid_xx=rid_xx+(cc*i_vi)*cross_xx
          rid_yy=rid_yy+(cc*i_vi)*cross_yy
          rid_zz=rid_zz+(cc*i_vi)*cross_zz
                    crs_xx(i) =  cross_xx
                    crs_yy(i) =  cross_yy
                    crs_zz(i) =  cross_zz
         case (3)
          i_vi = 1.0d0/dr_23(i)
          rid_xx=rid_xx+(cc*i_vi)*x_23(i1)
          rid_yy=rid_yy+(cc*i_vi)*y_23(i1)
          rid_zz=rid_zz+(cc*i_vi)*z_23(i1)
          


         case default
            print*, 'ERROR: in DoDommyCoords iStyle not implemented ', iStyle
            STOP
        end select
        
          xxx(id) = xxx(ii) + rid_xx
          yyy(id) = yyy(ii) + rid_yy
          zzz(id) = zzz(ii) + rid_zz
          ri_xx(i) = rid_xx
          ri_yy(i) = rid_yy
          ri_zz(i) = rid_zz
          rk_xx(i)=xxx(id)-xxx(kk)
          rk_yy(i)=yyy(id)-yyy(kk)
          rk_zz(i)=zzz(id)-zzz(kk)
          rj_xx(i)=xxx(id)-xxx(jj)
          rj_yy(i)=yyy(id)-yyy(jj)
          rj_zz(i)=zzz(id)-zzz(jj)
          fct(i)  = factor

     enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    call periodic_images(ri_xx(1:Ndummies),ri_yy(1:Ndummies),ri_zz(1:Ndummies))
    call periodic_images(rj_xx(1:Ndummies),rj_yy(1:Ndummies),rj_zz(1:Ndummies))    
    call periodic_images(rk_xx(1:Ndummies),rk_yy(1:Ndummies),rk_zz(1:Ndummies))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (stress_update) then

      lstr_xx = 0.0d0 ; lstr_xy = 0.0d0 ; lstr_xz = 0.0d0 ; 
      lstr_yx = 0.0d0 ; lstr_yy = 0.0d0 ; lstr_yz = 0.0d0 ; 
      lstr_zx = 0.0d0 ; lstr_zy = 0.0d0 ; lstr_zz = 0.0d0 ; 

    endif    

!open(unit=14,file='fort.14',recl=1000)
!write(14,*) Natoms
  

     i1 = 0
     do i = 1, Ndummies
       i1 = i1 + 1
       ii = all_dummy_connect_info(i,2)
       jj = all_dummy_connect_info(i,1)
       kk = all_dummy_connect_info(i,3)
       id = map_dummy_to_atom(i)
       iStyle = i_Style_dummy(i)
        aa = all_dummy_params(i,1)
        bb = all_dummy_params(i,2)
        cc = all_dummy_params(i,3)
        factor=fct(i1)

        x12 = x_12(i1); y12 = y_12(i1) ; z12 = z_12(i1)
        x13 = x_13(i1); y13 = y_13(i1) ; z13 = z_13(i1)
        x23 = x_23(i1); y23 = y_23(i1) ; z23 = z_23(i1)
        
          rid_xx=ri_xx(i)
          rid_yy=ri_yy(i)
          rid_zz=ri_zz(i)
          rjd_xx=rj_xx(i)
          rjd_yy=rj_yy(i)
          rjd_zz=rj_zz(i)
          rkd_xx=rk_xx(i)
          rkd_yy=rk_yy(i)
          rkd_zz=rk_zz(i)
          fdx = Fd_xx(i) ; fdy = Fd_yy(i) ; fdz = Fd_zz(i)
        select case (iStyle)
                
         case (1,2)
          cross_xx= crs_xx(i)
          cross_yy= crs_yy(i)
          cross_zz= crs_zz(i)
          cross_dot_Fd = cross_xx*Fdx + cross_yy*Fdy + cross_zz*Fdz          
          i_vi = 1.0d0/VV(i1)


          fiout_xx=(Fdy*(z13-z12)+Fdz*(y12-y13))*i_vi
          fiout_yy=(Fdx*(z12-z13)+Fdz*(x13-x12))*i_vi
          fiout_zz=(Fdx*(y13-y12)+Fdy*(x12-x13))*i_vi
          fjout_xx=(Fdy*(-z13)+Fdz*y13)*i_vi
          fjout_yy=(Fdz*(-x13)+Fdx*z13)*i_vi
          fjout_zz=(Fdx*(-y13)+Fdy*x13)*i_vi
          fkout_xx=(Fdy*z12-Fdz*y12)*i_vi
          fkout_yy=(Fdz*x12-Fdx*z12)*i_vi
          fkout_zz=(Fdx*y12-Fdy*x12)*i_vi


          i_vi3 = i_vi*(i_vi*i_vi)
          c1 = i_vi3*cross_dot_Fd
          r13_2 = dr_sq_13(i1)
          r12_2 = dr_sq_12(i1)
          C2 = R12_dot_R13(i1)
          fiout_xx = fiout_xx + c1*(r13_2*x12 + r12_2*x13 - C2*(x13 + x12))
          fiout_yy = fiout_yy + c1*(r13_2*y12 + r12_2*y13 - C2*(y13 + y12))
          fiout_zz = fiout_zz + c1*(r13_2*z12 + r12_2*z13 - C2*(z13 + z12))
          fjout_xx = fjout_xx - c1*(r13_2*x12 -  C2*x13)    
          fjout_yy = fjout_yy - c1*(r13_2*y12 -  C2*y13)
          fjout_zz = fjout_zz - c1*(r13_2*z12 -  C2*z13)
          fkout_xx = fkout_xx - c1*(r12_2*x13 -  C2*x12)  
          fkout_yy = fkout_yy - c1*(r12_2*y13 -  C2*y12)
          fkout_zz = fkout_zz - c1*(r12_2*z13 -  C2*z12)

      
          
          ftmp_xx = factor*(fd_xx(i)-f1_xx(i))
          ftmp_yy = factor*(fd_yy(i)-f1_yy(i))
          ftmp_zz = factor*(fd_zz(i)-f1_zz(i))
          tmp1_xx = fd_xx(i) - ftmp_xx + fiout_xx*cc
          tmp1_yy = fd_yy(i) - ftmp_yy + fiout_yy*cc
          tmp1_zz = fd_zz(i) - ftmp_zz + fiout_zz*cc
          tmp2_xx = (1.0d0-aa)*ftmp_xx + fjout_xx*cc
          tmp2_yy = (1.0d0-aa)*ftmp_yy + fjout_yy*cc
          tmp2_zz = (1.0d0-aa)*ftmp_zz + fjout_zz*cc
          tmp3_xx = aa*ftmp_xx         + fkout_xx*cc
          tmp3_yy = aa*ftmp_yy         + fkout_yy*cc
          tmp3_zz = aa*ftmp_zz         + fkout_zz*cc

         case (3)

          i_vi = 1.0d0/dr_23(i)
          i_vi2 = i_vi*i_vi
          Fd_dot_r23 = x23 * Fdx + y23 * Fdy + z23 * Fdz
          ftmp_xx = factor * (Fdx - f1_xx(i)) 
          ftmp_yy = factor * (Fdy - f1_yy(i)) 
          ftmp_zz = factor * (Fdz - f1_zz(i)) 

          tmp1_xx = Fdx - ftmp_xx
          tmp1_yy = Fdy - ftmp_yy
          tmp1_zz = Fdz - ftmp_zz
          C2 = Fd_dot_r23*i_vi2
          C3 = cc*i_vi
          tmp2_xx = (1.0d0-aa)*ftmp_xx-C3*(Fdx-C2*x23)
          tmp2_yy = (1.0d0-aa)*ftmp_yy-C3*(Fdy-C2*y23)
          tmp2_zz = (1.0d0-aa)*ftmp_zz-C3*(Fdz-C2*z23)              
          tmp3_xx = ftmp_xx - tmp2_xx
          tmp3_yy = ftmp_yy - tmp2_yy
          tmp3_zz = ftmp_zz - tmp2_zz


         case default
            print*, 'ERROR: in DoDommyCoords iStyle not implemented ', iStyle
            STOP
          end select
          
          
          fdummy_xx(ii) = fdummy_xx(ii) + tmp1_xx
          fdummy_yy(ii) = fdummy_yy(ii) + tmp1_yy
          fdummy_zz(ii) = fdummy_zz(ii) + tmp1_zz
          fdummy_xx(jj) = fdummy_xx(jj) + tmp2_xx
          fdummy_yy(jj) = fdummy_yy(jj) + tmp2_yy
          fdummy_zz(jj) = fdummy_zz(jj) + tmp2_zz     
          fdummy_xx(kk) = fdummy_xx(kk) + tmp3_xx
          fdummy_yy(kk) = fdummy_yy(kk) + tmp3_yy
          fdummy_zz(kk) = fdummy_zz(kk) + tmp3_zz     
if (stress_update) then
          
          sxx = rid_xx*tmp1_xx + rjd_xx*tmp2_xx + rkd_xx*tmp3_xx
          sxy = rid_yy*tmp1_xx + rjd_yy*tmp2_xx + rkd_yy*tmp3_xx
          sxz = rid_zz*tmp1_xx + rjd_zz*tmp2_xx + rkd_zz*tmp3_xx
          
          syx = rid_xx*tmp1_yy + rjd_xx*tmp2_yy + rkd_xx*tmp3_yy
          syy = rid_yy*tmp1_yy + rjd_yy*tmp2_yy + rkd_yy*tmp3_yy
          syz = rid_zz*tmp1_yy + rjd_zz*tmp2_yy + rkd_zz*tmp3_yy
          
          szx = rid_xx*tmp1_zz + rjd_xx*tmp2_zz + rkd_xx*tmp3_zz
          szy = rid_yy*tmp1_zz + rjd_yy*tmp2_zz + rkd_yy*tmp3_zz
          szz = rid_zz*tmp1_zz + rjd_zz*tmp2_zz + rkd_zz*tmp3_zz
          
          lstr_xx = lstr_xx - sxx ; lstr_xy = lstr_xy - sxy ; lstr_xz = lstr_xz - sxz ; 
          lstr_yx = lstr_yx - syx ; lstr_yy = lstr_yy - syy ; lstr_yz = lstr_yz - syz ; 
          lstr_zx = lstr_zx - szx ; lstr_zy = lstr_zy - szy ; lstr_zz = lstr_zz - szz ; 
endif          
          fdummy_xx(id) = 0.0d0
          fdummy_yy(id) = 0.0d0
          fdummy_zz(id) = 0.0d0   
          
     enddo  ! main cycle over dummies

if (stress_update) then
    stress_dummy(1) = lstr_xx
    stress_dummy(2) = lstr_yy
    stress_dummy(3) = lstr_zz
    
    stress_dummy(4) = (lstr_xx+lstr_yy+lstr_zz)/3.0d0
    
    stress_dummy(5) = lstr_xy
    stress_dummy(6) = lstr_xz
    stress_dummy(7) = lstr_yz
    
    stress_dummy(8) = lstr_yx
    stress_dummy(9) = lstr_zx
    stress_dummy(10) = lstr_zy
    stress = stress + stress_dummy
    
    
    deallocate(x_12,y_12,z_12,dr_sq_12)
    deallocate(x_13,y_13,z_13,dr_sq_13)
    deallocate(x_23,y_23,z_23,dr_sq_23,dr_23)
    deallocate(x12_a_x23,y12_a_y23,z12_a_z23,dr_sq_12_a_23)
    deallocate(VV,R12_dot_R13,fct)
    deallocate(ri_xx,ri_yy,ri_zz,rj_xx,rj_yy,rj_zz,rk_xx,rk_yy,rk_zz)
    deallocate(crs_xx,crs_yy,crs_zz,f1_xx,f1_yy,f1_zz) 
endif
   
end subroutine Do_Dummy_Forces
    
    
    
  

 end module dummy_module
    
