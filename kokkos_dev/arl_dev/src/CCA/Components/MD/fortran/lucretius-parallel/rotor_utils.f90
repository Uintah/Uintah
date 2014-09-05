module rotor_utils
implicit none
private :: orient
public :: get_quaternions
public :: get_mol_orient_atom_xyz
public :: atom_in_lab_frame
public :: torques_in_body_frame
public :: V_in_body_frame
public :: renormalize_quaternions

contains

subroutine get_quaternions
use ALL_atoms_data, only : xxx,yyy,zzz,all_atoms_mass,Natoms,atom_in_which_molecule
use ALL_mols_data, only : start_group,end_group,Nmols, N_atoms_per_mol,i_type_molecule,&
                          mol_xyz
use ALL_rigid_mols_data, only :  qn
use mol_type_data, only : mol_type_xyz0
use random_generator_module, only : ranf
use minimize
implicit none
integer i,j,k,iter,N,kkk, N_for_optim,j1
real(8) t(3),sm,q(4),ren,var,X(4), ERROR,ERROR1,aaa,xyz_cm(3)
real(8), allocatable :: xyz_body_0(:,:), qn_old(:,:), FF(:,:),Xnew(:,:),dx(:),dx_old(:),qn_saved(:,:)
real(8),save :: dummy = 0.8d0
real(8), allocatable :: d(:,:) , r(:,:), d1(:,:),r1(:,:), Ad(:,:),Ax(:,:)
real(8), allocatable :: work3(:,:),work3_0(:,:)
logical l_got_this_one,failure

kkk=maxval(N_atoms_per_mol)
allocate(xyz_body_0(Natoms,3),qn_old(Nmols,4),FF(Natoms,3),qn_saved(Nmols,4))
allocate(Xnew(maxval(N_atoms_per_mol),3))
allocate(d(kkk,3),d1(kkk,3),r(kkk,3),r1(kkk,3),Ad(kkk,3),Ax(kkk,3))
allocate(work3(kkk,3),work3_0(kkk,3))
allocate(dx(Nmols),dx_old(Nmols))

do i = 1, Nmols
 t = 0.0d0
 sm  = 0.0d0
 do j = start_group(i),end_group(i)
   t(1) = t(1) + all_atoms_mass(j)*xxx(j)
   t(2) = t(2) + all_atoms_mass(j)*yyy(j)
   t(3) = t(3) + all_atoms_mass(j)*zzz(j)
   sm = sm + all_atoms_mass(j)
 enddo
 mol_xyz(i,:) = t(:)/sm  ! mass centra of molecules 
enddo
 xyz_body_0(:,1) = xxx(:) - mol_xyz(atom_in_which_molecule(:),1)
 xyz_body_0(:,2) = yyy(:) - mol_xyz(atom_in_which_molecule(:),2)
 xyz_body_0(:,3) = zzz(:) - mol_xyz(atom_in_which_molecule(:),3)

 do i = 1, Nmols
   qn(i,:) = (/ 0.0d0,0.0d0,0.0d0,1.0d0 /)  ! initial guess
 enddo

 dx = 1.0d90
 do iter = 1, 100   ! 1000 randoms attomepts to find something close
   do i = 1, Nmols
     N=N_atoms_per_mol(i)
     if (N > 1) then
     do k = 1, 4 ;   qn_old(i,k) = ranf(dummy) ;  enddo
     ren = dsqrt(dot_product(qn_old(i,:),qn_old(i,:)))
     do k = 1, 4 ;   qn_old(i,k) = qn_old(i,k)/ren ;  enddo  ! renormalized them
     call orient(N, mol_type_xyz0(i_type_molecule(i),1:N,1:3), qn_old(i,:),Xnew(1:N,1:3))
     dx_old(i) = sum((Xnew(1:N,1:3)-mol_type_xyz0(i_type_molecule(i),1:N,1:3))**2)
     if ( dx_old(i) < dx(i) ) then
        qn(i,:) = qn_old(i,:)
        dx(i)=dx_old(i)
     endif
   endif ! N > 1 
   enddo
 enddo

qn_saved = qn 

 do iter = 1, 100   ! 1000 randoms attomepts to find something close
   do i = 1, Nmols
     N=N_atoms_per_mol(i)
     if (N > 1) then
     do k = 1, 4 ;   var = ranf(dummy) ;  qn_old(i,k) = qn_saved(i,k) + 0.1d0*var ; enddo
     ren = dsqrt(dot_product(qn_old(i,:),qn_old(i,:)))
     do k = 1, 4 ;   qn_old(i,k) = qn_old(i,k)/ren ;  enddo  ! renormalized them
     call orient(N, mol_type_xyz0(i_type_molecule(i),1:N,1:3), qn_old(i,:),Xnew(1:N,1:3))
     dx_old(i) = sum((Xnew(1:N,1:3)-mol_type_xyz0(i_type_molecule(i),1:N,1:3))**2)
     if ( dx_old(i) < dx(i) ) then
        qn(i,:) = qn_old(i,:)
        dx(i)=dx_old(i)
     endif
   endif ! N > 1
   enddo
 enddo

 xyz_cm = (/ 0.0d0,0.0d0,0.d0 /)
 do i = 1, Nmols
     N_for_optim=N_atoms_per_mol(i)
     if (N_for_optim .gt. 1) then
       work3(1:N_for_optim,1:3) = xyz_body_0(start_group(i):end_group(i),1:3)
       work3_0(1:N_for_optim,1:3) = mol_type_xyz0(i_type_molecule(i),1:N_for_optim,1:3)
!       do j = 1, N_for_optim-1
!       do j1 = 1+j,N_for_optim
!       print*,j,j1,dsqrt((work3(j,1)-work3(j1,1))**2+(work3(j,2)-work3(j1,2))**2+(work3(j,3)-work3(j1,3))**2),&
!         dsqrt((work3_0(j,1)-work3_0(j1,1))**2+(work3_0(j,2)-work3_0(j1,2))**2+(work3_0(j,3)-work3_0(j1,3))**2)
!       enddo
!       enddo

       l_got_this_one=.false.
       iter=0
       do while (.not.l_got_this_one)
          iter=iter+1
          call opt(work3_0(1:N_for_optim,1:3),&
            work3(1:N_for_optim,1:3),N_for_optim,4,xyz_cm,qn(i,:),aaa)
            l_got_this_one=dabs(aaa).lt.1.0d-10
            failure = iter > 10000
            if (failure) then
               print*, 'ERROR: in rotor_utils%get_quaternions; NO convergence',aaa
               STOP
            endif 
       enddo !while
       ren = dsqrt(dot_product(qn(i,:),qn(i,:)))
       qn(i,:) = qn(i,:)/ren
     endif
 print*, i,'QN found iter precision=',iter,aaa
 enddo

deallocate(xyz_body_0,qn_old,FF,qn_saved)
deallocate(Xnew)
deallocate(dx,dx_old)
deallocate(work3,work3_0)
deallocate(d,d1,r,r1,Ad,Ax)

print*, 'QUATERNIONS WERE SUCCESFULLY GENERATED; THE SIMULATION WILL NOW PROCEED'
end subroutine get_quaternions

subroutine orient(N,X0,qn, X)
integer, intent(IN) ::N
real(8),intent(IN) :: X0(N,3), qn(4)
real(8), intent(out) :: X(N,3)
real(8) o(9)
integer i
real(8) dr(3)

   o(1)=-qn(1)**2+qn(2)**2-qn(3)**2+qn(4)**2
   o(2)=-2.0d0*(qn(1)*qn(2)+qn(3)*qn(4))
   o(3)=2.0d0*(qn(2)*qn(3)-qn(1)*qn(4))
   o(4)=2.0d0*(qn(3)*qn(4)-qn(1)*qn(2))
   o(5)=qn(1)**2-qn(2)**2-qn(3)**2+qn(4)**2
   o(6)=-2.0d0*(qn(1)*qn(3)+qn(2)*qn(4))
   o(7)=2.0d0*(qn(2)*qn(3)+qn(1)*qn(4))
   o(8)=2.0d0*(qn(2)*qn(4)-qn(1)*qn(3))
   o(9)=-qn(1)**2-qn(2)**2+qn(3)**2+qn(4)**2

do i = 1, N
   dr(:) = x0(i,:)
   x(i,1)= o(1)*dr(1) + o(2)*dr(2) + o(3)*dr(3)
   x(i,2)= o(4)*dr(1) + o(5)*dr(2) + o(6)*dr(3)
   x(i,3)= o(7)*dr(1) + o(8)*dr(2) + o(9)*dr(3)
enddo
end subroutine orient

SUBROUTINE get_mol_orient_atom_xyz
use ALL_rigid_mols_data, only : mol_orient, qn, xyz_body
use mol_type_data, only : N_mols_of_type,N_type_atoms_per_mol_type,mol_type_xyz0,N_type_molecules
 implicit none
 integer i,j,k,i1,j1,k1,i2,i_sum,iii
 real(8) dr(3)
   mol_orient(:,1)=-qn(:,1)**2+qn(:,2)**2-qn(:,3)**2+qn(:,4)**2
   mol_orient(:,2)=-2.0d0*(qn(:,1)*qn(:,2)+qn(:,3)*qn(:,4))
   mol_orient(:,3)=2.0d0*(qn(:,2)*qn(:,3)-qn(:,1)*qn(:,4))
   mol_orient(:,4)=2.0d0*(qn(:,3)*qn(:,4)-qn(:,1)*qn(:,2))
   mol_orient(:,5)=qn(:,1)**2-qn(:,2)**2-qn(:,3)**2+qn(:,4)**2
   mol_orient(:,6)=-2.0d0*(qn(:,1)*qn(:,3)+qn(:,2)*qn(:,4))
   mol_orient(:,7)=2.0d0*(qn(:,2)*qn(:,3)+qn(:,1)*qn(:,4))
   mol_orient(:,8)=2.0d0*(qn(:,2)*qn(:,4)-qn(:,1)*qn(:,3))
   mol_orient(:,9)=-qn(:,1)**2-qn(:,2)**2+qn(:,3)**2+qn(:,4)**2

 i1=0
  i2=0
  i_sum=0
  do i=1,N_type_molecules
   do iii=1,N_mols_of_type(i)
      i1=i1+1
      do j=1,N_type_atoms_per_mol_type(i)
        i2=i2+1
        dr(:)= mol_type_xyz0(i,j,:)
        xyz_body(i2,1)= &
               mol_orient(i1,1)*dr(1) + mol_orient(i1,2)*dr(2) + mol_orient(i1,3)*dr(3)
        xyz_body(i2,2)= &
               mol_orient(i1,4)*dr(1) + mol_orient(i1,5)*dr(2) + mol_orient(i1,6)*dr(3)
        xyz_body(i2,3)= &
               mol_orient(i1,7)*dr(1) + mol_orient(i1,8)*dr(2) + mol_orient(i1,9)*dr(3)
      enddo
   enddo
   enddo
END SUBROUTINE get_mol_orient_atom_xyz

subroutine atom_in_lab_frame
use ALL_mols_data, only : mol_xyz
use ALL_rigid_mols_data, only :  xyz_body
use ALL_atoms_data, only : xxx,yyy,zzz, atom_in_which_molecule,Natoms
implicit none
integer i
do i = 1, Natoms
 xxx(i) = xyz_body(i,1) + mol_xyz(atom_in_which_molecule(i),1)
 yyy(i) = xyz_body(i,2) + mol_xyz(atom_in_which_molecule(i),2)
 zzz(i) = xyz_body(i,3) + mol_xyz(atom_in_which_molecule(i),3)
enddo
end subroutine atom_in_lab_frame

subroutine  torques_in_body_frame
use ALL_rigid_mols_data, only : mol_orient, mol_torque
use sizes_data, only : Nmols
implicit none
integer i
real(8) t(3) 
 do i=1,Nmols
  t(:)=mol_torque(i,:)
  mol_torque(i,1)=mol_orient(i,1)*t(1)+mol_orient(i,4)*t(2)+mol_orient(i,7)*t(3)
  mol_torque(i,2)=mol_orient(i,2)*t(1)+mol_orient(i,5)*t(2)+mol_orient(i,8)*t(3)
  mol_torque(i,3)=mol_orient(i,3)*t(1)+mol_orient(i,6)*t(2)+mol_orient(i,9)*t(3)  
 enddo  
end subroutine  torques_in_body_frame

 subroutine V_in_body_frame(N,V,V1)
 use ALL_rigid_mols_data, only : mol_orient
 integer, intent(IN) :: N
  real(8),intent(IN) ::  V(N,3)
  real(8),intent(OUT) ::  V1(N,3)
  real(8) t(3)
  integer i
  do i = 1,N
    t(:) = V(i,:)
    V1(i,1)=mol_orient(i,1)*t(1)+mol_orient(i,4)*t(2)+mol_orient(i,7)*t(3)
    V1(i,2)=mol_orient(i,2)*t(1)+mol_orient(i,5)*t(2)+mol_orient(i,8)*t(3)
    V1(i,3)=mol_orient(i,3)*t(1)+mol_orient(i,6)*t(2)+mol_orient(i,9)*t(3) 
  enddo
  end subroutine V_in_body_frame

subroutine renormalize_quaternions
use ALL_rigid_mols_data, only : qn
use sizes_data, only : Nmols
 implicit none
 integer i
 real(8) t
  do i = 1,Nmols
    t = qn(i,1)**2 + qn(i,2)**2 + qn(i,3)**2 + qn(i,4)**2
    qn(i,:) = qn(i,:) / dsqrt(t)
  enddo
end subroutine renormalize_quaternions

end module rotor_utils
