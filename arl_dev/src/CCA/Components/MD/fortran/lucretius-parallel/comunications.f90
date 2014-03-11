module comunications
 
 interface COMM_bcast
   module procedure bcast_integer_scalar
   module procedure bcast_integer_array_1D
   module procedure bcast_integer_array_2D
   module procedure bcast_integer_array_3D
   module procedure bcast_real_scalar
   module procedure bcast_real_array_1D
   module procedure bcast_real_array_2D
   module procedure bcast_real_array_3D
   module procedure bcast_logical_scalar
   module procedure bcast_logical_array_1D
   module procedure bcast_logical_array_2D
   module procedure bcast_logical_array_3D
 end interface COMM_bcast

 contains

  subroutine bcast_integer_scalar(i)
  integer , intent(IN) :: i
  integer ierr
!call MPI_BCAST(V,i,MPI_INTEGER,0,MPI_Comm_world,ierr)
  end subroutine bcast_integer_scalar

  subroutine bcast_integer_array_1D(V)
  integer, intent(IN) :: V(:)
  integer ierr,i0,i1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
!call MPI_BCAST(V,i1-i0+1,MPI_INTEGER,0,MPI_Comm_world,ierr) 
  end subroutine bcast_integer_array_1D

  subroutine bcast_integer_array_2D(V)
  integer, intent(IN) :: V(:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    N = (i1-i0+1)*(j1-j0+1)
!call MPI_BCAST(V,N,MPI_INTEGER,0,MPI_Comm_world,ierr)
  end subroutine bcast_integer_array_2D

  subroutine bcast_integer_array_3D(V)
  integer, intent(IN) :: V(:,:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
    N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
!call MPI_BCAST(V,N,MPI_INTEGER,0,MPI_Comm_world,ierr)
  end subroutine bcast_integer_array_3D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bcast_real_scalar(i)
  real(8) , intent(IN) :: i
  integer ierr
!call MPI_BCAST(V,i,MPI_REAL8,0,MPI_Comm_world,ierr)
  end subroutine bcast_real_scalar

  subroutine bcast_real_array_1D(V)
  real(8), intent(IN) :: V(:)
  integer ierr,i0,i1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
!call MPI_BCAST(V,i1-i0+1,MPI_REAL8,0,MPI_Comm_world,ierr)
  end subroutine bcast_real_array_1D

  subroutine bcast_real_array_2D(V)
  real(8), intent(IN) :: V(:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    N = (i1-i0+1)*(j1-j0+1)
!call MPI_BCAST(V,N,MPI_REAL8,0,MPI_Comm_world,ierr)
  end subroutine bcast_real_array_2D

  subroutine bcast_real_array_3D(V)
  real(8), intent(IN) :: V(:,:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
    N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
!call MPI_BCAST(V,N,MPI_REAL8,0,MPI_Comm_world,ierr)
  end subroutine bcast_real_array_3D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bcast_logical_scalar(i)
  logical , intent(IN) :: i
  integer ierr
!call MPI_BCAST(V,i,MPI_LOGICAL,0,MPI_Comm_world,ierr)
  end subroutine bcast_logical_scalar

  subroutine bcast_logical_array_1D(V)
  logical, intent(IN) :: V(:)
  integer ierr,i0,i1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
!call MPI_BCAST(V,i1-i0+1,MPI_LOGICAL,0,MPI_Comm_world,ierr)
  end subroutine bcast_logical_array_1D

  subroutine bcast_logical_array_2D(V)
  logical, intent(IN) :: V(:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    N = (i1-i0+1)*(j1-j0+1)
!call MPI_BCAST(V,N,MPI_LOGICAL,0,MPI_Comm_world,ierr)
  end subroutine bcast_logical_array_2D

  subroutine bcast_logical_array_3D(V)
  logical, intent(IN) :: V(:,:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
    N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
!call MPI_BCAST(V,N,MPI_LOGICAL,0,MPI_Comm_world,ierr)
  end subroutine bcast_logical_array_3D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end module comunications


