module comunications
implicit none
public COMM_bcast
public COMM_init
public COMM_exit
public COMM_syncronize
public COMM_gsum
 
 interface COMM_bcast  ! broadcast something from master to all nodes
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

 interface COMM_gsum ! sum up something from all nodes to master and broadcast to all nodes if request broadcast (default=no broadcast)
   module procedure sum_integer_scalar
   module procedure sum_integer_array_1D
   module procedure sum_integer_array_2D
   module procedure sum_integer_array_3D
   module procedure sum_real_scalar
   module procedure sum_real_array_1D
   module procedure sum_real_array_2D
   module procedure sum_real_array_3D
   module procedure sum_logical_scalar
   module procedure sum_logical_array_1D
   module procedure sum_logical_array_2D
   module procedure sum_logical_array_3D
 end interface COMM_gsum


type mpi_type       ! data structure that contain info about paralel environment
 integer :: my_rank ! the id of CPU
 integer :: Ncpus   ! the total number of computing CPUs
 logical :: is_master ! if master node
 logical :: is_slave  ! if not master then is slave
 integer :: master_id ! the id of master node (typically 0)
end type mpi_type
type(mpi_type) COMM

 contains

 subroutine COMM_init  ! initilize the paralel environment
  use mpi
  integer ierr
  call MPI_INIT(ierr)
  call MPI_COMM_RANK( MPI_COMM_WORLD, COMM%my_rank, ierr )
  call MPI_COMM_SIZE( MPI_COMM_WORLD, COMM%Ncpus, ierr )  
!  print*, 'Initialize paralel environment in node ',COMM%my_rank, ' number of CPUS=',COMM%Ncpus
  COMM%master_id=0
  COMM%is_master=COMM%my_rank==COMM%master_id
  COMM%is_slave=.not.COMM%is_master
 end subroutine COMM_init

 subroutine COMM_exit   ! exit paralel environment
   integer ierr
   call MPI_FINALIZE(ierr)
 end subroutine COMM_exit


   subroutine COMM_syncronize   ! put a blocking barier to wait for all data transfer to finish before proceeding to next step
   use mpi
   integer ierr
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
   end subroutine COMM_syncronize

!------------------------------

  subroutine bcast_integer_scalar(i)
  use mpi
  integer , intent(IN) :: i
  integer ierr
call MPI_BCAST(i,1,MPI_INTEGER,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_integer_scalar

  subroutine bcast_integer_array_1D(V)
  use mpi
  integer, intent(IN) :: V(:)
  integer ierr,i0,i1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
call MPI_BCAST(V,i1-i0+1,MPI_INTEGER,COMM%master_id,MPI_COMM_WORLD,ierr) 
  end subroutine bcast_integer_array_1D

  subroutine bcast_integer_array_2D(V)
  use mpi
  integer, intent(IN) :: V(:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    N = (i1-i0+1)*(j1-j0+1)
call MPI_BCAST(V,N,MPI_INTEGER,COMM%master_id,MPI_COMM_WORLD,ierr)
print* 'make BCAST'
  end subroutine bcast_integer_array_2D

  subroutine bcast_integer_array_3D(V)
  use mpi
  integer, intent(IN) :: V(:,:,:)
  integer ierr,i0,j0,i1,j1,N,k0,k1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
    N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
call MPI_BCAST(V,N,MPI_INTEGER,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_integer_array_3D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bcast_real_scalar(i)
  use mpi
  real(8) , intent(IN) :: i
  integer ierr
call MPI_BCAST(i,1,MPI_REAL8,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_real_scalar

  subroutine bcast_real_array_1D(V)
  use mpi
  real(8), intent(IN) :: V(:)
  integer ierr,i0,i1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
call MPI_BCAST(V,i1-i0+1,MPI_REAL8,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_real_array_1D

  subroutine bcast_real_array_2D(V)
  use mpi
  real(8), intent(IN) :: V(:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    N = (i1-i0+1)*(j1-j0+1)
call MPI_BCAST(V,N,MPI_REAL8,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_real_array_2D

  subroutine bcast_real_array_3D(V)
  use mpi
  real(8), intent(IN) :: V(:,:,:)
  integer ierr,i0,j0,i1,j1,N,k0,k1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
    N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
call MPI_BCAST(V,N,MPI_REAL8,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_real_array_3D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine bcast_logical_scalar(i)
  use mpi
  logical , intent(IN) :: i
  integer ierr
 call MPI_BCAST(i,1,MPI_LOGICAL,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_logical_scalar

  subroutine bcast_logical_array_1D(V)
  use mpi
  logical, intent(IN) :: V(:)
  integer ierr,i0,i1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
call MPI_BCAST(V,i1-i0+1,MPI_LOGICAL,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_logical_array_1D

  subroutine bcast_logical_array_2D(V)
  use mpi
  logical, intent(IN) :: V(:,:)
  integer ierr,i0,j0,i1,j1,N
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    N = (i1-i0+1)*(j1-j0+1)
call MPI_BCAST(V,N,MPI_LOGICAL,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_logical_array_2D

  subroutine bcast_logical_array_3D(V)
  use mpi
  logical, intent(IN) :: V(:,:,:)
  integer ierr,i0,j0,i1,j1,N,k0,k1
    i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
    j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
    k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
    N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
call MPI_BCAST(V,N,MPI_LOGICAL,COMM%master_id,MPI_COMM_WORLD,ierr)
  end subroutine bcast_logical_array_3D
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


   subroutine sum_integer_scalar(i,do_bcast) 
   use mpi
   integer, intent(INOUT) :: i
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,isum
   call MPI_REDUCE(i,isum,1,MPI_INTEGER, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   i=isum
   if (present(do_bcast)) then
    if (do_bcast)  call MPI_BCAST(i,1,MPI_INTEGER, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif

   end subroutine sum_integer_scalar
  
   subroutine sum_integer_array_1D(V,do_bcast)
   use mpi
   integer, intent(INOUT) :: V(:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,j0,i1,j1,N
   integer , allocatable :: Vsum(:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   N = (i1-i0+1)
   allocate(Vsum(N))
   call MPI_REDUCE(V,Vsum,N,MPI_INTEGER, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast)  call MPI_BCAST(V,N,MPI_INTEGER, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif

   deallocate(Vsum)
   end subroutine sum_integer_array_1D

   subroutine sum_integer_array_2D(V,do_bcast)
   use mpi
   integer, intent(INOUT) :: V(:,:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,j0,i1,j1,N
   integer , allocatable :: Vsum(:,:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
   N = (i1-i0+1)*(j1-j0+1)

   allocate(Vsum(i1-i0+1,j1-j0+1))
   call MPI_REDUCE(V,Vsum,N,MPI_INTEGER, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast)  call MPI_BCAST(V,N,MPI_INTEGER, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif

   deallocate(Vsum)
   end subroutine sum_integer_array_2D

   subroutine sum_integer_array_3D(V,do_bcast)
   use mpi
   integer, intent(INOUT) :: V(:,:,:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,j0,i1,j1,N,k0,k1
   integer , allocatable :: Vsum(:,:,:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
   k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
   N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)

   allocate(Vsum(i1-i0+1,j1-j0+1,k1-k0+1))
   call MPI_REDUCE(V,Vsum,N,MPI_INTEGER, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast)  call MPI_BCAST(V,N,MPI_INTEGER, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif
   deallocate(Vsum)
   end subroutine sum_integer_array_3D


   subroutine sum_real_scalar(i,do_bcast)
   use mpi
   real(8), intent(INOUT) :: i
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr
   real(8) isum
   call MPI_REDUCE(i,isum,1,MPI_REAL8, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   i=isum
   if (present(do_bcast)) then
    if (do_bcast) call MPI_BCAST(i,1,MPI_REAL8, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif

   end subroutine sum_real_scalar

 
   subroutine sum_real_array_1D(V,do_bcast)
   use mpi
   real(8), intent(INOUT) :: V(:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,j0,i1,j1,N
   real(8) , allocatable :: Vsum(:)
print*,'IN GOOD PLACE'
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   N = (i1-i0+1)
print*,'i0 i1 N=',i0,i1,N
   allocate(Vsum(N))
   call MPI_REDUCE(V,Vsum,N,MPI_REAL8, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then 
    if (do_bcast) call MPI_BCAST(V,N,MPI_REAL8, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif
   deallocate(Vsum) 
   end subroutine sum_real_array_1D

   subroutine sum_real_array_2D(V,do_bcast)
   use mpi
   real(8), intent(INOUT) :: V(:,:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,j0,i1,j1,N
   real(8) , allocatable :: Vsum(:,:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
   N = (i1-i0+1)*(j1-j0+1)
   allocate(Vsum(i1-i0+1,j1-j0+1))
   call MPI_REDUCE(V,Vsum,N,MPI_REAL8, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast) call MPI_BCAST(V,N,MPI_REAL8, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif

   deallocate(Vsum)
   end subroutine sum_real_array_2D

   subroutine sum_real_array_3D(V,do_bcast)
   use mpi
   real(8), intent(INOUT) :: V(:,:,:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,j0,i1,j1,N,k0,k1
   real(8) , allocatable :: Vsum(:,:,:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
   k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
   N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
   allocate(Vsum(i1-i0+1,j1-j0+1,k1-k0+1))
   call MPI_REDUCE(V,Vsum,N,MPI_REAL8, MPI_SUM, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast) call MPI_BCAST(V,N,MPI_REAL8, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif
   deallocate(Vsum)
   end subroutine sum_real_array_3D

   subroutine sum_logical_scalar(i,do_bcast)
   use mpi
   logical, intent(INOUT) :: i
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr
   logical isum
   call MPI_REDUCE(i,isum,1,MPI_LOGICAL, MPI_LOR, COMM%master_id, MPI_COMM_WORLD, ierr )
   i=isum
   if (present(do_bcast)) then
    if (do_bcast) call  MPI_BCAST(i,1,MPI_LOGICAL, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif

   end subroutine sum_logical_scalar

   subroutine sum_logical_array_1D(V,do_bcast)
   use mpi
   logical, intent(INOUT) :: V(:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,i1,j0,j1,k0,k1,N
   logical,allocatable :: Vsum(:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   N = (i1-i0+1)
   allocate(Vsum(N))
   call MPI_REDUCE(V,Vsum,N,MPI_LOGICAL, MPI_LOR, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast) call  MPI_BCAST(V,N,MPI_LOGICAL, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif
   deallocate(Vsum)
   end subroutine sum_logical_array_1D

   subroutine sum_logical_array_2D(V,do_bcast)
   use mpi
   logical, intent(INOUT) :: V(:,:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,i1,j0,j1,k0,k1,N
   logical,allocatable :: Vsum(:,:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
   N = (i1-i0+1)*(j1-j0+1)
   allocate(Vsum((i1-i0+1),(j1-j0+1)))
   call MPI_REDUCE(V,Vsum,N,MPI_LOGICAL, MPI_LOR, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast) call  MPI_BCAST(V,N,MPI_LOGICAL, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif
   deallocate(Vsum)
   end subroutine sum_logical_array_2D

   subroutine sum_logical_array_3D(V,do_bcast)
   use mpi
   logical, intent(INOUT) :: V(:,:,:)
   logical, optional, intent(IN) :: do_bcast
   logical l
   integer ierr,i0,i1,j0,j1,k0,k1,N
   logical,allocatable :: Vsum(:,:,:)
   i0 = lbound(V,dim=1); i1 = ubound(V,dim=1)
   j0 = lbound(V,dim=2); j1 = ubound(V,dim=2)
   k0 = lbound(V,dim=3); k1 = ubound(V,dim=3)
   N = (i1-i0+1)*(j1-j0+1)*(k1-k0+1)
   allocate(Vsum((i1-i0+1),(j1-j0+1),(k1-k0+1)))
   call MPI_REDUCE(V,Vsum,N,MPI_LOGICAL, MPI_LOR, COMM%master_id, MPI_COMM_WORLD, ierr )
   V=Vsum
   if (present(do_bcast)) then
    if (do_bcast) call  MPI_BCAST(V,N,MPI_LOGICAL, COMM%master_id,MPI_COMM_WORLD,ierr)
   endif
   deallocate(Vsum)
   end subroutine sum_logical_array_3D


end module comunications


