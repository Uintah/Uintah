! Print some 1D 2D 3D arrays into a file. NOT used in this version of the code 

 module printer_arrays
 implicit none
 
 interface print_in_file
   module procedure print_scalar !(V,nf)
   module procedure print_1D_array !(x,V,nf)
   module procedure print_2D_array !(x,V,nf)
   module procedure print_3D_array !(x,y,V,nf)
   module procedure print_4D_array
 end interface print_in_file

 private :: print_scalar, print_1D_array, print_2D_array, print_3D_array, print_4D_array
 contains
    subroutine print_scalar(V,nf)
    character(*) , intent(IN):: nf
    real(8), intent(IN) :: V
    open(unit=691,file=trim(nf))
    write(691,*) V
    close(691)
   end subroutine print_scalar

   subroutine print_1D_array(x,V,nf)
    character(*) , intent(IN):: nf
     real(8), intent(IN) :: V(:),x(:)
     integer up_x,up_V,down_x,down_V,N,i
     up_x = ubound(x,dim=1) ; down_x = lbound(x,dim=1) ; up_V = ubound(V,dim=1) ; down_V = lbound(V,dim=1)
     if ((up_x-down_x) .ne. (up_V-down_V)) then
         print*, 'ERROR in print_1D_array ; the arrays has different shapes and sizes'
         STOP
       endif
     open(unit=691,file=trim(nf),recl=10000)
       N= up_x-down_x + 1
       do i = 0,N-1
          write(691,*) x(down_x+i),' ',V(down_V+i)
       enddo
       close(691)
   end subroutine print_1D_array

   subroutine print_2D_array(x,V,nf)
       character(*) , intent(IN):: nf
       real(8), intent(IN) :: V(:,:),x(:)
       integer  up_x,up_V,down_x,down_V,N,i,up_V2,down_V2,k
       up_x = ubound(x,dim=1) ; down_x = lbound(x,dim=1) ; up_V = ubound(V,dim=1) ; down_V = lbound(V,dim=1)
       up_V2 = ubound(V,dim=2);  down_V2 = lbound(V,dim=2)
       
       if ((up_x-down_x) .ne. (up_V-down_V)) then
         print*, 'ERROR in print_2D_array ; the arrays has different shapes and sizes'
         STOP
       endif
       open(unit=691,file=trim(nf),recl=10000)
       N= up_x-down_x + 1
       do i = 0,N-1
          write(691,*) x(down_x+i),' ',((V(down_V+i,k),' '),k=down_V2,up_V2)
       enddo
       close(691)
   end subroutine print_2D_array

   subroutine print_3D_array(x,y,V,nf)
        character(*) , intent(IN):: nf
       real(8), intent(IN) :: V(:,:,:),x(:),y(:)
       integer  up_x,up_V1,down_x,down_V1,N,i, up_y,down_y,up_V2,down_V2,N1,N2,k,j
       integer up_V3,down_V3
!print*, '-----got in routine print shape(v)=',shape(V)
       up_x = ubound(x,dim=1) ; down_x = lbound(x,dim=1) ; up_V1 = ubound(V,dim=1) ; down_V1 = lbound(V,dim=1)
       up_y = ubound(y,dim=1) ; down_y = lbound(y,dim=1) ; up_V2 = ubound(V,dim=2) ; down_V2 = lbound(V,dim=2)
       up_V3 = ubound(V,dim=3) ; down_V3 = lbound(V,dim=3);
       if ((up_x-down_x) .ne. (up_V1-down_V1)) then
         print*, 'ERROR in print_3D_array ; the arrays x and V has different shapes and sizes', up_x,down_x,&
                up_x-down_x,up_V1,down_V1,up_V1-down_V1
  
          print*,'shape=', shape(V), up_V1, down_V1, up_V2, down_V2, ubound(V,dim=3), lbound(V,dim=3)
         STOP
       endif
       if ((up_y-down_y) .ne. (up_V2-down_V2)) then
         print*, 'ERROR in print_3D_array ; the arrays y and V has different shapes and sizes'
         STOP
       endif
 
       open(unit=691,file=trim(nf),recl=10000)
       N1= up_x-down_x + 1 ; N2 = up_y-down_y+1
       do i = 0,N1-1
       do j = 0,N2-1
      write(691,*) x(i+down_x),y(j+down_y),' ',(V(i+down_V1,j+down_V2,k),' ',k=down_V3,up_V3) 
       enddo
       enddo
       close(691)
!print*, '----got out of print array'
    end subroutine print_3D_array

   subroutine print_4D_array(x,y,V,nf)
        character(*) , intent(IN):: nf
       real(8), intent(IN) :: V(:,:,:,:),x(:),y(:)
       integer  up_x,up_V1,down_x,down_V1,N,i, up_y,down_y,up_V2,down_V2,N1,N2,k,j,kk
       integer up_V3,down_V3, up_V4,down_V4
!print*, '-----got in routine print shape(v)=',shape(V)
       up_x = ubound(x,dim=1) ; down_x = lbound(x,dim=1) ; up_V1 = ubound(V,dim=1) ; down_V1 = lbound(V,dim=1)
       up_y = ubound(y,dim=1) ; down_y = lbound(y,dim=1) ; up_V2 = ubound(V,dim=2) ; down_V2 = lbound(V,dim=2)
       up_V3 = ubound(V,dim=3) ; down_V3 = lbound(V,dim=3);
       up_V4 = ubound(V,dim=4) ; down_V4 = lbound(V,dim=4);
       if ((up_x-down_x) .ne. (up_V1-down_V1)) then
         print*, 'ERROR in print_3D_array ; the arrays x and V has different shapes and sizes', up_x,down_x,&
                up_x-down_x,up_V1,down_V1,up_V1-down_V1

          print*,'shape=', shape(V), up_V1, down_V1, up_V2, down_V2, ubound(V,dim=3), lbound(V,dim=3)
         STOP
       endif
       if ((up_y-down_y) .ne. (up_V2-down_V2)) then
         print*, 'ERROR in print_3D_array ; the arrays y and V has different shapes and sizes'
         STOP
       endif

       open(unit=691,file=trim(nf),recl=10000)
       N1= up_x-down_x + 1 ; N2 = up_y-down_y+1
       do i = 0,N1-1
       do j = 0,N2-1
      write(691,*) x(i+down_x),y(j+down_y),' ',((V(i+down_V1,j+down_V2,k,kk),' ',k=down_V3,up_V3) , kk = down_V4, up_V4)
       enddo
       enddo
       close(691)
!print*, '----got out of print array'
    end subroutine print_4D_array

end module  printer_arrays
