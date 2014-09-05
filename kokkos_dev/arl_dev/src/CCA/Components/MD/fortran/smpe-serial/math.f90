module math
implicit none
public :: gauss
public :: gsolve
public :: det3
public :: invert3
public :: get_minors_3x3
contains


      subroutine gauss(matrix,defsiz,siz,pivot)
!C
!C     This subroutine performs gaussian elimination with scaled partial
!C     pivoting (rows).  the multipliers are stored in the eliminated entries.
!C     row pivoting information is stored in a vector.
!C
      implicit none
!C
!C     *****shared variables
!C
      integer siz,defsiz,pivot(defsiz)
      real(8) matrix(defsiz,defsiz)
!C
!C     *****local variables
!C
      integer row,ind1,ind2,ix,brow,bstore
      real(8) big,bigent,mult,zzero,store

      zzero=1.0d-07
      do row=1,siz
        pivot(row)=row
      end do
      do row=1, siz-1
        big=0.0
        brow=0
        do ind1=row,siz
          bigent=0.0d0
          do ind2=row+1,siz
            if (abs(matrix(ind1,ind2)) .gt. bigent) then
              bigent=abs(matrix(ind1,ind2))
            end if
          end do
!C
!C     *****check for singular matrix
!C
          if ((abs(matrix(ind1,row)) .le. zzero) .and. ( bigent .le. zzero)) then
            write (6,*) 'matrix is singular'
          end if

          if (bigent .lt. zzero)  bigent=zzero
          if (abs(matrix(ind1,row)/bigent) .gt. big) then
            big=abs(matrix(ind1,row)/bigent)
            brow=ind1
          end if
        end do
!C
!C     *****perform pivot, update pivot vector
!C
        if (brow .ne. row) then
          do ind1=1,siz
            store=matrix(row,ind1)
            matrix(row,ind1)=matrix(brow,ind1)
            matrix(brow,ind1)=store
          end do
          bstore=pivot(row)
          pivot(row)=pivot(brow)
          pivot(brow)=bstore
        end if

        do ind1=row+1,siz
          mult=matrix(ind1,row)/matrix(row,row)
          do ind2=row+1,siz
            matrix(ind1,ind2)=matrix(ind1,ind2)-mult*matrix(row,ind2)
          end do
          matrix(ind1,row)=mult
        end do
      end do

      end  subroutine gauss


      subroutine gsolve(matrix,defsiz,siz,b,xt,pivot)
!C
!C     This subroutine takes a gauss eliminated matrix with
!C       multipliers stored in eliminated entries, a b vector, and
!C       a pivot vector containing row pivot information associated
!C       with the matrix and performs backward substitution to yield the
!C       solution vector
!C
      implicit none
      integer defsiz,siz,pivot(defsiz)
      real(8) b(defsiz),matrix(defsiz,defsiz),xt(defsiz)
      integer ind1,ind2
      real(8) mult,suma
      real(8) btemp(20)
!C
!C     *****pivot b vector
!C
      do ind1=1,siz
        btemp(ind1)=b(ind1)
      end do
!C
      do ind1=1,siz
        b(ind1)=btemp(pivot(ind1))
      end do
!C
!C     *****perform elimination on b vector
!C
      do ind1=1,siz-1
        do ind2=ind1+1,siz
          mult=matrix(ind2,ind1)
          b(ind2)=b(ind2)-b(ind1)*mult
        end do
      end do

      xt(siz)=b(siz)/matrix(siz,siz)
      do ind1=siz-1,1,-1
        suma=0.0
        do ind2=ind1+1,siz
          suma=suma+matrix(ind1,ind2)*xt(ind2)
        end do
        xt(ind1)=(b(ind1)-suma)/matrix(ind1,ind1)
      end do
   end subroutine gsolve


   real(8) FUNCTION det3(a,b,c)
      implicit none
      real(8) a(3),b(3),c(3)
      det3 = a(1)*(b(2)*c(3)-b(3)*c(2)) -a(2)*(b(1)*c(3)-b(3)*c(1)) +a(3)*(b(1)*c(2)-b(2)*c(1))
   end FUNCTION det3

  subroutine invert3(a,a_inv,det)
    real(8), intent(IN) ::  a(9)
    real(8), intent(OUT) :: a_inv(9),det
    real(8) i_det

    a_inv(1)=a(5)*a(9)-a(6)*a(8) ; a_inv(2)=a(3)*a(8)-a(2)*a(9) ; a_inv(3)=a(2)*a(6)-a(3)*a(5)
    a_inv(4)=a(6)*a(7)-a(4)*a(9) ; a_inv(5)=a(1)*a(9)-a(3)*a(7) ; a_inv(6)=a(3)*a(4)-a(1)*a(6)
    a_inv(7)=a(4)*a(8)-a(5)*a(7) ; a_inv(8)=a(2)*a(7)-a(1)*a(8) ; a_inv(9)=a(1)*a(5)-a(2)*a(4)
    det=a(1)*a_inv(1)+a(4)*a_inv(2)+a(7)*a_inv(3)
    if(dabs(det) > 0.d0) then 
      i_det = 1.d0/det
    else
      i_det = 0.0d0
    endif
    a_inv(:)=i_det*a_inv(:)

   end subroutine invert3

   subroutine get_minors_3x3(a,a11,a12,a13,a21,a22,a23,a31,a32,a33)
      real(8), intent(IN) :: a(9)
      real(8), intent(OUT) :: a11,a12,a13,a21,a22,a23,a31,a32,a33
      a11=a(5)*a(9)-a(6)*a(8)   ! line is first ; colomn second
      a12=a(6)*a(7)-a(4)*a(9)
      a13=a(4)*a(8)-a(5)*a(7)
      a21=a(8)*a(3)-a(2)*a(9)
      a22=a(1)*a(9)-a(3)*a(7)
      a23=a(2)*a(7)-a(1)*a(8)
      a31=a(2)*a(6)-a(3)*a(5)
      a32=a(3)*a(4)-a(1)*a(6)
      a33=a(1)*a(5)-a(2)*a(4)
   end subroutine get_minors_3x3



end module math
