module random_generator_module
public :: ran2
public :: randomize_config
contains


real(8) function ran2(idum) ! random number from 0 to 1
! and worked with it all my Ph.D. It's from numerical receipes
        implicit none
        integer, parameter :: im1=2147483563,im2=2147483399, &
                 IMM1=IM1-1,ia1=40014,ia2=40692,iq1=53668,iq2=52774,ir1=12211,ir2=3791, &
                 NTAB=32,NDIV=1+IMM1/Ntab
        real(8) , parameter :: eps=1.2d-7,RNMX=1.0d0-eps,am=1.0d0/2147483563.0d0
        integer i,j,k,iy,idum,idum2
        integer iv(Ntab)
        save iv,iy,idum2
        data idum2/123456789/ ,iv/NTAB*0/,iy/0/
        if (idum.le.0) then
          idum=max(-idum,1)
          idum2=idum
          do j=NTAB+8,1,-1
            k=idum/iq1
            idum=ia1*(idum-k*iq1)-k*ir1
            if (idum.lt.0) then
              idum=idum+im1
            endif
            if (j.le.NTAB) then
              iv(j)=idum
            endif
          enddo
          iy=iv(1)
         endif
         k=idum/iq1
         idum=ia1*(idum-k*iq1)-k*ir1
         if (idum.lt.0) then
            idum=idum+im1
         endif
         k=idum2/iq2
         idum2=ia2*(idum2-k*iq2)-k*ir2
         if (idum2.lt.0) then
           idum2=idum2+im2
         endif

         j=1+iy/NDIV
         iy=iv(j)-idum2
         iv(j)=idum
         if (iy.lt.1) then
           iy=iy+imm1
         endif
         ran2=min(AM*iy,RNMX)
end function ran2

subroutine randomize_config(displacement,iseed, xxx,yyy,zzz)
real(8), intent(IN) ::  displacement
integer, intent(INOUT) :: iseed
real(8), intent(INOUT) :: xxx(:),yyy(:),zzz(:)
integer i0,i1,i,j,k
i0=ubound(xxx,dim=1); i1=lbound(xxx,dim=1)
do i = 1, i0
xxx(i) = xxx(i) + (ran2(iseed)-0.5d0) * displacement
yyy(i) = yyy(i) + (ran2(iseed)-0.5d0) * displacement
zzz(i) = zzz(i) + (ran2(iseed)-0.5d0) * displacement
enddo
end subroutine randomize_config
end module random_generator_module

