module random_generator_module
public :: ran2
public :: GAUSS_DISTRIB
public ::  RANF
contains


real(8) function ran2(idum)
! This one was recomended by Prof. N. Cann as being a good random generator
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


    REAL(8) FUNCTION GAUSS_DISTRIB ( DUMMY )
! Gaussian random distribution (for velocities init)
! token from CCP5 (www.ccp5.com)
    implicit none
        REAL(8)       A1, A3, A5, A7, A9
        PARAMETER ( A1 = 3.949846138, A3 = 0.252408784 )
        PARAMETER ( A5 = 0.076542912, A7 = 0.008355968 )
        PARAMETER ( A9 = 0.029899776                   )

        REAL(8)        SUMA, R, R2
        REAL(8)        DUMMY
        INTEGER     I


        SUMA = 0.0d0

        DO I = 1, 12
           SUMA = SUMA + RANF ( DUMMY )
        ENDDO

        R  = ( SUMA - 6.0 ) / 4.0
        R2 = R * R

        GAUSS_DISTRIB = (((( A9 * R2 + A7 ) * R2 + A5 ) * R2 + A3 ) * R2 +A1 ) * R

    END FUNCTION GAUSS_DISTRIB



     REAL(8) FUNCTION RANF ( DUMMY )
! Uniform random variable
! Token from CCP5
        INTEGER     L, C, M
        PARAMETER ( L = 1029, C = 221591, M = 1048576 )
        INTEGER, SAVE :: SEED=0
        REAL(8)        DUMMY

        SEED = MOD ( SEED * L + C, M )
        RANF = dble ( SEED ) / dble(M)

    END FUNCTION RANF

end module random_generator_module

