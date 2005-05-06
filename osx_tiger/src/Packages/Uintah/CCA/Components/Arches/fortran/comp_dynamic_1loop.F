c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_1loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision sij1c, sij2c, sij3c, sij4c, sij5c, sij6c
      double precision isic, trace, denc
      double precision uvelc, vvelc, wvelc

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO K = KST, KEND
         DO J = JST, JEND
            DO I = IST, IEND   
               sij1c = sij1(i,j,k)
               sij2c = sij2(i,j,k)
               sij3c = sij3(i,j,k)
               sij4c = sij4(i,j,k)
               sij5c = sij5(i,j,k)
               sij6c = sij6(i,j,k)
               isic = sqrt(2.0D0*(sij1c*sij1c+sij2c*sij2c+
     +                            sij3c*sij3c+2.0D0*(sij4c*sij4c+
     +                            sij5c*sij5c+sij6c*sij6c)))
C              trace has been neglected
C              trace = (sij1c+sij2c+sij3c)/3.0D0
               trace = 0.0D0
               uvelc = ccuvel(i,j,k)
               vvelc = ccvvel(i,j,k)
               wvelc = ccwvel(i,j,k)
               denc = den(i,j,k)

               isi(i,j,k) = isic

               betaij1(i,j,k) = denc*isic*(sij1c-trace)
               betaij2(i,j,k) = denc*isic*(sij2c-trace)
               betaij3(i,j,k) = denc*isic*(sij3c-trace)
               betaij4(i,j,k) = denc*isic*sij4c
               betaij5(i,j,k) = denc*isic*sij5c
               betaij6(i,j,k) = denc*isic*sij6c

C              required to compute Leonard term
               rhouu(i,j,k) = denc*uvelc*uvelc
               rhouv(i,j,k) = denc*uvelc*vvelc
               rhouw(i,j,k) = denc*uvelc*wvelc
               rhovv(i,j,k) = denc*vvelc*vvelc
               rhovw(i,j,k) = denc*vvelc*wvelc
               rhoww(i,j,k) = denc*wvelc*wvelc
               rhou(i,j,k) = denc*uvelc
               rhov(i,j,k) = denc*vvelc
               rhow(i,j,k) = denc*wvelc

            END DO
         END DO
      END DO

      RETURN
      END
