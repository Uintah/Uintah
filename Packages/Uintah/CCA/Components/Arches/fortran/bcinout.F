c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/bcinout_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      if (xminus) then
         I = ist -1
         DO 140 K=KST,KEND
            DO 130 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin + den(I,J,K)*sns(j)*stb(k)*
     $                 max(0.0D0,uu(i,j,k))
                  flowout = flowout - pt5*(den(I,J,K)+den(i+1,j,k))*
     $                 sns(j)*stb(k)*min(0.0D0,uu(i+1,j,k))
               endif
 130        CONTINUE
 140     CONTINUE
      endif
      if (xplus) then
         I = iend
         DO 190 K=KST,KEND
	    DO 180 J=JST,JEND
               IF (PCELL(iend+1,J,K) .EQ. PFIELD) THEN
c                  flowin = flowin - den(I,J,K)*sns(j)*stb(k)*
c     $                 min(0.0D0,uu(i+1,j,k))
                  flowout = flowout + pt5*(den(I-1,J,K)+den(i,j,k))
c     $                 *sns(j)*stb(k)*max(0.0D0,uu(i+1,j,k))
     $                 *sns(j)*stb(k)*uu(i,j,k)
               END IF
 180        CONTINUE
 190     CONTINUE
      endif
C
C	Compute PBC for J=1 (South) side of domain
C
      if (yminus) then
         J = jst-1
         DO 240 K=KST,KEND
            DO 230 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin + den(I,J,K)*sew(i)*stb(k)*
     $                 max(0.0D0,vv(i,j,k))
                  flowout = flowout - pt5*(den(I,J,K)+den(i,j+1,k))*
     $                 sew(i)*stb(k)*min(0.0D0,vv(i,j+1,k))
               END IF
 230        CONTINUE
 240     CONTINUE
      endif
C     
C     Compute PBC for J=NJ (North) side of domain
C     
      if (yplus) then
         J = jend+1
         DO 290 K=KST,KEND
	    DO 280 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin - den(I,J,K)*sew(i)*stb(k)*
     $                 min(0.0D0,vv(i,j+1,k))
                  flowout = flowout + pt5*(den(I,J,K)+den(i,j-1,k))
     $                 *sew(i)*stb(k)*max(0.0D0,vv(i,j,k))
               END IF
 280        CONTINUE
 290     CONTINUE
      endif

C
C	Calculate PBC for K=1 (Bottom) side of domain
C
      if (zminus) then
         K = kst-1
         DO 340 J=JST,JEND
	    DO 330 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin + den(I,J,K)*sew(i)*sns(j)*
     $                 max(0.0D0,ww(i,j,k))
                  flowout = flowout - pt5*(den(I,J,K)+den(i,j,k+1))*
     $                 sew(i)*sns(j)*min(0.0D0,ww(i,j,k+1))
               END IF
 330        CONTINUE
 340     CONTINUE
      endif
C
C	Compute PBC for K=NK (Top) side of domain
C
      if (zplus) then
         K = kend+1
         DO 390 J=JST,JEND
	    DO 380 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin - den(I,J,K)*sew(i)*sns(j)*
     $                 min(0.0D0,ww(i,j,k+1))
                  flowout = flowout + pt5*(den(I,J,K)+den(i,j,k-1))
     $                 *sew(i)*sns(j)*max(0.0D0,ww(i,j,k))
               END IF
 380        CONTINUE
 390     CONTINUE
      endif
C     
      RETURN
      END

