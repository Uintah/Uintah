c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE bcinout(domLoU, domHiU, UU,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
     &     domLoDen, domHiDen, 
     $     idxLo, idxHi, DEN, 
     $     domLo, domHi, PCELL,
     &     PFIELD, deltat,
     $     FLOWIN, FLOWOUT,
     $     domLoGeom, domHiGeom,
     $     SEW, SNS, STB,
     $     xminus, xplus, yminus, yplus, zminus, zplus,
     &     ierr)
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLoDen(3), domHiDen(3)
      integer idxLo(3), idxHi(3)
      double precision DEN(domLoDen(1):domHiDen(1), 
     $     domLoDen(2):domHiDen(2), 
     &     domLoDen(3):domHiDen(3))
      integer domLo(3), domHi(3)
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer domLoGeom(3), domHiGeom(3)
      double precision SEW(domLoGeom(1):domHiGeom(1))
      double precision SNS(domLoGeom(2):domHiGeom(2))
      double precision STB(domLoGeom(3):domHiGeom(3))
      double precision FLOWIN, FLOWOUT, deltat
      integer PFIELD
      integer xminus, xplus, yminus, yplus, zminus, zplus
      integer ierr

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
     $                 max(0.0,uu(i,j,k))
                  flowout = flowout - pt5*(den(I,J,K)+den(i+1,j,k))*
     $                 sns(j)*stb(k)*min(0.0,uu(i-1,j,k))
               endif
 130        CONTINUE
 140     CONTINUE
      endif
      if (xplus) then
         I = iend+1
         DO 190 K=KST,KEND
	    DO 180 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  flowin = flowin - den(I,J,K)*sns(j)*stb(k)*
     $                 min(0.0,uu(i+1,j,k))
                  flowout = flowout + pt5*(den(I,J,K)+den(i-1,j,k))
     $                 *sns(j)*stb(k)*max(0.0,uu(i,j,k))
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
     $                 max(0.0,vv(i,j,k))
                  flowout = flowout - pt5*(den(I,J,K)+den(i,j+1,k))*
     $                 sew(i)*stb(k)*min(0.0,vv(i,j+1,k))
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
     $                 min(0.0,vv(i,j+1,k))
                  flowout = flowout + pt5*(den(I,J,K)+den(i,j-1,k))
     $                 *sew(i)*stb(k)*max(0.0,vv(i,j,k))
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
     $                 max(0.0,ww(i,j,k))
                  flowout = flowout - pt5*(den(I,J,K)+den(i,j,k+1))*
     $                 sew(i)*sns(j)*min(0.0,ww(i,j,k+1))
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
     $                 min(0.0,ww(i,j,k+1))
                  flowout = flowout + pt5*(den(I,J,K)+den(i,j,k-1))
     $                 *sew(i)*sns(j)*max(0.0,ww(i,j,k))
               END IF
 380        CONTINUE
 390     CONTINUE
      endif
C     
      RETURN
      END

