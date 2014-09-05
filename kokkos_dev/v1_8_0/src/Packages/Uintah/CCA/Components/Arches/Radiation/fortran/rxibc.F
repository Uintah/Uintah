      SUBROUTINE rxibc(I,II,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,tg,
     &xintbc,xintfc,pcell,wall,symtry,ffield,pfield,sfield,
     &pbcfld,abskg,omu,qfluxe,qfluxw)
C***********************************************************************
C	SUBROUTINE RXIBC - sets intensity boundary conditions for the 
C	current I-point
C       Modified by Gautham 03/02.
C***********************************************************************
      IMPLICIT NONE

      integer WALL,SYMTRY,FFIELD,PFIELD,SFIELD,PBCFLD,I,II,
     &JJ,KK,L,SN,IST,IEND,JST,JEND,KST,KEND

      integer PCELL(IST:IEND,JST:JEND,KST:KEND)

      double precision TG(IST:IEND,JST:JEND,KST:KEND)
      double precision XINTBC(IST:IEND,JST:JEND,KST:KEND)
      double precision XINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision ABSKG(IST:IEND,JST:JEND,KST:KEND)
      double precision OMU(SN*(SN+2))

c     Added these two arrays to the boundary condition for the validation 
c     problem -Gautham

      double precision qfluxe(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxw(IST:IEND,JST:JEND,KST:KEND)

      double precision AF,PI,SIG,PI1

	DATA PI/3.141593/,SIG/5.67E-08/

C	Intensity bc's for J-row, eta (M) determines sweep direction;
C	BC's when sweeping from J=NJ side (from North to South)
 
	PI1 = 1.0 / PI

C	I-loop - picks I-cell, performs E-W (x) sweeps thru I
C	         in given J-row, direction depends on mu (L)

C	Sweep from I=NI side (from East to West)

	          IF (OMU(L) .LT. 0) THEN

                  II = IST+IEND-I

C	Do intensity calculations only if FFIELD

	              IF (PCELL(II,JJ,KK) .EQ. FFIELD) THEN

C    CAUTION:I have hardcoded these boundary conditions for validation.
C    These must be removed later. -Gautham

          IF((II.EQ.(IEND-1)).AND.(PCELL(II+1,JJ,KK).EQ.WALL))THEN
          TG(II+1,JJ,KK)=0.0
          ABSKG(II+1,JJ,KK)=1.0
          ENDIF

C     End of Boundary Conditions hardcoding.

C	Set intensity boundary condition for current I-cell
C	when sweeping from I=NI side (from East to West)

	              IF ((II.EQ.(IEND-1).AND.
     1                            PCELL(II+1,JJ,KK).EQ.FFIELD) .OR. 
     2                           PCELL(II+1,JJ,KK).EQ.PFIELD .OR.
     3                           PCELL(II+1,JJ,KK).EQ.SFIELD .OR.
     4                           PCELL(II+1,JJ,KK).EQ.PBCFLD .OR.
     5                           PCELL(II+1,JJ,KK).EQ.WALL) THEN

c	     XINTBC(II,JJ,KK) = PI1 * (ABSKG(II+1,JJ,KK)
c     2                         * SIG * TG(II+1,JJ,KK) * TG(II+1,JJ,KK)
c     3                         * TG(II+1,JJ,KK) * TG(II+1,JJ,KK))

c     Modified boundary conditions for the validation problem

              XINTBC(II,JJ,KK) = PI1 * ((1.0 - ABSKG(II+1,JJ,KK))
     1                         * qfluxe(II+1,JJ,KK) + ABSKG(II+1,JJ,KK)
     2                         * SIG * TG(II+1,JJ,KK) * TG(II+1,JJ,KK)
     3                         * TG(II+1,JJ,KK) * TG(II+1,JJ,KK))

                      ELSE
                           XINTBC(II,JJ,KK) = XINTFC(II+1,JJ,KK)

	              END IF

	              END IF

C	Sweep from I=1 side (from West to East)

	      ELSE

                    II=I

C	Do intensity calculations only if FFIELD

	              IF (PCELL(II,JJ,KK) .EQ. FFIELD) THEN

C    CAUTION:I have hardcoded these boundary conditions for validation.
C    These must be removed later. -Gautham

          IF((II.EQ.(IST+1)).AND.(PCELL(II-1,JJ,KK).EQ.WALL))THEN
          TG(II-1,JJ,KK)=0.0
          ABSKG(II-1,JJ,KK)=1.0
          ENDIF

C     End of Boundary Conditions hardcoding.

C	Set intensity boundary condition for current I-cell
C	when sweeping from I=1 side (from West to East)
	                
                      IF ((II.EQ.(IST+1).AND.
     1                           PCELL(II-1,JJ,KK).EQ.FFIELD) .OR. 
     2                           PCELL(II-1,JJ,KK).EQ.PFIELD .OR.
     3                           PCELL(II-1,JJ,KK).EQ.SFIELD .OR.
     4                           PCELL(II-1,JJ,KK).EQ.PBCFLD .OR.
     5                           PCELL(II-1,JJ,KK).EQ.WALL) THEN

c	      XINTBC(II,JJ,KK)= PI1 * (ABSKG(II-1,JJ,KK) 
c     2                         * SIG * TG(II-1,JJ,KK) * TG(II-1,JJ,KK)
c     3                         * TG(II-1,JJ,KK) * TG(II-1,JJ,KK))

c     Modified boundary conditions for the validation problem

              XINTBC(II,JJ,KK) = PI1 * ((1.0 - ABSKG(II-1,JJ,KK)) 
     1                         * qfluxw(II-1,JJ,KK) + ABSKG(II-1,JJ,KK) 
     2                         * SIG * TG(II-1,JJ,KK) * TG(II-1,JJ,KK)
     3                         * TG(II-1,JJ,KK) * TG(II-1,JJ,KK))

                      ELSE
                           XINTBC(II,JJ,KK) = XINTFC(II-1,JJ,KK)

	              END IF
 
	              END IF

	          END IF

                  RETURN
                  
                  END


