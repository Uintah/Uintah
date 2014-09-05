      SUBROUTINE ryibc(J,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,tg,
     &yintbc,yintfc,pcell,wall,symtry,ffield,pfield,sfield,
     &pbcfld,abskg,oeta,qfluxn,qfluxs)  
C***********************************************************************
C	SUBROUTINE RYIBC - sets intensity boundary conditions for the 
C	current J-row; returns JJ
C       Modified by Gautham 03/02.
C***********************************************************************
      IMPLICIT NONE

      integer WALL,SYMTRY,FFIELD,PFIELD,SFIELD,PBCFLD,INW,I,J,JJ,
     &KK,L,SN,IST,IEND,JST,JEND,KST,KEND

      integer PCELL(IST:IEND,JST:JEND,KST:KEND)

      double precision TG(IST:IEND,JST:JEND,KST:KEND)
      double precision YINTBC(IST:IEND,JST:JEND,KST:KEND)
      double precision YINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision ABSKG(IST:IEND,JST:JEND,KST:KEND)
      double precision OETA(SN*(SN+2))

c     Added these two arrays to the boundary condition for the validation 
c     problem -Gautham

      double precision qfluxn(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxs(IST:IEND,JST:JEND,KST:KEND)

      double precision PI,SIG,PI1

	DATA PI/3.141593/,SIG/5.67E-08/

C	Intensity bc's for J-row, eta (M) determines sweep direction;
C	BC's when sweeping from J=NJ side (from North to South)
 
	PI1 = 1.0 / PI

        IF (OETA(L) .LT. 0) THEN

	  JJ = JST+JEND-J

	    DO 100 I=IST+1,IEND-1

C    CAUTION:I have hardcoded these boundary conditions for validation.
C    These must be removed later. -Gautham

            IF((JJ.EQ.(JEND-1)).AND.(PCELL(I,JJ+1,KK).EQ.WALL))THEN
            TG(I,JJ+1,KK)=0.0
            ABSKG(I,JJ+1,KK)=1.0
            ENDIF

C     End of Boundary Conditions hardcoding.

	      IF ((JJ.EQ.(JEND-1).AND.
     1                 PCELL(I,JJ+1,KK).EQ.FFIELD) .OR. 
     2                 PCELL(I,JJ+1,KK).EQ.PFIELD .OR.
     3                 PCELL(I,JJ+1,KK).EQ.SFIELD .OR.
     4                 PCELL(I,JJ+1,KK).EQ.PBCFLD .OR.
     5                 PCELL(I,JJ+1,KK).EQ.WALL) THEN

c	    YINTBC(I,JJ,KK) = PI1 * (ABSKG(I,JJ+1,KK) 
c     2                      * SIG * TG(I,JJ+1,KK) * TG(I,JJ+1,KK) 
c     3                      * TG(I,JJ+1,KK) * TG(I,JJ+1,KK))

c     Modified boundary conditions for the validation problem

            YINTBC(I,JJ,KK) = PI1 * ((1.0 - ABSKG(I,JJ+1,KK)) 
     1                      * qfluxn(I,JJ+1,KK) + ABSKG(I,JJ+1,KK) 
     2                      * SIG * TG(I,JJ+1,KK) * TG(I,JJ+1,KK) 
     3                      * TG(I,JJ+1,KK) * TG(I,JJ+1,KK))

	      ELSE
	        YINTBC(I,JJ,KK) = YINTFC(I,JJ+1,KK)

	      END IF

100	    CONTINUE


C	BC's when sweeping from J=1 side (from South to North)

	ELSE

	  JJ = J

	    DO 120 I=IST+1,IEND-1

C    CAUTION:I have hardcoded these boundary conditions for validation.
C    These must be removed later. -Gautham

            IF((JJ.EQ.(JST+1)).AND.(PCELL(I,JJ-1,KK).EQ.WALL))THEN
            TG(I,JJ-1,KK)=0.0
            ABSKG(I,JJ-1,KK)=1.0
            ENDIF

C     End of Boundary Conditions hardcoding.

	      IF ((JJ.EQ.(JST+1).AND.
     1                  PCELL(I,JJ-1,KK).EQ.FFIELD) .OR. 
     2                 PCELL(I,JJ-1,KK).EQ.PFIELD .OR.
     3                 PCELL(I,JJ-1,KK).EQ.SFIELD .OR.
     4                 PCELL(I,JJ-1,KK).EQ.PBCFLD .OR.
     5                 PCELL(I,JJ-1,KK).EQ.WALL) THEN

c       	    YINTBC(I,JJ,KK) = PI1 * (ABSKG(I,JJ-1,KK) 
c     2                      * SIG * TG(I,JJ-1,KK) * TG(I,JJ-1,KK) 
c     3                      * TG(I,JJ-1,KK) * TG(I,JJ-1,KK))

c     Modified boundary conditions for the validation problem

            YINTBC(I,JJ,KK) = PI1 * ((1.0 - ABSKG(I,JJ-1,KK)) 
     1                      * qfluxs(I,JJ-1,KK) + ABSKG(I,JJ-1,KK) 
     2                      * SIG * TG(I,JJ-1,KK) * TG(I,JJ-1,KK) 
     3                      * TG(I,JJ-1,KK) * TG(I,JJ-1,KK))

	      ELSE
	        YINTBC(I,JJ,KK) = YINTFC(I,JJ-1,KK)
	      END IF

120	    CONTINUE

	END IF

	RETURN
	END
