      SUBROUTINE rzibc(K,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,tg,
     &zintbc,zintfc,pcell,wall,symtry,ffield,pfield,sfield,
     &pbcfld,abskg,oxi,qfluxt,qfluxb)
C***********************************************************************
C	SUBROUTINE RZIBC - sets intensity boundary conditions for the 
C	current K-plane; returns KK
C       Modified by Gautham 03/02.
C***********************************************************************
      IMPLICIT NONE

      integer WALL,SYMTRY,FFIELD,PFIELD,SFIELD,PBCFLD,INW,I,J,K,
     &KK,L,SN,IST,IEND,JST,JEND,KST,KEND

      integer PCELL(IST:IEND,JST:JEND,KST:KEND)

      double precision TG(IST:IEND,JST:JEND,KST:KEND)
      double precision ZINTBC(IST:IEND,JST:JEND,KST:KEND)
      double precision ZINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision ABSKG(IST:IEND,JST:JEND,KST:KEND)
      double precision OXI(SN*(SN+2))

c     Added these two arrays to the boundary condition for the validation 
c     problem -Gautham

      double precision qfluxt(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxb(IST:IEND,JST:JEND,KST:KEND)

      double precision PI,SIG,PI1
 
        DATA PI/3.141593/,SIG/5.67E-08/

C	Intensity bc's for K-plane, xi (N) determines sweep direction;
C	BC's when sweeping from K=NK side (from Top to Bottom)

	PI1 = 1.0 / PI

	IF (OXI(L).LT. 0) THEN

	  KK = KST+KEND-K

	        DO 210 J=JST+1,JEND-1
	        DO 200 I=IST+1,IEND-1

C    CAUTION:I have hardcoded these boundary conditions for validation.
C    These must be removed later. -Gautham

          IF((KK.EQ.(KEND-1)).AND.(PCELL(I,J,KK+1).EQ.WALL))THEN
          TG(I,J,KK+1)=0.0
          ABSKG(I,J,KK+1)=1.0
          ENDIF

C     End of Boundary Conditions hardcoding.

	          IF ((KK.EQ.(KEND-1).AND.
     1                     PCELL(I,J,KK+1).EQ.FFIELD) .OR. 
     2                     PCELL(I,J,KK+1).EQ.PFIELD .OR.
     3                     PCELL(I,J,KK+1).EQ.SFIELD .OR.
     4                     PCELL(I,J,KK+1).EQ.PBCFLD .OR.
     5                     PCELL(I,J,KK+1).EQ.WALL) THEN

c	            ZINTBC(I,J,KK) = PI1 * (ABSKG(I,J,KK+1)
c     2                              * SIG * TG(I,J,KK+1) * TG(I,J,KK+1)
c     3                              * TG(I,J,KK+1) * TG(I,J,KK+1)) 

c     Modified boundary conditions for the validation problem

	            ZINTBC(I,J,KK) = PI1 * ((1.0 - ABSKG(I,J,KK+1)) 
     1                              * qfluxt(I,J,KK+1) + PI1 
     2                              * ABSKG(I,J,KK+1)
     3                              * SIG * TG(I,J,KK+1) * TG(I,J,KK+1)
     4                              * TG(I,J,KK+1) * TG(I,J,KK+1))

	          ELSE
	            ZINTBC(I,J,KK) = ZINTFC(I,J,KK+1)

	          END IF
 
200	        CONTINUE
210	        CONTINUE

C	BC's when sweeping from K=1 side (from Bottom to Top)

	ELSE

	  KK = K

	  DO 260 J=JST+1,JEND-1
	  DO 250 I=IST+1,IEND-1

C    CAUTION:I have hardcoded these boundary conditions for validation.
C    These must be removed later. -Gautham

          IF((KK.EQ.(KST+1)).AND.(PCELL(I,J,KK-1).EQ.WALL))THEN
          TG(I,J,KK-1)=0.0
          ABSKG(I,J,KK-1)=1.0
          ENDIF

C     End of Boundary Conditions hardcoding.

	          IF ((KK.EQ.(KST+1).AND.
     1                     PCELL(I,J,KK-1).EQ.FFIELD) .OR. 
     2                     PCELL(I,J,KK-1).EQ.PFIELD .OR.
     3                     PCELL(I,J,KK-1).EQ.SFIELD .OR.
     4                     PCELL(I,J,KK-1).EQ.PBCFLD .OR.
     5                     PCELL(I,J,KK-1).EQ.WALL) THEN

c	            ZINTBC(I,J,KK) = PI1 * (ABSKG(I,J,KK-1)
c     2                              * SIG * TG(I,J,KK-1) * TG(I,J,KK-1)
c     3                              * TG(I,J,KK-1) * TG(I,J,KK-1)) 

c     Modified boundary conditions for the validation problem

	            ZINTBC(I,J,KK) = PI1 * ((1.0 - ABSKG(I,J,KK-1)) 
     1                              * qfluxb(I,J,KK-1) + PI1 
     2                              * ABSKG(I,J,KK-1)
     3                              * SIG * TG(I,J,KK-1) * TG(I,J,KK-1)
     4                              * TG(I,J,KK-1) * TG(I,J,KK-1))

	          ELSE
	            ZINTBC(I,J,KK) = ZINTFC(I,J,KK-1)

	          END IF

250	   CONTINUE
260	   CONTINUE

	END IF

	RETURN
	END
