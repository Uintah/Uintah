      SUBROUTINE rdomcintm(II,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,
     &sew,sns,stb,AREAEW,omu,oeta,oxi,sn,abskg,esrcg,pcell,
     &ffield,AF,xintbc,xintfc,yintbc,yintfc,zintbc,zintfc,cintm,
     &NEGINT)

      IMPLICIT NONE

      integer FFIELD,NEGINT,II,JJ,KK,L,SN,AF,IST,IEND,JST,JEND,
     &KST,KEND
      integer PCELL(IST:IEND,JST:JEND,KST:KEND)

      double precision ABSKG(IST:IEND,JST:JEND,KST:KEND)
      double precision ESRCG(IST:IEND,JST:JEND,KST:KEND)
      double precision XINTBC(IST:IEND,JST:JEND,KST:KEND)
      double precision XINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision YINTBC(IST:IEND,JST:JEND,KST:KEND)
      double precision YINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision ZINTBC(IST:IEND,JST:JEND,KST:KEND)
      double precision ZINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision CINTM(IST:IEND,JST:JEND,KST:KEND)
      double precision VESRCG(IST:IEND)
      double precision AREAEW
      double precision AREAN(IST:IEND)
      double precision AREATB(IST:IEND)
      double precision SEW(IST:IEND)
      double precision SNS(IST:IEND)
      double precision STB(IST:IEND)
      double precision VOL(IST:IEND)
      double precision DM1(IST:IEND)
      double precision DM2(IST:IEND)
      double precision DM3(IST:IEND)
      double precision DM4(IST:IEND)
      double precision DM5(IST:IEND)
      double precision DENOM(IST:IEND)
      double precision OXI(SN*(SN+2))
      double precision OMU(SN*(SN+2))
      double precision OETA(SN*(SN+2))

C       Load vectorizable area,volume arrays and vesrcg
c In bcenthalpy.F, RV(JJ) has been removed, therefore it is removed 
c here also.

C                  ARNS(II) = RV(JJ) * SEW(II) * STB(KK)
                  AREAN(II) = SEW(II)*STB(KK)                 
                  AREATB(II) = SEW(II) * SNS(JJ)

c                  VOL(II) = RA(JJ) * SNS(JJ) * SEW(II) * STB(KK)
                  VOL(II) = SEW(II) * SNS(JJ) * STB(KK)
	          VESRCG(II) = VOL(II) * ESRCG(II,JJ,KK)

	          DM1(II) = ABS(OMU(L)) * AREAEW * (1.0 + AF)
	          DM2(II) = ABS(OETA(L)) * AREAN(II) * (1.0 + AF)
	          DM3(II) = ABS(OXI(L)) * AREATB(II) * (1.0 + AF)
	          DM4(II) =  ABSKG(II,JJ,KK) * VOL(II)
	          DENOM(II) = DM1(II) + DM2(II) + DM3(II) + DM4(II)

C	Calculate exit-face intensities at CV faces opposite BC faces,
C	this is based on Carlson and Lathrop's diamond differencing
C	scheme (central difference if AF = 1.0).

	            IF (PCELL(II,JJ,KK) .EQ. FFIELD) THEN

	              CINTM(II,JJ,KK) = 
     1                              ( DM1(II) * XINTBC(II,JJ,KK)
     2                              + DM2(II) * YINTBC(II,JJ,KK) 
     3                              + DM3(II) * ZINTBC(II,JJ,KK)
     4                              + VESRCG(II)) / DENOM(II)
               
      	     XINTFC(II,JJ,KK) = (1.0 + AF) * CINTM(II,JJ,KK)
     1                            - AF * XINTBC(II,JJ,KK)
	     YINTFC(II,JJ,KK) = (1.0 + AF) * CINTM(II,JJ,KK)
     1                            - AF * YINTBC(II,JJ,KK)
	     ZINTFC(II,JJ,KK) = (1.0+AF) * CINTM(II,JJ,KK)
     1                            - AF * ZINTBC(II,JJ,KK)

	              IF ((XINTFC(II,JJ,KK).LT.0.0) .OR. 
     1                 (YINTFC(II,JJ,KK).LT.0.0)
     1                  .OR. (ZINTFC(II,JJ,KK).LT.0.0))NEGINT = 1

	            ELSE

	              XINTFC(II,JJ,KK) = 0.0
	              YINTFC(II,JJ,KK) = 0.0
	              ZINTFC(II,JJ,KK) = 0.0
	              CINTM(II,JJ,KK)  = 0.0

	            END IF

	RETURN
	END

