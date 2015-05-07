

      SUBROUTINE DIAMM_CALC(NBLK,NINSV,DTARG,UI,SIGARG,D,SVARG,USM)
C***********************************************************************
C
C     Description:
C           Drucker-Prager plasticity model with elastic strain induced
C           anisotropy.
C
C***********************************************************************
C
C     input arguments
C     ===============
C      NBLK       int                   Number of blocks to be processed
C      NINSV      int                   Number of internal state vars
C      DTARG      dp                    Current time increment
C      UI       dp,ar(nprop)            User inputs
C      D          dp,ar(6)              Strain increment
C
C     input output arguments
C     ======================
C      STRESS   dp,ar(6)                stress
C      SVARG    dp,ar(ninsv)            state variables
C
C     output arguments
C     ================
C      USM      dp                      uniaxial strain modulus
C
C***********************************************************************
C
C      stresss and strains, plastic strain tensors
C          11, 22, 33, 12, 23, 13
C
C***********************************************************************
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C.............................................................parameters
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

      PARAMETER (PZERO=0.0D0,PONE=0.1D1,PTWO=0.2D1,PTHREE=0.3D1)
      PARAMETER (PFOUR=0.4D1,PFIVE=0.5D1,PSIX=0.6D1)
      PARAMETER (PSEVEN=0.7D1,PEIGHT=0.8D1,PNINE=0.9D1,PTEN=0.1D2)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (P3HALF=1.5D0)
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (P2THIRD=PTWO*PTHIRD)
      PARAMETER (  ROOT2=0.141421356237309504880168872420969807856967D1)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
      PARAMETER (  ROOT6=0.244948974278317809819728407470589139196595D1)
      PARAMETER (  TOOR2=0.707106781186547524400844362104849039284836D0)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
      PARAMETER (  TOOR6=0.408248290463863016366214012450981898660991D0)
      PARAMETER ( ROOT23=0.816496580927726032732428024901963797321982D0)
      PARAMETER ( ROOT32=0.122474487139158904909864203735294569598297D1)
      PARAMETER (TOL1M10= 1.0D-10,TOL1M20= 1.0D-20,TOL1M50=1.0D-50)
      PARAMETER (TOLJ=0.9999D0,YLDTOL=1.D-2)
C.................................................................common
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
C.................................................................passed
      DIMENSION SVARG(NINSV),SIGARG(6),USM(NBLK)
      DIMENSION UI(*),D(6)
C..................................................................local
      DIMENSION SV(NDMMISV),RN(6),RM(6)
C     symmetric strain tensors
      DIMENSION DE(6),DEDEV(6),DEP(6),DEE(6)
C     symmetric stress tensors
      DIMENSION TAUN(6),TAUP(6),QSTAUT(6),QSTAUN(6),QSTAU(6),
     &     QSTAUP(6),DTAUT(6),DTAU(6),DEPDEV(6)
C     symmetric basis tensors
      DIMENSION ES(6),EZ(6),RIDENT(6),EDS(6)
      SAVE EZ
      DATA EZ/TOOR3,TOOR3,TOOR3,PZERO,PZERO,PZERO/
      SAVE RIDENT
      DATA RIDENT/PONE,PONE,PONE,PZERO,PZERO,PZERO/
C     Needed for induced anisotropy
      DIMENSION E(6),EDEV(6)
C...................................................................data
      LOGICAL INELASTIC
      DATA INELASTIC/.FALSE./
C...............................................................external
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DIAMM_CALC
C
      IF(NINSV.NE.NDMMISV)CALL BOMBED('bad ninsv sent to Diamm')

C     Copy passed field arguments that are subject to
C     being updated into local arrays and initialize other variables

C--------------------------------------------------- start initial setup
C     Put user inputs in common variables
      CALL DMMPAR_PUT(UI)

C     Total strain rate square magnitude
      EQDOT= DMMTMAG(D)

C     Quasistatic and dynamic Kirchhoff stress at beginning of step
      DO IJ=1,6
         TAUN(IJ)= SIGARG(IJ)*SVARG(KRJ)
         QSTAUN(IJ)= SVARG(KQSSIG+IJ)*SVARG(KRJ)
      ENDDO
      RI1N= DMMTRACE(QSTAUN)
      RTJ2N= DMMROOTJ2(QSTAUN)

C     ISV's
      DO IROW=1,NINSV
         SV(IROW)= SVARG(IROW)
      ENDDO

C     Set default values for all output-only state variables
      SV(KEQDOT)= PZERO
      SV(KI1)= PZERO
      SV(KROOTJ2)= PZERO

C     Initialize variables that change with deformation
      T= SV(KT)
      U= SV(KEU)
      R= SV(KR)                 ! Density
      TMLT= DMMTMULT(T,T0,TM,XP) ! J-C homologous temperature
      EQPS= SV(KEQPS)           ! Equivalent plastic strain
      EQPV= SV(KEQPV)           ! Equivalent plastic strain

C     Strain increments
      DO IJ=1,6
         DE(IJ)= D(IJ)*DTARG
      ENDDO
      TRDE= DMMTRACE(DE)
      CALL DMMGETDEV(DE,DEDEV)  ! deviatoric strain increment
      DO IJ=1,6
         E(IJ)= SV(KE+IJ)
      ENDDO
      CALL DMMGETDEV(E,EDEV)
      BETA2= PHALF*DMMDBD(EDEV,EDEV)
      EVOL= SV(KEVOL)
      BETA1= DMMTRACE(E)
C----------------------------------------------------- end initial setup

C--------------------------------------------------- begin stress update
C     Total elastic strain, its deviator, and invariants

C     Calculate the stress-invariant dependent elastic props
      CALL DMM_MODULI(UI,U,R,T,BETA1,BETA2,EQPS,RI1N,RTJ2N,BM,SM)
      TWOG= PTWO*SM
      USM(NBLK)= PTHIRD*(PTWO*TWOG+PTHREE*BM)
      ETA= TWOG/PTHREE/BM

C     Trial stress
      DO IJ=1,6
         DTAUT(IJ)= BM*TRDE*RIDENT(IJ) + TWOG*DEDEV(IJ)
     &        + TWOG1*(TRDE*EDEV(IJ) + DMMDBD(DEDEV,EDEV)*RIDENT(IJ))
         QSTAUT(IJ)= QSTAUN(IJ) + DTAUT(IJ)
      ENDDO

C     Lode coordinates of trial stress state
      CALL DMMLODE(QSTAUT,ZT,ST,ES)

C     Call yield function
      GT= DMMYLDFNC(ZT,ST,EQPS,TMLT)
      IF(GT.LE.YLDTOL)THEN

C     Elastic
         DO IJ=1,6
            DEP(IJ)= PZERO
            DEE(IJ)= DE(IJ)
            QSTAUP(IJ)= QSTAUT(IJ)
            SV(KE+IJ)= SV(KE+IJ) + DEE(IJ) !total elastic strain
            DTAU(IJ)= DTAUT(IJ)
         ENDDO
         DEQPS= PZERO           ! Equivalent plastic strain rate
         GO TO 30
      ENDIF
      IF(WANTSELASTIC)THEN
         CALL BOMBED('Plastic loading has occurred')
      ENDIF

C     Plastic
      INELASTIC=.TRUE.
C     Apply oblique return to put stress on yield surface

C     Newton iterations to find magnitude of projection from the trial
C     stress state to the yield surface.
      DO I=1,25


C     Yield normal RN and flow direction RM
      RNZ= (A2*A3*EXP(A2*ZT) + A4)*TMLT
      RNS= TOOR2

C     Flow direction
      RMZ= (A2*A3*EXP(A2*ZT) + A4G)*TMLT
      RMS= RNS

      DO IJ=1,6
         RN(IJ)= RNZ*EZ(IJ)+RNS*ES(IJ)
         RM(IJ)= RMZ*EZ(IJ)+RMS*ES(IJ)
      ENDDO

C     Scaled components of coupling tensor
         ZZ= ROOT3*GP*DMMDBD(QSTAUN,RM)
         zz=pzero
         ZS= PZERO

C     Components of A tensor
         AZ= BM*RMZ + ROOT3*TWOG1*DMMDBD(EDEV,RM)
         AS= TWOG*RMS

C     Scaled components of return direction tensor
         FAC = SQRT((ZT**2+ST**2)/(RMZ**2+RMS**2))/BM
         PZ= FAC*(AZ + ZZ)
         PS= FAC*(AS + ZS)

C     Hardening modulus
         HY= DMMHYFNC(A3,A4,EQPS,RMS)
         HT= DMMHTFNC(QSTAU,RM,R0,CV)
         DFDY= TMLT
         DFDT= DMMDYLDFNCDT(TOOR3*DMMTRACE(QSTAUN),EQPS,T)
         H= -ALPH*(DFDY*HY+DFDT*HT)
ctim     until hardening is really properly implemented, set h to zero
         h=pzero

C     Projection "magnitude"
         BETA = -GT/(RNZ*PZ + RNS*PS)

C     Improved estimates for the test stress
         Z = ZT + BETA*PZ
         S = ST + BETA*PS

C     STEP 10: Check for convergence
         IF(ABS(BETA).LT.TOL1M10)GO TO 20

C     If not converged, set Z and S trial to the updated values of Z
C     and S and set G trial to the value of the yield function
C     corresponding to the updated Z and S.
         ZT = Z
         ST = S
         TMLT= DMMTMULT(T,T0,TM,XP)
         GT= DMMYLDFNC(ZT,ST,EQPS,TMLT)
      ENDDO

      CALL BOMBED('Newton iterations failed')

 20   CONTINUE

C     Updated quasistatic stress and increment
      DO IJ=1,6
         QSTAUP(IJ)= Z*EZ(IJ) + S*ES(IJ)
         DTAU(IJ)= QSTAUP(IJ) - QSTAUN(IJ)
      ENDDO
      CALL DMMLODE(DTAU,DZ,DS,EDS)


C     Elastic strain increment and updated elastic strain
      IF(ANISO)THEN
         ZETA= (BETA2*TWOG1**2)/SM/BM
         DDD= PSIX*BM*SM*(PONE-ZETA)/(TWOG1**2)
         Y1= DMMTRACE(DTAU)
         Y2= DMMDBD(EDEV,DTAU)
         ALPHA1= PONE/DDD*(P2THIRD*BETA2*Y1/BM - Y2/TWOG1)
         ALPHA2= PONE/DDD*(P3HALF*Y2/SM - Y1/TWOG1)
      ELSE
         ALPHA1= PZERO
         ALPHA2= PZERO
      ENDIF
      DO IJ=1,6
         DEE(IJ)= PTHIRD/BM*DZ*EZ(IJ) + PONE/TWOG*DS*EDS(IJ)
     &        + ALPHA1*RIDENT(IJ) + ALPHA2*EDEV(IJ)
         SV(KE+IJ)= SV(KE+IJ) + DEE(IJ) !total elastic strain
      ENDDO


C     Plastic strain increment
      DO IJ=1,6
         DEP(IJ)= DE(IJ)-DEE(IJ)
      ENDDO
      CALL DMMGETDEV(DEP,DEPDEV)
      DEQPS = SQRT(DMMDBD(DEPDEV,DEPDEV)+PTHIRD*(DMMTRACE(DEP))**2)
      IF(DEQPS.LT.1.D-16)THEN
         DO IJ=1,6
            DEP(IJ) = PZERO
         ENDDO
         DEQPS = PZERO
      ENDIF
 30   CONTINUE
      SV(KEQPS)= EQPS+DEQPS



C     Update temperature
      DTE= -SV(KT)*GP*TRDE      ! elastic tmpr increment
      DTP= UI(IPTQC)*DMMENINC(QSTAUP,DEP,R0)/CV ! plastic tmpr increment
      SV(KT)=MIN(TM,SV(KT)+DTE+DTP)
      SV(KF4)= DMMENINC(QSTAUP,DEP,R0)

C     Update quasistatic ISVs
      SV(KR)= R*EXP(-TRDE)
      SV(KRJ)= R0/SV(KR)
      SV(KEU)= SV(KEU) + DMMENINC(QSTAUP,DE,R0)
      SV(KCS)= SQRT(BM/R0)
      SV(KEVOL)= EVOL+DMMTRACE(DE)
      SV(KEQPV)= SV(KEQPV)+DMMTRACE(DEP)

C     Dynamic overstress TAUP
      CALL OVERSTRESS(DTARG,SV,EQDOT,TAUN,DTAUT,QSTAUN,QSTAUP,TAUP)

C     State variables at end of step
      SV(KROOTJ2)= DMMROOTJ2(TAUP)/SV(KRJ)
      SV(KI1)= DMMTRACE(TAUP)/SV(KRJ)
      SV(KEQDOT)= EQDOT
      SV(KEVOL)= SVARG(KEVOL)+DMMTRACE(D)*DTARG
      CALL DMMGETDEV(SV(KE+1),EDEV)
      SV(KAM)= STIFFANISOMEAS(BM,TWOG,TWOG1,DMMDBD(EDEV,EDEV))

C     Update energy to include work increment from dynamic stress
      SV(KEU)= SV(KEU)+(DMMENINC(TAUP,D,R0)-DMMENINC(QSTAUP,D,R0))*DTARG

C     Update passed arguments
c     Dynamic and quasistatic stress arrays
      DO IJ=1,6
         SIGARG(IJ)= TAUP(IJ)/SV(KRJ)
         SV(KQSSIG+IJ)= QSTAUP(IJ)/SV(KRJ)
      END DO

C     State variable array
      DO IROW=1,NINSV
         SVARG(IROW)= SV(IROW)
      ENDDO

      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMCHK(UI,GC,DC)
C
C***********************************************************************
C     REQUIRED MIG DATA CHECK ROUTINE
C     Checks validity of user inputs for DMM model.
C     Sets defaults for unspecified user input.
C     Adjusts user input to be self-consistent.
C
C     input
C     -----
C       UI: user input as read and stored by host code.
C
C       Upon entry, the UI array contains the user inputs EXACTLY
C       as read from the user.  These inputs must be ordered in the
C       UI array as indicated in the file kmmpnt.Blk.
C       See kmmpnt.Blk for parameter definitions and keywords.
C
C       DC: Not used with this model
C
C    Other output
C    ------------
C       GC: Not used with this model
C       DC: Not used with this model
C       Because GC and DC are not used, you may call this routine
C       with a line of the form "CALL DMMCHK(UI,UI,UI)"
C
C***********************************************************************
C  author:  Rebecca Brannon
C
C  yymmdd:usernam:   m o d i f i c a t i o n
C  ---------------------------------------------------------------------
C  090713:tim fuller:Created original data check
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

C
      DIMENSION UI(*), GC(*), DC(*)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)
      PARAMETER (HUGE=1.D80)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
C
C  ...local
      CHARACTER*6 IAM
      PARAMETER( IAM = 'DMMCHK' )
      DIMENSION A(8)
      LOGICAL DEJAVU
      DATA DEJAVU/.FALSE./

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     avoid compiler warning of unused dummy arguments
      dum=gc(1)
c
      DEJAVU=(NINT(UI(IPDEJAVU)).NE.0)
C
      IF(.NOT.DEJAVU)THEN
        CALL LOGMES('############# DMM build 100222')
      ENDIF
C
C     Check validity of user-supplied model parameters
C
      IF(UI(IPR0).LT.PZERO)CALL FATERR(IAM,'R0 must be positive')
      IF(UI(IPB0).LE.PZERO)THEN
         IF(UI(IPC0).LE.PZERO)THEN
            CALL FATERR(IAM,'B0 or C0 must be positive')
         ELSE
            UI(IPB0)=UI(IPR0)*UI(IPC0)*UI(IPC0)
         ENDIF
      ENDIF
      IF(UI(IPG0).LE.PZERO)CALL FATERR(IAM,'G0 must be positive')
      IF(UI(IPA1).LT.PZERO)CALL FATERR(IAM,'A1 must be nonnegative')
      IF(UI(IPA2).LT.PZERO)CALL FATERR(IAM,'A2 must be nonnegative')
      IF(UI(IPA3).LT.PZERO)CALL FATERR(IAM,'A3 must be nonnegative')
      IF(UI(IPA4).LT.PZERO)CALL FATERR(IAM,'A4 must be nonnegative')
      IF(UI(IPA4PF).LT.PZERO)CALL FATERR(IAM,'A4PF must be nonnegative')

c     Set defaults
      IF(UI(IPA4PF).EQ.PZERO)UI(IPA4PF)=UI(IPA4)  !default A4PF = A4

c     Convert derivative of G w.r.t. pressure to derivative w.r.t.
c     elastic volume change
      IF(.NOT.DEJAVU)THEN
         IF(UI(IPIDK).NE.1)UI(IPB1) = -UI(IPB1)*UI(IPB0)
         IF(UI(IPG1).LE.PZERO)THEN
            UI(IPG1)=PZERO
            UI(IPAN)=PZERO
         ELSE
            IF(UI(IPIDG).NE.1)UI(IPG1) = -UI(IPG1)*UI(IPB0)
         ENDIF
      ENDIF

C     Get Drucker Prager coefficients if compressive strength is given
      IF(UI(IPA1).LT.HUGE.AND.UI(IPSC).GT.PZERO)THEN
         IF(UI(IPA4).LE.PZERO)THEN
            UI(IPA4)=ROOT3*(UI(IPSC)-UI(IPA1))/(UI(IPSC)+UI(IPA1))
c     UI(IPA1)=PTWO/ROOT3*UI(IPSC)*UI(IPA1)/(UI(IPSC)+UI(IPA1))
         ENDIF
      ENDIF
      IF(UI(IPT0).LE.PZERO)UI(IPT0)=298.D0
      IF(UI(IPR0).LE.PZERO)UI(IPR0)=PONE
      IF(UI(IPTM).LE.PZERO)UI(IPTM)=1.D99
      IF(UI(IPCV).LE.PZERO)UI(IPCV)=PONE
      IF(UI(IPGP).LE.PZERO)UI(IPGP)=PZERO
      IF(UI(IPXP).LE.PZERO)UI(IPXP)=PONE
      IF(UI(IPTQC).LE.PZERO)UI(IPTQC)=PONE

      IF(NINT(UI(IPIDK)).EQ.0)THEN
         CS= UI(IPC0)
         S1= UI(IPS1)
         GP= UI(IPGP)
         CALL KEREOSMGJ(CS,S1,0.D0,GP,0.D0,1.D0,1.D0,A(1),A(6))
         DC(1) = A(1)
         DC(2) = A(2)
         DC(3) = A(3)
         DC(4) = A(4)
         DC(5) = A(5)
         DC(6)= 1.D0
         DC(7)= 0.D0
         DC(8)= 0.D0
         DC(9)= 0.D0
         DC(10)= 0.D0
         DC(11)= 1.D0
         DC(12)= 0.D0
         DC(13)= UI(IPR0)/(1.D0-1.D0/MAX(1.000001D0,S1))
      ENDIF

      IF(.NOT.DEJAVU)THEN
         IF(UI(IPA4PF).EQ.UI(IPA4))THEN
            CALL LOGMES(IAM//': FYI, This material is associative')
         ELSE
            CALL LOGMES(IAM//': FYI, This material is nonassociative')
         ENDIF
        CALL LOGMES('############# DMM data check complete')
        UI(IPDEJAVU)=PONE
        IF(PTHREE*UI(IPB0).LT.PTWO*UI(IPG0))THEN
           CALL LOGMES
     & (IAM//' Warning: neg Poisson (to avoid warning, set 3*B0>2*G0)')
        ENDIF
      ENDIF
      RETURN
      END !SUBROUTINE DMMCHK
C
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMRXV(UI,GC,DC,
     &  NX, NAMEA, KEYA, RINIT, RDIM, IADVCT, ITYPE)
C**********************************************************************
C     REQUESTED EXTRA VARIABLES FOR KAYENTA
C
C     This subroutine creates lists of the internal state variables
C     needed for DMM. This routine merely sends a
C     LIST of internal state variable requirements back to the host
C     code.   IT IS THE RESPONSIBILITY OF THE HOST CODE to loop over
C     the items in each list to actually establish necessary storage
C     and (if desired) set up plotting, restart, and advection
C     control for each internal state variable.
C
C     called by: host code after all input data have been checked
C
C     input
C     -----
C          UI = user input array
C          GC = unused for this model (placeholder)
C          DC = unused for this model (placeholder)
C
C     output
C     ------
C          NX = number of extra variables                    [DEFAULT=0]
C       NAMEA = single character array created from a string array
C               (called NAME) used locally in this routine to register
C               a descriptive name for each internal state variable.
C        KEYA = single character array created from a string array
C               (called KEY) used locally in this routine to register
C               a plot keyword for each internal state variable.
C          | Note: NAMEA and KEYA are created from the local variables |
C          | NAME and KEY by calls to the subroutine TOKENS, which     |
C          | is a SERVICE routine presumed to ALREADY exist within the |
C          | host code (we can provide this routine upon request).     |
C          | "NAME" is a fortran array of strings. "NAMEA" is a one    |
C          | dimensional array of single characters. For readability,  |
C          | most of this subroutine writes to the NAME array. Only at |
C          | the very end is NAME converted to NAMEA by calling a      |
C          | the utility routine TOKENS. The KEY array is similarly    |
C          | converted to KEYA.  These conversions are performed       |
C          | because host codes written in C or C++ are unable to      |
C          | process FORTRAN string arrays. Upon request, we can       |
C          | provide a utility routine that will convert BACK to       |
C          | FORTRAN string arrays if your host code is FORTRAN.       |
C          | Likewise, we can provide a C++ routine that will allow    |
C          | parsing the single-character arrays to convert them back  |
C          | to strings if your code is C++. Alternatively, you can    |
C          | simply ignore the NAMEA and KEYA outputs of this routine  |
C          | if your host code does not wish to establish plotting     |
C          | information.                                              |
C
C       RINIT = initial value for each ISV               [DEFAULT = 0.0]
C        RDIM = physical dimension exponents             [DEFAULT = 0.0]
C               This variable is dimensioned RDIM(7,*) for the 7 base
C               dimensions (and * for the number of extra variables):
C
C                      1 --- length
C                      2 --- mass
C                      3 --- time
C                      4 --- temperature
C                      5 --- discrete count
C                      6 --- electric current
C                      7 --- luminous intensity
C
C                Suppose, for example, that an ISV has units of stress.
C                Dimensionally, stress is length^(1) times mass^(-1)
C                times time^(-2). Therefore, this routine would return
C                1.0, -1.0, and -2.0 as the first three values of the
C                RDIM array. Host codes that work only in one unit
C                set (e.g., SI) typically ignore the RDIM output.
C
C      IADVCT = advection option                           [DEFAULT = 0]
C                    = 0 advect by mass-weighted average
C                    = 1 advect by volume-weighted average
C                    = 2 don't advect
C            The advection method will often be ignored by host codes.
C            It is used for Eulerian codes and for Lagrangian codes that
C            re-mesh (and therefore need guidance about how to "mix"
C            internal state variables). Note: a value of 2 implies that
C            the ISV is output only.
C
C        ITYPE = variable type (see migtionary preface)    [DEFAULT = 1]
C                  1=scalar
C                  6=2nd-order symmetric tensor
C        The component ordering for ITYPE=6 is 11, 22, 33, 12, 23, 31.
C        Consequently, the 11 component is the first one to be requested
C        in tensor lists, and its IFLAG is set to 6. To indicate that
C        subsequent ISVs are the remaining components of the same tensor,
C        the next five ISVs are given an IFLAG value of -6.
C        Host codes that don't change basis can ignore ITYPE.
C
C***********************************************************************
C
C  author:  Rebecca Brannon
C
C    who    yymmdd  M O D I F I C A T I O N
C  -------  ------  ----------------------------------------------------
C  rmbrann  030809  Created original extra variable routine
C

C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C This include block defines parameters used in Kayenta.
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers
C         brannon@mech.utah.edu
C         oestrac@sandia.gov
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.

C
C Just numbers
      PARAMETER (
     *   PZERO  = 0.0D0        ,
     *   P1MEPS = 1.0D0 - 1.0D-10,
     *   PFORTH = 0.25D0       ,
     *   PTHIRD = 0.3333333333333333333333333333333333333333333333D0,
     *   PHALF  = 0.5D0        ,
     *   PONE   = 1.0D0        ,
     *   P4THIRD= 1.3333333333333333333333333333333333333333333333D0,
     *   P10SEVENTH=10.0D0/7   ,
     *   P3HALF = 1.5D0        ,
     *   PTWO   = 2.0D0        ,
     *   PTHREE = 3.0D0        ,
     *   PFOUR  = 4.0D0        ,
     *   PFIVE  = 5.0D0        ,
     *   PSIX   = 6.0d0        ,
     *   PTEN   = 10.0D0       ,
     *   PTWELVE= 12.0D0       ,
     *   P5TEEN = 15.0D0
     *   )
      PARAMETER (
     *   P20    = 2.0D1        ,
     *   BIGNUM = 1.0D30       ,
     *   P14TH  = 0.25D0       ,
     *   P34TH  = 0.75D0       ,
     *   P1P05  = 1.05D0       ,
     *   P1P1   = 1.1D0        ,
     *   P1P14  = 1.14D0       ,
     *   P1P2   = 1.2D0        ,
     *   P1P5   = 1.5D0        ,
     *   P2P1   = 2.1D0        ,
     *   P2P5   = 2.5D0        ,
     *   P3P5   = 3.5D0        ,
     *   P3P524 = 3.524D0
     *   )
      PARAMETER (
     *   P0005  = 0.0005D0     ,
     *   P0032  = 0.0032D0     ,
     *   P001   = 0.001D0      ,
     *   P002   = 0.002D0      ,
     *   P007   = 0.007D0      ,
     *   POINT01= 0.01D0       ,
     *   POINT02= 0.02D0       ,
     *   POINT03= 0.03D0       ,
     *   POINT05= 0.05D0       ,
     *   POINT06= 0.06D0       ,
     *   POINT1 = 0.1D0        ,
     *   P125   = 0.125D0      ,
     *   POINT15= 0.15D0
     *   )
      PARAMETER (
     *   POINT33= 0.33D0       ,
     *   POINT4 = 0.4D0        ,
     *   POINT49= 0.49D0       ,
     *   POINT501= 0.501D0     ,
     *   POINT55= 0.55D0       ,
     *   POINT9 = 0.9D0        ,
     *   POINT95= 0.95D0       ,
     *   POINT99= 0.99D0       ,
     *   POINT999= 0.999D0     ,
     *   POINT9999=0.9999D0    ,
     *   PSIXTH = 1.0D0/6      ,
     *   TWOTHD = 2.0D0/3
     *   )
      PARAMETER (
     *   P5SIXTHS= 5.0D0/6     ,
     *   S9THS  = 16.0D0/9     ,
     *   SIXTIETH=1.0D0/60     ,
     *   P1M30  = 1.0D-30      ,
     *   P1M20  = 1.0D-20      ,
     *   P1M15  = 1.0D-15      ,
     *   P1M9   = 1.0D-9       ,
     *   P1M7   = 1.0D-7       ,
     *   P1E3   = 1.0D3        ,
     *   P1E4   = 1.0D4        ,
     *   P1E6   = 1.0D6        ,
     *   P1E10  = 1.0D10       ,
     *   P1E12  = 1.0D12       ,
     *   P1E15  = 1.0D15       ,
     *   P1E20  = 1.0D20
     *   )
C
c---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C Particular numbers
      PARAMETER (
     *   ROOT2 = 1.414213562373095048801688724209698078569671875377D0,
     *   ROOTO2 = PONE/ROOT2,
     *   TOOR2 = PONE/ROOT2,
     *   ROOT3=0.173205080756887729352744634150587236694281D1,
     *   SQRT3  = 1.73205080756887729352744634150587236694280525381D0,
     *   TWOORT3= PTWO/SQRT3,
     *   SQRT2O100 = ROOT2/1.0d-2,
     *   ROOT23 = 0.81649658092772603273242802490196379732198249355D0,
     *   ROOT32 = PONE/ROOT23,
     *   COS120 = -PHALF,
     *   SIN120 = SQRT3/PTWO,
     *   TOOR3=0.577350269189625764509148780501957455647602D0
     *   )
C
C All kinds of pi
      PARAMETER (
     *   PI     = 3.1415926535897932384626433832795028841971693993D0,
     *   PIO2   = PI/PTWO,
     *   TWOPI  = PTWO*PI,
     *   RADDEG = PI/180,
     *   DEGRAD = 180/PI,
     *   TWOTHDPI = TWOPI/PTHREE
     *   )
C
C Tolerances
      PARAMETER (
     *   TOL1M3 = 1.0D-3,
     *   TOL1M4 = 1.0D-4,
     *   TOL1M5 = 1.0D-5,
     *   TOL3M6 = 3.0D-6,
     *   TOL1M6 = 1.0D-6,
     *   TOL1M7 = 1.0D-7,
     *   TOL1M8 = 1.0D-8,
     *   TOL1M9 = 1.0D-9,
     *   TOL1M10= 1.0D-10,
     *   TOL1M12= 1.0D-12,
     *   TOL1M14= 1.0D-14,
     *   TOL1M20= 1.0D-20
     *   )
C
C Flags
      PARAMETER (
     *   UNDEF  = 1.23456D-7,
     *   NOTDEF = -654321
     *   )
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

C
      INTEGER MMCN,MMCK,MNUNIT,MMCNA,MMCKA
      PARAMETER (MMCN=50,MMCK=10,MNUNIT=7)
      PARAMETER (MMCNA=NDMMISV*MMCN,MMCKA=NDMMISV*MMCK)
C
      CHARACTER*(MMCN) NAME(NDMMISV)
      CHARACTER*(MMCK) KEY(NDMMISV)
      CHARACTER*1 NAMEA(*), KEYA(*)
      DIMENSION IADVCT(*),ITYPE(*)
      DIMENSION UI(*), GC(*), DC(*), RINIT(*), RDIM(7,*)
      LOGICAL SERIAL
      DATA SERIAL/.FALSE./
C
      CHARACTER*6 IAM
      PARAMETER(IAM='DMMRXV')

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMRXV
*     These next lines are only to avoid compiler warnings
      dum=dc(1)
      dum=gc(1)
      SERIAL= UI(IPTEST).NE.0
      VINIT=0.D0
*
      CALL LOGMES('############# Requesting DMM variables')
C
C
C
C
C     ***********************************************************
C     ***   SET INITIAL VALUES FOR STATE VARIABLES            ***
C     ***                                                     ***
C     ***   The first argument is the user input (previously  ***
C     ***   obtained by reading an input set). The next       ***
C     ***   argument, RINIT, is the list of initial values    ***
C     ***   for each of the internal state variables listed   ***
C     ***   below.                                            ***
C     ***                                                     ***
C     ***********************************************************
C
C
C     For now, the ISVs will be initialized using an assumption
C     that the initial stress is zero.  Host codes that support a
C     nonzero initial stress field will need to call
C     ISOTROPIC_DMMMATERIAL_INIT with the
C     initial stress in each element to reset the
C     ISVs appropriately.
C
      DO ISV=1,NDMMISV
         RINIT(ISV)=PZERO
      ENDDO
C
C
      NX=0
C
C----------------------------------------------------------------------1
      NX=NX+1                                                    !KEQDOT
      IF(NX.NE.KEQDOT)CALL BOMBED(IAM//' KEQDOT pointer wrong')
      NAME(NX) = 'Magnitude of the total strain rate'
      KEY(NX) = 'EQDOT'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)=  PZERO  ! units: 1/time
      RDIM(2,NX)=  PZERO
      RDIM(3,NX)= -PONE
      RINIT(NX)=VINIT
C----------------------------------------------------------------------2
      NX=NX+1                                                       !kI1
      IF(NX.NE.KI1)CALL BOMBED(IAM//' KI1 pointer wrong')
      NAME(NX)='First invariant (trace) of stress'
C     Note: stress is positive in tension. Therefore, I1 is typically
C     negative in compression. The mechanical pressure (positive in
C     compression) and mean stress (positive in tension) are given by
C           pressure    = - I1/3
C           mean stress = + I1/3
      KEY(NX)='I1'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
      RINIT(NX)=VINIT
C----------------------------------------------------------------------3
      NX=NX+1                                                   !KROOTJ2
      IF(NX.NE.KROOTJ2)CALL BOMBED(IAM//' KROOTJ2 pointer wrong')
      NAME(NX)='Square root of second stress invariant, SQRT(J2)'
C     The second stress invariant is given by
C            J2 = (1/2) trace(S.S)
C     where S is the stress deviator. Therefore,
C       ROOTJ2 = magnitude(S)/sqrt(2)
      KEY(NX)='ROOTJ2'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
      RINIT(NX)=VINIT
C----------------------------------------------------------------------4
      NX=NX+1                                                     !KEQPS
      IF(NX.NE.KEQPS)CALL BOMBED(IAM//' KEQPS pointer wrong')
      NAME(NX)='Equivalent plastic SHEAR strain'
C     EQPS is the time integral of
C           SQRT[2.0]*magnitude[deviator[strain rate]
C     It is defined in this way to be conjugate to ROOTJ2.  In other
C     words, the inner product of the stress deviator with the strain
C     rate is equal to ROOTJ2 times the rate of EQPS.
      KEY(NX)='EQPS'
      IADVCT(NX)=0        ! input and output
C     itype=1  (scalar)
C     RDIM=default        ! units:  dimensionless
      RINIT(NX)=VINIT
C----------------------------------------------------------------------6
      NX=NX+1                                                     !KEVOL
      IF(NX.NE.KEVOL)CALL BOMBED(IAM//' KEVOL pointer wrong')
      NAME(NX)='volumetric strain'
      KEY(NX)='EVOL'
      IADVCT(NX)=0        ! output only
      ITYPE(NX)=1
C     itype=1  (scalar)
      RINIT(NX)=VINIT
C-----------------------------------------------------------------------7
      NX=NX+1                                                      !KTMPR
      IF(NX.NE.KT)CALL BOMBED(IAM//' KTMPR pointer wrong')
      NAME(NX)='Temperature'
      KEY(NX)='TMPR'
      IADVCT(NX)=1        ! input/output
      RDIM(4,NX)=  PONE
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPT0)
      ENDIF
C-----------------------------------------------------------------------8
      NX=NX+1                                                     !KSNDSP
      IF(NX.NE.KCS)CALL BOMBED(IAM//'1 KSNDSP pointer wrong')
      NAME(NX)='sound speed'
      KEY(NX)='SNDSP'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX)=  PONE
      RDIM(3,NX)= -PONE
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPC0)
      ENDIF
C-----------------------------------------------------------------------9
      NX=NX+1                                                       !KRHO
      IF(NX.NE.KR)CALL BOMBED(IAM//' KRHO pointer wrong')
      NAME(NX)='Density'
      KEY(NX)='DENS'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX) = -PTHREE
      RDIM(2,NX) =  PONE
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPR0)
      ENDIF
C----------------------------------------------------------------------10
      NX=NX+1                                                     !KENRGY
      IF(NX.NE.KEU)CALL BOMBED(IAM//' KENRGY pointer wrong')
      NAME(NX)='Energy'
      KEY(NX)='ENRGY'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX) =  PTWO
      RDIM(2,NX) =  PONE
      RDIM(3,NX) = -PTWO
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=UI(IPT0)*UI(IPCV)
      ENDIF
C----------------------------------------------------------------------13
      NX=NX+1                                                      !KEOS1
      IF(NX.NE.KRJ)CALL BOMBED(IAM//' KRJ pointer wrong')
      NAME(NX)='Jacobian of deformation'
      KEY(NX)='JACOBIAN'
      IF(SERIAL)THEN
         RINIT(NX)=VINIT
      ELSE
         RINIT(NX)=PONE
      ENDIF
      IADVCT(NX)=1
C----------------------------------------------------------------------14
      NX=NX+1                                                      !KEOS2
      IF(NX.NE.KAM)CALL BOMBED(IAM//' KAM pointer wrong')
      NAME(NX)='Measure of anisotropy'
      KEY(NX)='AM'
      IADVCT(NX)=1
      RINIT(NX)=0.D0
C----------------------------------------------------------------------15
      NX=NX+1                                                      !KEOS3
      IF(NX.NE.KEQPV)CALL BOMBED(IAM//' KEQPV pointer wrong')
      NAME(NX)='plastic volume strain'
      KEY(NX)='EQPV'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------16
      NX=NX+1                                                      !KEOS4
      IF(NX.NE.KF4)CALL BOMBED(IAM//' KF4 pointer wrong')
      NAME(NX)='Free EOS ISV 4'
      KEY(NX)='F4'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------17
      NX=NX+1                                                    !KQSSIGXX
      IF(NX.NE.KQSSIGXX)CALL BOMBED(IAM//' KQSSIGXX pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGXX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------18
      NX=NX+1                                                    !KQSSIGYY
      IF(NX.NE.KQSSIGYY)CALL BOMBED(IAM//' KQSSIGYY pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGYY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------19
      NX=NX+1                                                    !KQSSIGZZ
      IF(NX.NE.KQSSIGZZ)CALL BOMBED(IAM//' KQSSIGZZ pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGZZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------20
      NX=NX+1                                                    !KQSSIGXY
      IF(NX.NE.KQSSIGXY)CALL BOMBED(IAM//' KQSSIGXY pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='EQSSIGXY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------21
      NX=NX+1                                                   !KQSSIGYZ
      IF(NX.NE.KQSSIGYZ)CALL BOMBED(IAM//' KQSSIGYZ pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGYZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------22
      NX=NX+1                                                   !KQSSIGZX
      IF(NX.NE.KQSSIGZX)CALL BOMBED(IAM//' KQSSIGZX pointer wrong')
      NAME(NX)='Quasistatic stress'
      KEY(NX)='QSSIGZX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------23
      NX=NX+1                                                    !KEXX
      IF(NX.NE.KEXX)CALL BOMBED(IAM//' KEXX pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EXX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------24
      NX=NX+1                                                    !KEYY
      IF(NX.NE.KEYY)CALL BOMBED(IAM//' KEYY pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EYY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------25
      NX=NX+1                                                    !KEZZ
      IF(NX.NE.KEZZ)CALL BOMBED(IAM//' KEZZ pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EZZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------26
      NX=NX+1                                                    !KEXY
      IF(NX.NE.KEXY)CALL BOMBED(IAM//' KEXY pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EEXY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------27
      NX=NX+1                                                   !KEYZ
      IF(NX.NE.KEYZ)CALL BOMBED(IAM//' KEYZ pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EYZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------28
      NX=NX+1                                                   !KEZX
      IF(NX.NE.KEZX)CALL BOMBED(IAM//' KEZX pointer wrong')
      NAME(NX)='Elastic strain'
      KEY(NX)='EZX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------29
      NX=NX+1                                                   !KEJ2
      IF(NX.NE.KEJ2)CALL BOMBED(IAM//' KEJ2 pointer wrong')
      NAME(NX)='j2 invariant of elastic strain deviator'
      KEY(NX)='EDEVJ2'
      IADVCT(NX)=1
      RINIT(NX)=VINIT


C     ##################################################################
      IF(NX.GT.NDMMISV)CALL BOMBED
     & ('INCREASE NDMMISV IN ROUTINE DMMRXV AND IN DATA FILE')
C     convert NAME and KEY to character streams NAMEA and KEYA
C     (See note about TOKENS in prolog of this routine)
      CALL TOKENS(NX,NAME,NAMEA)
      CALL TOKENS(NX,KEY ,KEYA )
C     CALL LOGMES('############# exiting DMMRXV')
      RETURN
      END !SUBROUTINE DMMRXV
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMGETDEV(A,B)
C***********************************************************************
C     PURPOSE: Return deviatoric part of second order tensor
C
C input
C -----
C    A: Second order tensor
C
C output
C -----
C    B: Deviatoric part of A
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C............................................................parameters
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTHREE=3.D0,PTHIRD=PONE/PTHREE)
C.................................................................passed
      DIMENSION A(6),B(6)
C..................................................................local
      DIMENSION S(6)
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      S(1) = PTHIRD*(A(1)+A(2)+A(3))
      DO IJ=1,3
         B(IJ)=A(IJ)-S(1)
      ENDDO
      DO IJ=4,6
         B(IJ)=A(IJ)
      ENDDO
C     Check that deviatoric
      S(1)=(B(1)+B(2)+B(3))*PTHIRD
      IF(S(1).NE.PZERO)THEN
         B(1)=B(1)-S(1)
         B(2)=B(2)-S(1)
         B(3)=-(B(1)+B(2))
      ENDIF
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMPAR_PUT(UI)
C***********************************************************************
C     PURPOSE: This routine transfers property values in the "UI" array
C     into the DMMPROP common blocks and computes derived constants
C
C input
C -----
C    UI: property array
C
C output
C -----
C    /DMMPROPL/
C    /DMMPROPI/
C    /DMMPROPR/
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C.............................................................parameters
      PARAMETER (PZERO=0.0D0,PTWO=0.2D1,PTHREE=0.3D1,PSIX=6.0D0)
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
      PARAMETER ( ROOT23=0.816496580927726032732428024901963797321982D0)
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

C.................................................................common
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
C.................................................................passed
      DIMENSION UI(*)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMPAR_PUT
C     Logicals
      ANISO=(NINT(UI(IPAN)).NE.0)
      RATEDEP=(UI(IPT1).NE.PZERO)
      WANTSELASTIC=(UI(IPA1).GT.1D90)
C     Properties
      B0= UI(IPB0)
      B1= UI(IPB1)
      B2= UI(IPB2)
      G0= UI(IPG0)
      IF(ANISO)THEN
         G1= PTHIRD*UI(IPG1)
         TWOG1=PTWO*G1
      ELSE
         G1= UI(IPG1)
         TWOG1=PZERO
      ENDIF
      G2= UI(IPG2)
      G3= UI(IPG3)
      PR= (PTHREE*B0-PTWO*G0)/(PSIX*B0+PTWO*G0)
      A1= TOOR3*UI(IPA1)
      A2= ROOT3*UI(IPA2)
      A3= TOOR3*UI(IPA3)
      A4G= ROOT3*UI(IPA4PF)
      A4= UI(IPA4)
      R0= UI(IPR0)
      T0= UI(IPT0)
      TM= UI(IPTM)
      XP= UI(IPXP)
      C0= UI(IPC0)
      CV= UI(IPCV)
      GP= UI(IPGP)
      S1= UI(IPS1)
      T1= UI(IPT1)
      T2= UI(IPT2)
      T3= UI(IPT3)
      T4= UI(IPT4)
      IDK=NINT(UI(IPIDK))
      IDG=NINT(UI(IPIDG))
      RETURN
      END ! SUBROUTINE DMMPAR_PUT

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMLODE(A,AZ,AS,AES)
C***********************************************************************
C     PURPOSE: Return Lode components of second order tensor A
C
C input
C -----
C    A: Second order tensor
C
C output
C -----
C    AZ:  Z component of A
C    AS:  S component of A
C    AES: Basis tensor of deviatoric part of A
C

C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C.............................................................parameters
      PARAMETER (TOL1M10=1.D-10,HUGE=1.D90)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
C.................................................................common
C.................................................................passed
      DIMENSION A(6),AES(6)
C...............................................................external
C..........................................................local scalars
C...........................................................local arrays
      DIMENSION S(6)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMLODE
      AZ = TOOR3*DMMTRACE(A)
      CALL DMMGETDEV(A,S)
      AS = DMMTMAG(S)
      DUM=AS
      IF(AS.LT.TOL1M10)DUM=HUGE
      DO IJ=1,6
         AES(IJ) = S(IJ)/DUM
      ENDDO
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMM_MODULI(UI,U,R,T,EV,BETA2,EQPS,RI1,RTJ2,BM,SM)
C***********************************************************************
C     PURPOSE: Determines non-linear elastic properties
C
C input
C -----
C    UI: UI array
C    U:  Internal energy
C    R:  Density
C    T:  Temperature
C    EV: Elastic volume strain
C    BETA2: Second invariant of elastic strain tensor
C    EQPS:  Equivalent plastic strain
C    RI1:   First invariant of stress tensor
C
C output
C -----
C    BM:  Bulk modulus
C    SM:  Shear modulus
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C............................................................parameters
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0,PTEN=10.D0)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
C.................................................................common
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
C.................................................................passed
      DIMENSION UI(*)
C..................................................................local
      DIMENSION S(9),UIMG(22)
C...................................................................data
C...............................................................external
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DMM_MODULI
C     avoid compiler warnings of unused dummy arguments
      dum=beta2-eqps

C     Compute bulk modulus
      IF(IDK.EQ.0)THEN
C     Kerley EOS to compute sound speed
         P= -PTHIRD*RI1
         CALL EOSARRAYREORDER(UI,UIMG)
         CALL KEREOSMGV(1,1,UIMG,UI,UI(IPDC1),R,U,0,0,P,T,CS,S)

      ELSEIF(IDK.EQ.1)THEN
         CS = SQRT((B0 + B1*EXP(-B2/MAX(-RI1,1.D-40)))/R0)

      ELSEIF(IDK.EQ.2)THEN
         IF(EV.LT.PZERO)THEN
            CS= SQRT((B0 + B1*EV + PHALF*B2*EV**2)/R0)
         ELSE
            CS= SQRT(B0/R0)
         ENDIF

      ENDIF
      BM= R0*CS**2

C     Compute shear modulus
      IF(IDG.EQ.0)THEN
C     Shear modulus from SGC formula with elastic-plastic coupling
         RJFAC= (R0/R)**PTHIRD
         C= -0.D0
         SM= MAX(G0,G0 + G1*(EV*RJFAC+C*EQPS) + G2*(T-300.D0))

      ELSEIF(IDG.EQ.1)THEN
         SM= G0*(PONE-G1*EXP(-G2*RTJ2))/(PONE-G1)

      ELSEIF(IDG.EQ.2)THEN
         SM= MAX(G0,G0 + G1*EV + G2*(T-300.D0))

      ELSEIF(IDG.EQ.3)THEN
C     Shear modulus from elasticity relations
         SM= PTHREE*BM*(PONE-PTWO*PR)/(PTWO*(PONE+PR))
      ENDIF

C     Set shear modulus to zero if melt temp is exceeded
      IF(T.GE.TM)THEN
         CALL LOGMES('Melt temp reached, setting G=0')
         SM= G0
      ENDIF


      RETURN

      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE OVERSTRESS(DT,SV,EQDOT,SIGOLD,DSIGT,QSSIGOLD,QSSIG,SIG)
C***********************************************************************
C     PURPOSE: Compute Duvaut-Lions overstress
C
C input
C -----
C    DT:       Timestep
C    SV:       State variable array
C    EQDOT:    Equivalent strain rate
C    SIGOLD:   Stress at beginning of timestep
C    DSIGT:    Trial stress increment
C    QSSIGOLD: Quasistatic stress at beginning of timestep
C    QSSIG:    Current quasistatic stress
C
C output
C -----
C    SIG: Dynamic stress at end of step
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|ported/modified Kayenta routine for DMM
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C.............................................................parameters
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PSIX=6.D0)
      PARAMETER (TOL1M20=1.D-20)
      PARAMETER (EUNDERFLOW=-34.53877639491D0*PONE)
C.................................................................common
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
C.................................................................passed
      DIMENSION SIGOLD(6),QSSIGOLD(6),QSSIG(6),DSIGT(6),SIG(6),SV(*)
C..................................................................local
      DIMENSION SIGV(6)
C...................................................................data
C...............................................................external
C....................................................statement functions
C
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc OVERSTRESS
      IF(.NOT.RATEDEP)THEN
         DO IJ=1,6
            SIG(IJ)= QSSIG(IJ)
         ENDDO
         DO I=1,NDMMISV
            SV(I)= SV(I)
         ENDDO
      ENDIF
      GO TO 777
      call bombed('rate dependence has not been set up')

      dum=sv(1)
C     Set value for relaxation time. The value of tau is set based on
C     data being available for sig_over/sig_qs vs. log(eqdot) for
C     uniaxial strain. It is assumed that the data show a bilinear
C     relationship.


      !RNUM = abs(CHI*KMMDBD(SIG,D(1,NBLK),6))
      !DNOM = KMMDBD(PPPD,D(1,NBLK),6)*KMMDBD(QQQD,D(1,NBLK),6)
      !DUM=RNUM/DNOM
      IF(T2.EQ.PZERO)THEN
         TCHAR = T1
      ELSEIF(T3.EQ.PZERO.OR.EQDOT.LT.T3)THEN
         call bombed('Tim: dum is being used before initialized')
         TCHAR = DUM*(T1+T2*LOG(EQDOT)-PONE)
      ELSE
         call bombed('Tim: dum is being used before initialized')
         TCHAR = DUM*(T1+(T2-T4)*LOG(T3)+T4*LOG(EQDOT)-PONE)
      ENDIF

      IF(TCHAR.GT.TOL1M20)THEN
C     Manage underflow (which will result in overflow for RHI):
         RAT=DT/TCHAR
         IF(RAT.GT.ABS(EUNDERFLOW))THEN
C     ...extremely large timesteps (large value of rat)
C     (this usually also means low rates -- see comment below)
            RHI=PZERO
            RLO=PONE
            IF(RAT.LT.1.D40)THEN
               RMID=-PONE/RAT
            ELSE
               RMID=PZERO
            ENDIF
         ELSEIF(RAT.GT.1.D-3)THEN
C     ...intermediate timesteps
            RHI=(PONE-EXP(-RAT))/RAT
            RLO=PONE-RHI
            RMID=EXP(-RAT)-RHI
         ELSE
C     ...extremely small timesteps (small value of rat)
C     (this usually means high rates -- see comment below)
C     Use a series expansions of the above expressions
            RLO=RAT*(PHALF-RAT/PSIX)
            RHI=PONE-RLO
            RMID=RAT*(RAT*PTHIRD-PHALF)
         ENDIF
         DO IJ=1,6
            SIGV(IJ) = SIGOLD(IJ) + DSIGT(IJ)
            SIG(IJ) = RHI*SIGV(IJ) + RLO*QSSIG(IJ)
     &           + RMID*(SIGOLD(IJ) - QSSIGOLD(IJ))
         ENDDO
      ENDIF
 777  CONTINUE
      RETURN
      END


C***********************************************************************
C                       R M M  F U N C T I O N S
C***********************************************************************
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTMAG(A)
C     Purpose: Magnitude of second order tensor A
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(6)
      DMMTMAG=SQRT(DMMDBD(A,A))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMDBD(A,B)
C     Purpose: Double dot product of second order tensors A and B
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (PTWO=2.D0)
      DIMENSION A(6),B(6)
      DMMDBD=     A(1)*B(1)+A(2)*B(2)+A(3)*B(3)
     $    + PTWO*(A(4)*B(4)+A(5)*B(5)+A(6)*B(6))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTRACE(A)
C     Purpose: Trace of second order tensor A
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(6)
      PARAMETER (TOL=1.D-15,PZERO=0.D0)
      DMMTRACE=A(1)+A(2)+A(3)
      IF(ABS(DMMTRACE).LT.TOL)DMMTRACE=PZERO
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMROOTJ2(A)
C     Purpose: Square root of second invariant of second order tensor A
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(6),AD(6)
      PARAMETER (PHALF=0.5D0)
      CALL DMMGETDEV(A,AD)
      DMMROOTJ2=SQRT(PHALF*DMMDBD(AD,AD))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMENINC(A,B,R)
C     Purpose: Energy inc. for stress A, strain inc. B and density R
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(6),B(6)
      DMMENINC = DMMDBD(A,B)/R
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTMULT(T,T0,TM,XP)
C     Purpose: Johnson-Cook homologous temperature
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(PZERO=0.D0,PONE=1.D0)
      TARG= MAX(T-T0,PZERO)
      DMMTMULT= PONE-(TARG/(TM-T0))**XP
      dmmtmult= pone
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMYLDFNC(ARGI1,ROOTJ2,EQPS,TMULT)
C     Purpose: Value of yield function
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
      PARAMETER(TOOR2=0.707106781186547524400844362104849039284836D0)
      PARAMETER ( ROOT23=0.816496580927726032732428024901963797321982D0)
      FF= A1 - A3*EXP(A2*ARGI1) - A4*ARGI1 + A5*(ROOT23*EQPS)**A6
      DMMYLDFNC= TOOR2*ROOTJ2 - FF*TMULT
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMDYLDFNCDT(D1,D2,D3)
C     Purpose: Derivative of yield function wrt T
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
      PARAMETER(PZERO=0.D0)
      TARG= MAX(D3-T0,PZERO)
      IF(TARG.EQ.PZERO)THEN
         DMMDYLDFNCDT= PZERO
      ELSE
         DMMDYLDFNCDT= (A1-A2*D1+A3*D2**A4)*XP/TARG*(TARG/(TM-T0))**XP
      ENDIF
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMHYFNC(A3,A4,EQPS,RMS)
C     Purpose:
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(PZERO=0.D0)
      IF(EQPS.GT.PZERO)THEN
         HY= A3*A4*EQPS**A4*SQRT(RMS*RMS)/EQPS
      ELSE
         HY= PZERO
      ENDIF
ctim
      DMMHYFNC=HY
      dmmhyfnc=PZERO
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMHTFNC(TAUN,RM,R0,CV)
C     Purpose:
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (PZERO=0.D0)
      DIMENSION TAUN(6),RM(6)
      DMMHTFNC= DMMENINC(TAUN,RM,R0)/CV
      dmmhtfnc=pzero
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION STIFFANISOMEAS(BM,TWOG,TWOG1,EDEVMAGSQ)
C     Purpose:
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (PONE=1.D0,PTWO=2.D0,PTHREE=3.D0,PFIVE=5.D0,PSIX=6.D0)
      PARAMETER (PNINE=9.D0)
      PARAMETER (PI=3.1415926535897932384626433832795028841971693993D0)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      XISOMAG=SQRT(PNINE*BM**2 + PFIVE*TWOG**2)
      XMAG=SQRT(PNINE*BM**2+PFIVE*TWOG**2+PSIX*TWOG1**2*EDEVMAGSQ)

C     Scalar measure of anisotropy
      STIFFANISOMEAS= PTWO/PI*ACOS(XISOMAG/XMAG)
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE EOSARRAYREORDER(UIOLD,UI)
C***********************************************************************
C     PURPOSE: Create new UI array with properties ordered as required
C     by the Kerley EOS
C
C input
C -----
C    UIOLD: Original property array
C
C output
C -----
C    UI: Newly re-ordered property array
C    DC: Derived Constants
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100422|tjfulle|created routine
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C............................................................parameters
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

C.................................................................common
C.................................................................passed
      DIMENSION UIOLD(*),UI(*)
C..................................................................local
CcccccccccccccccccccccccccccccccccccccccccccccccccccccccEOSARRAYREORDER
      UI(1)= UIOLD(IPR0)     ! - initial density (required)
      UI(2)= UIOLD(IPT0)     ! - initial temperature
      UI(3)= UIOLD(IPC0)     ! - sound speed (required)
      UI(4)= UIOLD(IPS1)     ! - coefficient of linear term US-UP fit
      UI(5)= UIOLD(IPGP)     ! - Gruneisen parameter
      UI(6)= UIOLD(IPCV)     ! - heat capacity (required)
      UI(7)= UI(6)*UI(2)     ! - shift in energy zero (see below)
      UI(8)= 0.D0            ! - initial porous density
      UI(9)= 0.D0            ! - crushup pressure
      UI(10)= 0.D0           ! - pressure at elastic limit
      UI(11)= 3670.D0        ! - sound speed in foam
      UI(12)= 1.D0           ! - number of subcycles in time step
      UI(13)= 0.D0           ! - coefficient of quad. term in US-UP fit
      UI(14)= 1.D0           ! - used for 2-state and MP models
      UI(15)= UIOLD(IPR0)    ! - alias for R0
      UI(16)= UIOLD(IPT0)    ! - alias for T0
      UI(17)= UIOLD(IPS1)    ! - alias for S1
      UI(18)= UIOLD(IPGP)    ! - alias for G0
      UI(19)= 0.D0           ! - coefficient of low-pressure term
      UI(20)= 1.D0           ! - constant in low-pressure term
      UI(21)= 1.D0           ! - power of compression in low-pressure term
      UI(22)= 2.D0           ! - power for alpha integration
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE KEREOSMGP
C     Input
     & (KP,X,Y,NP,
C     Output
     &  AP,QF,QM,XM)
C*********************************************Sesame Package************
C
C   PURPOSE.   KEREOSMGP: MGRUN model--polynomial fit to energy function
C
C   INPUT.
C     KP       = number of points to be fit
C     X        = table of 1-R0/RHO
C     Y        = table of energy function
C     NP       = order of polynomial
C
C   OUTPUT.
C     AP       = NP fit coefficients
C     QF       = rms deviation of fit (QF=-1 if equations are singular)
C     QM       = maximum relative error
C     XM       = value of X having maximum relative error
C
C   AUTHOR.    G. I. Kerley, 11/13/97
C   MODIFIED.
C
C   REMARKS.   Uses method of P. D. Crout, Trans. A.I.E.E., Vol. 60,
C              pg. 1235 (1941) for solving set of N linear equations
C              in N unknowns, SUMJ A(I,J)*X(J) = A(I,N+1), where A is
C              a symmetric matrix. A(I,N+1) = X(I) upon completion.
C
C**********************************************GIK 11/13/97*************
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
*-
      DIMENSION X(*),Y(*),AP(*)
      DIMENSION A(20,21)
      PARAMETER (ZERO=0.0D0,ONE=1.0D0)
*
*  Compute arrays.
*
      DO 4 I=1,NP
        A(I,NP+1) = ZERO
        DO 1 K=1,KP
          A(I,NP+1) = A(I,NP+1)+X(K)**I
 1      CONTINUE
        DO 3 J=1,NP
          A(I,J) = ZERO
          DO 2 K=1,KP
            A(I,J) = A(I,J)+X(K)**(I+J)/Y(K)
 2        CONTINUE
 3      CONTINUE
 4    CONTINUE
*
*  Solve for fit coefficients.
*
      QF = -ONE
      DO 7 I=1,NP
        II = I-1
        DO 6 J=I,NP+1
          IF(II.GE.1) THEN
            DO 5 K=1,II
 5          A(I,J) = A(I,J)-A(I,K)*A(K,J)
          ENDIF
          IF(J.NE.I) THEN
            IF(J.LT.NP+1) A(J,I)=A(I,J)
            IF(A(I,I).EQ.ZERO) RETURN
            A(I,J) = A(I,J)/A(I,I)
          ENDIF
 6      CONTINUE
 7    CONTINUE
      IF(NP.GE.2) THEN
        DO 9 I=2,NP
          K = NP-I+1
          KK = K+1
          DO 8 J=KK,NP
            A(K,NP+1) = A(K,NP+1)-A(K,J)*A(J,NP+1)
 8        CONTINUE
 9      CONTINUE
      ENDIF
*
*  Load fit coefficients into array AP.
*
      DO 10 I=1,NP
        AP(I) = A(I,NP+1)
 10   CONTINUE
*
*  Compute rms deviation of fit.
*
      QF = ZERO
      QM = ZERO
      KM = 1
      DO 12 K=1,KP
        YF = ZERO
        DO 11 I=1,NP
          YF = YF+AP(I)*X(K)**I
 11     CONTINUE
        DK = YF/Y(K)-ONE
        SK = ABS(DK)
        QF = QF+DK**2
        IF (SK.GT.QM) THEN
          QM = SK
          KM = K
        ENDIF
 12   CONTINUE
      QF = SQRT(QF/KP)
      XM = X(KM)
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE KEREOSMGV (MC,NC,UI,GC,DC,
C     Input
     &  RHO,ENRG,ALPH,NX,
C     Output
     &  PRES,TEMP,CS,
C     Scratch
     &  S)
C**********************************************EOS Package**************
C
C   PURPOSE.   KEREOSMGV: pressure & temperature as functions of
C              density & energy using Mie-Gruneisen EOS.
C
C   MIG INPUT.
C     MC       = dimensioning constant
C     NC       = number of points to process
C     UI       = user input array
C     GC       = global constants (not used here)
C     DC       = derived material constants array
C
C   INPUT.
C     RHO      = MASS_DENSITY
C     ENRG     = SPECIFIC_INTERNAL_ENERGY
C     ALPH     = EXTRA~1 (porosity parameter)
C     NX       = NX=1 if using porosity option, else NX=0
C
C   OUTPUT.
C     PRES     = THERMODYNAMIC_PRESSURE
C     TEMP     = ABSOLUTE_TEMPERATURE
C     CS       = SOUND_SPEED
C
C   SCRATCH.
C     S(MC,6)  = temporary storage
C       S(K,1) = dP/dR on return
C       S(K,2) = dP/dT on return
C       S(K,3) = dE/dT on return
C       S(K,4) = dE/dR on return
C
C   AUTHOR.    G. I. Kerley, 04/10/91
C   MODIFIED.  GIK-03/26/97-replaced with MIG version
C   MODIFIED.  GIK-11/20/97-added low-pressure modifications to Hugoniot
C   MODIFIED.  GIK-02/12/00-added trapping of EOS errors
C   modified:03/01/02-rlb-replaced 'call bombed' with rho(n)=r0.
C   modified:04/10/02-rlb-added line to set enrg=-1.0e10 if bad rho.
C                         this will cause an eosmap printout.
C   modified:04/24/02-rlb-changed enrg=-1e10 to enrg=-1.0.
C   modified:10/18/03-dac-ensured conservative sound speed returned for
C                         elastic region of p-alpha
C   modified:05/19/04-rgs-corrections to pwr modifications
C
C**********************************************GIK 02/12/00*************
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
      CHARACTER*10 DEN
      PARAMETER (ZERO=0.D0,ONE=1.0D0,TWO=2.0D0,THR=3.0D0,FOR=4.0D0,
     &           FIV=5.0D0,SIX=6.0D0,HALF=0.5D0)
      DIMENSION UI(*),GC(*),DC(*)
      DIMENSION RHO(MC), ENRG(MC), ALPH(MC),
     &          PRES(MC), TEMP(MC), CS(MC), S(MC,8)
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     avoid compiler warning of unused variables
      dum=gc(1)
*
      R0 = UI(1)      !initial density (required)
      T0 = UI(2)      !initial temperature
      C0 = UI(3)      !sound speed
      S0 = R0*C0**2   !Initial isentropic bulk modulus
      S1 = UI(4)      !coefficient of linear term US-UP fit
      S2 = UI(13)     !coefficient of quad. term in US-UP fit
      G0 = UI(5)      !Gruneisen parameter = -d(lnT)/d(lnv) = d(lnT)/d(lnr)
      CV = UI(6)      !heat capacity =
      DPT = G0*R0*CV  !PTv
      BX = UI(19)/C0  !=B/C0 = (coef of low pres term)/sndspd
      XB = UI(20)     !constant in low-pressure term
      FB = UI(21)     !=NB=power of compression in low-pressure term
      PWR = UI(22)    !power for alpha integration, default=2.0
      CX = TWO*FB     !=2*NB
      B0 = R0*(C0*(ONE-BX))**2-G0*DPT*T0
      A1 = DC(1)      !coefficient in temperature fit (derived const)
      A2 = DC(2)      !coefficient in temperature fit (derived const)
      A3 = DC(3)      !coefficient in temperature fit (derived const)
      A4 = DC(4)      !coefficient in temperature fit (derived const)
      A5 = DC(5)      !coefficient in temperature fit (derived const)
      PF = DC(10)     !parameter in temperature fit (derived const)
      XF = DC(11)     !parameter in temperature fit (derived const)
      CF = DC(12)     !parameter in temperature fit (derived const)
      RMX = DC(13)    !maximum allowed density
      AF = DC(9)      !=(A0-1)/(PS-PE)**2
      PS = UI(9)      !crushup pressure
      CE = UI(11)     !sound speed in foam

*
*  Check for realistic density.
*
      DO 1 N=1,NC
       IF(RHO(N).GE.RMX .OR. RHO(N).LE.ZERO)  THEN
C        Flag the error so we can get graceful shutdown or at least
C        an error report of some kind
C        This produces a fatal error with an ungraceful shutdown
         WRITE(DEN,'(1PE10.3)')  RHO(N)
         CALL FATERR('KEREOSMGV','UNATTAINABLE DENSITY, RHO='//DEN)
         RHO(N)=R0
         ENRG(N)=-ONE
       ENDIF
C
       S(N,6) = RHO(N)
       IF (NX.NE.0) S(N,6)=MAX(ONE,ALPH(N))*S(N,6)
 1    CONTINUE
*
      DO 2 N=1,NC
*
*  Compute zero Kelvin isentrope.
*
       S(N,1) = ONE-R0/S(N,6)
       IF (S(N,1).GT.ZERO) THEN
        S(N,4) = ONE-S1*S(N,1)
        S(N,5) = SQRT(S(N,4)**2-FOR*S2*S(N,1)**2)
        S(N,2) = TWO/(S(N,4)+S(N,5))
        S(N,7) = (S(N,1)/XB)**FB
        S(N,3) = BX*EXP(-S(N,7))
        S(N,8) = CF*S(N,7)/(S(N,7)+XF)**PF
        CS(N) = S(N,2)+S(N,1)*(S1+(S1*S(N,4)+FOR*S2*S(N,1))/S(N,5))
     &          *S(N,2)**2-S(N,3)*(ONE-CX*S(N,7))
        S(N,2) = S(N,2)-S(N,3)
        S(N,3) = S(N,1)*(A1+S(N,1)*(A2+S(N,1)*(A3+S(N,1)*(A4
     &           +S(N,1)*A5))))+S(N,8)
        S(N,4) = S(N,1)*(TWO*A1+S(N,1)*(THR*A2+S(N,1)*(FOR*A3
     &           +S(N,1)*(FIV*A4+S(N,1)*SIX*A5))))
     &           +S(N,8)*(ONE+PF*(ONE+PF*S(N,7)/(S(N,7)+XF)))
        TEMP(N) = T0*EXP(G0*S(N,1))
        S(N,5) = S0*(HALF-S(N,3))*(S(N,1)*S(N,2))**2/R0-CV*TEMP(N)
        PRES(N) = S0*S(N,1)*(ONE-G0*S(N,1)*S(N,3))*S(N,2)**2-DPT*TEMP(N)
        CS(N) = S0*S(N,2)*(CS(N)*(ONE-G0*S(N,1)*S(N,3))
     &          -S(N,1)*G0*S(N,4)*S(N,2))-DPT*G0*TEMP(N)
       ELSE
        S(N,5) = HALF*B0*S(N,1)**2/R0-DPT*T0*S(N,1)/R0-CV*T0
        PRES(N) = B0*S(N,1)-DPT*T0
        CS(N) = B0
       ENDIF
*
*  Find temperature from energy, then pressure and sound speed.
*
       TEMP(N) = (ENRG(N)-S(N,5)-UI(7))/CV
       PRES(N) = PRES(N)+DPT*TEMP(N)
       IF (NX.GT.0) PRES(N)=PRES(N)/MAX(ONE,ALPH(N))
       S(N,1) = CS(N)*R0/S(N,6)**2
       S(N,2) = DPT
       IF (NX.GT.0) S(N,2)=S(N,2)/MAX(ONE,ALPH(N))
       S(N,3) = CV
       S(N,4) = (PRES(N)-TEMP(N)*S(N,2))/RHO(N)**2
       CS(N) = MAX(ZERO,S(N,1)+TEMP(N)*(S(N,2)/RHO(N))**2/CV)
       CS(N) = SQRT(CS(N))
       IF (NX.GT.0) THEN
        IF (CE.GT.CS(N).AND.ALPH(N).LT.ONE+AF*MAX(ZERO,PS-PRES(N))**PWR)
     &     CS(N)=CE
       ENDIF
 2    CONTINUE

      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE KEREOSMGJ
C     Input
     & (CS,S1,S2,G0,B,XB,FB,
C     Output
     &  A1,A2)
C**********************************************EOS Package**************
C
C   PURPOSE.   KEREOSMGJ computes temperature fit for Mie-Gruneisen model
C
C   INPUT.
C     CS       = sound speed in US-UP fit
C     S1       = linear coefficient in US-UP fit
C     S2       = quadratic coefficient in US-UP fit
C     G0       = Gruneisen parameter
C     B        = coefficient of low-pressure term
C     XB       = constant in low-pressure term
C     FB       = power of compression in low-pressure term
C
C   OUTPUT.
C     A1(5)    = coefficients in polynomial
C     A2(3)    = fit parameters for B-dependent term
C        A2(1) = PF
C        A2(2) = XF
C        A2(3) = CF
C
C   AUTHOR.    G. I. Kerley, 11/14/97
C
C**********************************************GIK 11/14/97*************
C
*-   INCLUDE IMPDOUBL
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
*-
      PARAMETER (ZERO=0.0D0,ONE=1.0D0,TWO=2.0D0,FOR=4.0D0,KP=100)
      DIMENSION X(0:KP),Y1(0:KP),Y2(0:KP),A1(5),A2(3),CON(7)
      CON(1) = CS
      CON(2) = S1
      CON(3) = S2
      CON(4) = G0
      CON(5) = ZERO
      CON(6) = ONE
      CON(7) = ZERO
*
*  Compute table of points to be fit.
*
      IF (S2.GE.ZERO) THEN
        XMX = ONE/(S1+TWO*SQRT(S2))
      ELSE
        XMX = S1/(S1**2/TWO-TWO*S2)
      ENDIF
      DX = XMX/(KP+1)
      X(0) = ZERO
      Y1(0) = ZERO
      DO 1 K=1,KP
        X(K) = K*DX
        CALL KEREOSMGY(X(K-1),X(K),CON,Y1(K))
        Y1(K) = Y1(K)+Y1(K-1)
 1    CONTINUE
      DO 2 K=1,KP
        C1 = ONE-S1*X(K)
        C2 = SQRT(C1**2-FOR*S2*X(K)**2)
        US = TWO*CS/(C1+C2)
        Y1(K) = Y1(K)*EXP(G0*X(K))/(X(K)*US)**2
        Y2(K) = Y1(K)
 2    CONTINUE
      IF (B.EQ.ZERO) THEN
        PF = ZERO
        XF = ONE
        CF = ZERO
      ELSE
        CON(5) = B
        CON(6) = XB
        CON(7) = FB
        Y2(0) = ZERO
        DO 3 K=1,KP
          CALL KEREOSMGY(X(K-1),X(K),CON,Y2(K))
          Y2(K) = Y2(K)+Y2(K-1)
 3      CONTINUE
        KX = 0
        YX = ZERO
        DO 4 K=1,KP
          C1 = ONE-S1*X(K)
          C2 = SQRT(C1**2-FOR*S2*X(K)**2)
          US = TWO*CS/(C1+C2)-B*EXP(-(X(K)/XB)**FB)
          Y2(K) = Y2(K)*EXP(G0*X(K))/(X(K)*US)**2
          DY = Y2(K)-Y1(K)
          IF (DY.GT.YX) THEN
            KX = K
            YX = DY
          ENDIF
 4      CONTINUE
*
*  Compute fit coefficients for function Y2-Y1.
*
        YK = (X(KX)/XB)**FB
        PF = ONE+3.0/FB
        XF = (PF-ONE)*YK
        CF = YX*(YK+XF)**PF/YK
        DO 5 K=1,KP
          YK = (X(K)/XB)**FB
          YF = CF*YK/(YK+XF)**PF
          Y1(K) = Y2(K)-YF
 5      CONTINUE
      ENDIF
*
*  Save coefficients for B-dependent term
*
      A2(1) = PF
      A2(2) = XF
      A2(3) = CF
*
*  Call KEREOSMGP to compute fit coefficients for function Y1.
*
      CALL KEREOSMGP(KP,X(1),Y1(1),5,A1,QF,QM,XM)
*
*  Compute rms deviation of fit.
*
      QF = ZERO
      QM = ZERO
      KM = 1
      DO 7 K=1,KP
        YK = (X(K)/XB)**FB
        YF = CF*YK/(YK+XF)**PF
        DO 6 I=1,5
          YF = YF+A1(I)*X(K)**I
 6      CONTINUE
        DK = YF/Y2(K)-ONE
        SK = ABS(DK)
        QF = QF+DK**2
        IF (SK.GT.QM) THEN
          QM = SK
          KM = K
        ENDIF
*
*  Debug--print to file
*
*       WRITE(3,10)X(K),Y2(K),YF,DK
*10     FORMAT(4(1PE12.4))
*
 7    CONTINUE
      QF = SQRT(QF/KP)
      XM = X(KM)
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE KEREOSMGY
C     Input
     & (X1,X2,CON,
C     Output
     &  Y)
C**********************************************EOS Package**************
C
C   PURPOSE.   KEREOSMGY integrates temperature function for MGRUN model
C
C   INPUT.
C     X1       = lower limit of integration
C     X2       = upper limit of integration
C     CON      = MGRUN EOS constants
C       CON(1) = CS
C       CON(2) = S1
C       CON(3) = S2
C       CON(4) = G0
C       CON(5) = B
C       CON(6) = XB
C       CON(7) = FB
C
C   OUTPUT.
C     Y        = value of integral
C
C   AUTHOR.    G. I. Kerley, 11/13/97
C   MODIFIED.
C
C   REMARKS.   Uses 10-point Gauss quadrature to compute integral.
C
C**********************************************GIK 11/13/97*************
C
*-   INCLUDE IMPDOUBL
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
*-
      SAVE R,U
      PARAMETER (ZERO=0.0D0,HALF=0.5D0,ONE=1.0D0,TWO=2.0D0,FOR=4.0D0)
      DIMENSION R(5),U(5),CON(7)
      DATA (R(I),I=1,5)/0.147762112D0,0.134633360D0,0.109543181D0,
     1  0.074725675D0,0.033335672D0/
      DATA (U(I),I=1,5)/0.074437169D0,0.216697697D0,0.339704784D0,
     1  0.432531683D0,0.486953264D0/
*
      CS = CON(1)
      S1 = CON(2)
      AX = FOR*CON(3)
      G0 = CON(4)
      B = CON(5)
      XB = CON(6)
      FB = CON(7)
*
      DX = X2-X1
      XM = HALF*(X2+X1)
      Y = ZERO
      DO 1 I=1,5
        X = XM-DX*U(I)
        C1 = ONE-S1*X
        C2 = SQRT(C1**2-AX*X**2)
        F1 = TWO/(C1+C2)
        C3 = (X/XB)**FB
        F2 = B*EXP(-C3)
        US = CS*F1-F2
        DU = X*HALF*CS*F1**2*(S1+(S1*C1+AX*X)/C2)+FB*C3*F2
        DB = EXP(G0*X)
        Y = Y+R(I)*X*US*DU/DB
        X = XM+DX*U(I)
        C1 = ONE-S1*X
        C2 = SQRT(C1**2-AX*X**2)
        F1 = TWO/(C1+C2)
        C3 = (X/XB)**FB
        F2 = B*EXP(-C3)
        US = CS*F1-F2
        DU = X*HALF*CS*F1**2*(S1+(S1*C1+AX*X)/C2)+FB*C3*F2
        DB = EXP(G0*X)
        Y = Y+R(I)*X*US*DU/DB
 1    CONTINUE
      Y = DX*Y
      RETURN
      END


C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE TJFEOSMGV(R,CS)
C***********************************************************************
C     PURPOSE: Pressure and temperature from energy and density
C
C input
C -----
C    R:  Current density
C
C output
C -----
C    CS:  Soundspeed
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
C This include block defines calculation precision.
C
C
C If you find that direct use of this include impedes your installation
C of Kayenta, please contact Kayenta developers (rmbrann@sandia.gov),
C and we will work to resolve the problem.
C
C Altering Kayenta source code (or its includes) will result in
C loss of technical support.
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C............................................................parameters
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   d i m e n s i o n i n g    p a r a m e t e r s @@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  NBASICINPUTS: length of PROP array *not* including properties for
C                "add-on" options such as joints, alternative equations
C                of state, etc.
C
      PARAMETER (NBASICINPUTS=34,NMGDC=13)
C
C     Total number of properties
      PARAMETER (NDMMPROP=NBASICINPUTS+NMGDC)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@   p o i n t e r s   t o  p r o p e r t i e s @@@@@@@@@@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  IP[propname]: pointers to property array
C                Examples: the property B0 is in PROP(IPB0)
C                          and so on...
C-------------------------------------------------------------------------
      PARAMETER(IPB0       =  1) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB1       =  2) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPB2       =  3) !Initial intact elastic bulk modulus (stress)
      PARAMETER(IPG0       =  4) !Initial intact elastic shear modulus (stress)
      PARAMETER(IPG1       =  5)
      PARAMETER(IPG2       =  6)
      PARAMETER(IPG3       =  7)
      PARAMETER(IPA1       =  8) !Strength in uniaxial stress
      PARAMETER(IPA2       =  9) !
      PARAMETER(IPA3       = 10) !
      PARAMETER(IPA4       = 11) !
      PARAMETER(IPA5       = 12) !
      PARAMETER(IPA6       = 13) !
      PARAMETER(IPAN       = 14) !=1 if induced anisotropy is desired
      PARAMETER(IPR0       = 15) ! initial density
      PARAMETER(IPT0       = 16) ! Initial temperature
      PARAMETER(IPC0       = 17) ! Initial bulk sound speed
      PARAMETER(IPS1       = 18) ! linear US-UP fit term
      PARAMETER(IPGP       = 19) ! gruneisen parameter
      PARAMETER(IPCV       = 20) ! specific heat
      PARAMETER(IPTM       = 21) ! melt temperature
      PARAMETER(IPT1       = 22) ! Rate dep term
      PARAMETER(IPT2       = 23) ! Rate dep term
      PARAMETER(IPT3       = 24) ! Rate dep term
      PARAMETER(IPT4       = 25) ! Rate dep term
      PARAMETER(IPXP       = 26) ! Exponent in homologous temperature
      PARAMETER(IPSC       = 27) ! Strength in compression
      PARAMETER(IPIDK      = 28) ! Bulk modulus ID
      PARAMETER(IPIDG      = 29) ! Shear modulus ID
      PARAMETER(IPA4PF     = 30) ! Flow potential A2
      PARAMETER(IPTQC      = 31) ! Taylor-Quinney coefficient
      PARAMETER(IPF1       = 32) ! Free place holder
      PARAMETER(IPTEST     = 33) !=1 if run both iso and aniso in med
      PARAMETER(IPDEJAVU   = 34) !=1 if params have been checked or revised
      PARAMETER(IPDCPROP   = NBASICINPUTS)
      PARAMETER(IPDC1      = IPDCPROP+1)
      PARAMETER(IPDC2      = IPDCPROP+2)
      PARAMETER(IPDC3      = IPDCPROP+3)
      PARAMETER(IPDC4      = IPDCPROP+4)
      PARAMETER(IPDC5      = IPDCPROP+5)
      PARAMETER(IPDC6      = IPDCPROP+6)
      PARAMETER(IPDC7      = IPDCPROP+7)
      PARAMETER(IPDC8      = IPDCPROP+8)
      PARAMETER(IPDC9      = IPDCPROP+9)
      PARAMETER(IPDC10     = IPDCPROP+10)
      PARAMETER(IPDC11     = IPDCPROP+11)
      PARAMETER(IPDC12     = IPDCPROP+12)
      PARAMETER(IPDC13     = IPDCPROP+NMGDC)

C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C@@@@@@@@@@   p o i n t e r s   t o  s t a t e   v a r i a b l e s  @@@@@@@@@@
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C  K[isvname]: pointers to the state variable array
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      PARAMETER (NISV=26)
      PARAMETER (NDMMISV=NISV) !hardwired for SQA
      PARAMETER (NISOSTART=NISV)
      PARAMETER (NANISOSTART=NISOSTART+NISV)
C
C   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PARAMETER (KEQDOT  =1)  !Magnitude of the total strain rate
      PARAMETER (KI1     =2)  !I1 stress invariant
      PARAMETER (KROOTJ2 =3)  !RootJ2
      PARAMETER (KEQPS   =4) !Equivalent plastic SHEAR strain conj to ROOTJ2
      PARAMETER (KEVOL   =5) !Volumetric strain
      PARAMETER (KT      =6)  !KTMPR  - Temperature
      PARAMETER (KCS     =7)  !KSNDSP - Soundspeed
      PARAMETER (KR      =8)  !KR   - Density
      PARAMETER (KEU     =9)  !KEU - Internal energy
      PARAMETER (KRJ     =10) !Jacobian
      PARAMETER (KAM     =11)  !Anisotropy measure
      PARAMETER (KEQPV   =12)  !Free place for EOS ISV
      PARAMETER (KF4     =13)  !Free place for EOS ISV
C
C     Overstress isv
C
      PARAMETER (KQSSIG =13)
      PARAMETER (KQSSIGXX = KQSSIG + 1)
      PARAMETER (KQSSIGYY = KQSSIG + 2)
      PARAMETER (KQSSIGZZ = KQSSIG + 3)
      PARAMETER (KQSSIGXY = KQSSIG + 4)
      PARAMETER (KQSSIGYZ = KQSSIG + 5)
      PARAMETER (KQSSIGZX = KQSSIG + 6)
      PARAMETER (KQSSIGYX = KQSSIGXY)
      PARAMETER (KQSSIGZY = KQSSIGYZ)
      PARAMETER (KQSSIGXZ = KQSSIGZX)
C
C     Induced anisotropy isv
C
      PARAMETER (KE     =KQSSIG+6)
      PARAMETER (KEXX   =KE + 1)
      PARAMETER (KEYY   =KE + 2)
      PARAMETER (KEZZ   =KE + 3)
      PARAMETER (KEXY   =KE + 4)
      PARAMETER (KEYZ   =KE + 5)
      PARAMETER (KEZX   =KE + 6)
      PARAMETER (KEYX   =KEXY)
      PARAMETER (KEZY   =KEYZ)
      PARAMETER (KEXZ   =KEZX)
      PARAMETER (KEJ2   =KE + 7)

      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0)
C.................................................................common
C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
      INTEGER IDK,IDG
      SAVE /DMMPROPI/
      COMMON /DMMPROPI/IDK,IDG
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PR,
     $T1,T2,T3,T4,R0,T0,TM,C0,S1,GP,CV,XP,A5,A6
C.................................................................passed
C..................................................................local
C...................................................................data
C...............................................................external
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DMM_MODULI

c     The MG EOS has a limiting value of density above which it
c     predicts a decrease in the sound speed with increasing density.
c     Until I put in a better fix, we will assign to R, the value of
c     the density, the limiting value of R should the threshold be
c     crossed.  R is restored after the bulk modulus is computed.
      HOLDR= R
      R= MIN(R,(R0*(PONE+S1+PTWO*GP*S1))/(PTWO*GP*S1))
      RJ= R0/R
      EPS= PONE-RJ
      IF(EPS.GT.PZERO)THEN
         RNUM=GP*R**2*S1+(PONE+GP)*R0**2*S1-R*R0*(PONE+S1+PTWO*GP*S1)
         DNOM= ((R*(S1-PONE)-R0*S1)**3)
         DPDR= R0*C0**2*RNUM/DNOM
      ELSE
         DPDR= B0*R0/R0**2
      ENDIF
      CS= SQRT(DPDR)
      R= HOLDR


      RETURN
      END
