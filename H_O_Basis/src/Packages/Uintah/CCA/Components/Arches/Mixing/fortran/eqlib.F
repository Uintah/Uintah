      SUBROUTINE EQUIL (LOUT, LPRNT, LSAVE, EQST, LCNTUE, ICKWRK,
     1           RCKWRK, LENIEQ, IEQWRK, LENREQ, REQWRK, MM, KK,
     2           ATOM, KSYM, NOP, KMON, REAC, TIN, TEST, PIN, PEST,
     3           NCON, KCON, XCON, ierr)
C     3           NCON, KCON, XCON, heatLoss, hin, ierr)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
C     Input:  LOUT - unit number for output messages
C             LPRNT  - logical flag for printing results
C             LSAVE  - unit number for binary solution file
C             EQST   - logical flag for initializing Stanjan
C             LCNTUE - logical flag for continuation (initialize
C                      in calling program)
C             ICKWRK - CHEMKIN integer array
C             RCKWRK - CHEMKIN real array
C             MM     - total number of atomic elements
C             KK     - total number of molecular species
C             ATOM   - character array of element names
C             KSYM   - character array of species names
C             NOP    - integer equilibrium option number
C             KMON   - integer monitor flag
C             REAC   - real array of input mole fractions
C                      of the mixture
C             TEMP   - starting temperature for the problem
C             TEST   - estimated equilibrium temperature
C             PRES   - starting pressure for the problem
C             PEST   - estimated equilibrium pressure
C             NCON   - integer number of species to be held constant
C             KCON   - integer array of index numbers of the species
C                      to be held constant
C             XCON   - real mole fractions input for the species to
C                      be held constant
C             heatLoss - integer flag indicating adiabatic (=0) or 
C                      nonadiabatic (=1) problem; added by Jennifer
C                      Spinti on 12/28/2000
C     **No ouput variables.** - Jennifer Spinti changed this on 12/22/2000.
C     There is now one output variable
C             ierr   - integer flag returns 0 if equilibrium calculation 
C                       was successful, 1 if unsuccessful.
C
C      integer heatLoss
      DIMENSION ICKWRK(*), RCKWRK(*), IEQWRK(*), REQWRK(*), REAC(*),
     1          KCON(*), XCON(*)
      CHARACTER*(*) KSYM(*), ATOM(*)
      LOGICAL LPRNT, EQST, LCNTUE
C
      COMMON /IPAR/ NIWORK, NICMP, NIKNT, NRWORK, NRKNT, NRTMP, NRA,
     1              NRADD,  MAXTP, NCP,   NCP1,   NCP2,  NCP2T,
     2              NPHASE, NSMS,  NWT,   NXCON,  NKCON, NAMAX,
     3              NX1,    NX2,   NY1,   NY2,    NT1,   NT2,
     4              NP1,    NP2,   NV1,   NV2,    NWM1,  NWM2,
     5              NS1,    NS2,   NU1,   NU2,    NH1,   NH2,
     6              NC1,    NC2,   NCDET, NTEST,  NPEST
      COMMON /RPAR/ ER2CAL, RU, RUC, T298 
C
      IF (.NOT. EQST) THEN
C
C        INITIALIZE POINTERS
C
         CALL EQLEN (MM, KK, NCON, NITOT, NRTOT)
C
         LDEBUG = 0
         if (LDEBUG.eq.1) then
            WRITE  (LOUT, 15)
 15         FORMAT (/' EQUIL:  Chemkin interface for Stanjan-III',
     1           /'         CHEMKIN-II Version 3.0, December 1992',
C*****precision > double
     2           /'         DOUBLE PRECISION')
C*****END precision > double
C*****precision > single
C     2           /'         SINGLE PRECISION')
C*****END precision > single
C
            WRITE (LOUT, 7010) LENIEQ, NITOT, LENREQ, NRTOT
 7010       FORMAT (/'         WORKING SPACE REQUIREMENTS',
     1           /,'             PROVIDED        REQUIRED ',
     2           /,' INTEGER ' , 2I15,
     3           /,' REAL    ' , 2I15)
         endif
C
         IF (NITOT.GT.LENIEQ .OR. NRTOT.GT.LENREQ) THEN
            WRITE (LOUT, *)
     1           '  FATAL ERROR, NOT ENOUGH WORK SPACE PROVIDED'
            write (lout, *) nitot, lenieq, nrtot, lenreq
            STOP
         ENDIF
C
         if (LDEBUG.eq.1) THEN
            WRITE (LOUT, *) ' STANJAN:  Version 3.8C, May 1988 '
            WRITE (LOUT, *) '           W. C. Reynolds, Stanford Univ. '
         endif
C
C        ELEMENTAL COMPOSITION OF SPECIES
C
         CALL CKNCF (MM, ICKWRK, RCKWRK, IEQWRK(NICMP))
C
C        THERMODYNAMICS VARIABLES
C
         CALL CKATHM (NCP2, MAXTP-1, ICKWRK, RCKWRK, MAXTP,
     1                IEQWRK(NIKNT), REQWRK(NRTMP), REQWRK(NRA))
C
         DO 100 K = 1, KK
            REQWRK(NRKNT + K - 1) = dble (IEQWRK(NIKNT + K - 1))
  100    CONTINUE
C
         CALL CKWT (ICKWRK, RCKWRK, REQWRK(NWT))
         CALL CKRP (ICKWRK, RCKWRK, RU, RUC, PATM)
C
C        ERG-TO-CALORIE CONVERSION CONSISTENT WITH STANJAN
C
         T298 = 298.15D0
         CVCJ = 4.184D0
         ER2CAL = 1.0D0 / (CVCJ * 1.0D10)
         EQST = .TRUE.
C
      ENDIF
C
C     NORMALIZED INPUT MOLE FRACTIONS
C
      SUMR = 0.0D0
      DO 150 K = 1, KK
         REQWRK(NX1 + K - 1) = REAC(K)
         SUMR = SUMR + REAC(K)
  150 CONTINUE
      IF (SUMR .NE. 0.0D0) THEN
         DO 175 K = 1, KK
            REQWRK(NX1 + K - 1) = REAC(K) / SUMR
  175    CONTINUE
      ENDIF
      CALL CKXTY  (REQWRK(NX1), ICKWRK, RCKWRK, REQWRK(NY1))
C
C     THERMODYNAMIC PROPERTIES
C
      REQWRK(NT1) = TIN
      REQWRK(NTEST) = TEST
C     ***Jennifer Spinti changed this***12/28/2000
C      if (heatLoss.eq.0) then
         CALL CKHBMS(REQWRK(NT1), REQWRK(NY1), ICKWRK, RCKWRK,
     1        REQWRK(NH1))
C      else
C         reqwrk(nh1) = hin
C      endif
      CALL CKUBMS(REQWRK(NT1), REQWRK(NY1), ICKWRK, RCKWRK, REQWRK(NU1))
C
      REQWRK(NP1) =   PIN * PATM
      REQWRK(NPEST) = PEST * PATM
      CALL CKRHOX (REQWRK(NP1), REQWRK(NT1), REQWRK(NX1), ICKWRK,
     1             RCKWRK, RHO)
      REQWRK(NV1) = 1.0D0 / RHO
C
C     compute entropy of mixture in mass units
C
      CALL CKSMS (REQWRK(NT1), ICKWRK, RCKWRK, REQWRK(NSMS))
      S1 = 0.0D0
      DO 200 K = 1, KK
         SMS = REQWRK(NSMS + K - 1)
         WT  = REQWRK(NWT  + K - 1)
         Y1  = REQWRK(NY1  + K - 1)
         X1  = REQWRK(NX1  + K - 1)
         IF (X1.GT. 0) S1 = S1 + Y1*(SMS - (RU/WT) * LOG(X1 * PIN))
200   CONTINUE
      REQWRK(NS1) = S1
C
      CALL CKMMWY (REQWRK(NY1), ICKWRK, RCKWRK, REQWRK(NWM1))
      REQWRK(NWM2) = REQWRK(NWM1)
      REQWRK(NC1) = 0.0D0
      IF (NOP.EQ.10) THEN
         CALL CKCPBS (REQWRK(NT1), REQWRK(NY1), ICKWRK, RCKWRK, CP)
         CALL CKCVBS (REQWRK(NT1), REQWRK(NY1), ICKWRK, RCKWRK, CV)
         GAMMA = CP/CV
         REQWRK(NC1) = SQRT (GAMMA * RU * REQWRK(NT1) / REQWRK(NWM1))
      ENDIF
C
C        PROPERTY UNITS:
C           P: pressure in Pa (N/M**2)  1 Pa = 10 dynes/cm2
C           T: Kelvins
C           S: J/Kg-K
C           H: J/Kg
C           U: J/Kg
C           V: M**3/Kg
C

      REQWRK(NP1) = REQWRK(NP1) * 1.0D-1
      REQWRK(NPEST) = REQWRK(NPEST) * 1.0D-1
      REQWRK(NH1) = REQWRK(NH1) * 1.0D-4
      REQWRK(NS1) = REQWRK(NS1) * 1.0D-4
      REQWRK(NU1) = REQWRK(NU1) * 1.0D-4
      REQWRK(NV1) = REQWRK(NV1) * 1.0D-3
C
      DO 250 N = 1, NCON
         REQWRK(NXCON + N - 1) = XCON(N)
         IEQWRK(NKCON + N - 1) = KCON(N)
  250 CONTINUE
C
C          COMPUTE EQUILIBRIUM COMPOSITION
C
      ierr = 0
      CALL EQRUN (NOP, MM, KK, NCON, KMON, LOUT, ATOM, KSYM, IEQWRK,
     1            REQWRK, ICKWRK, RCKWRK, ierr)
      if (ierr.eq.1) return
C
C     CONVERT BACK TO CGS
C
      REQWRK(NP1) = REQWRK(NP1) * 1.0D+1 / PATM
      REQWRK(NP2) = REQWRK(NP2) * 1.0D+1 / PATM
      REQWRK(NH1) = REQWRK(NH1) * 1.0D+4
      REQWRK(NH2) = REQWRK(NH2) * 1.0D+4
      REQWRK(NS1) = REQWRK(NS1) * 1.0D+4
      REQWRK(NS2) = REQWRK(NS2) * 1.0D+4
      REQWRK(NU1) = REQWRK(NU1) * 1.0D+4
      REQWRK(NU2) = REQWRK(NU2) * 1.0D+4
      REQWRK(NV1) = REQWRK(NV1) * 1.0D+3
      REQWRK(NV2) = REQWRK(NV2) * 1.0D+3
      REQWRK(NC1) = REQWRK(NC1) * 1.0D+2
      REQWRK(NC2) = REQWRK(NC2) * 1.0D+2
      REQWRK(NCDET) = REQWRK(NCDET) * 1.0D+2
C
      IF (LPRNT) CALL EQPRNT (LOUT, LSAVE, LCNTUE, NOP, KK, KSYM,
     1           REQWRK)
C
      RETURN
      END
C
C------------------------------------------------------------------
C
      SUBROUTINE EQRUN (NOP, MM, KK, NCON, KMON, LOUT, ATOM, KSYM,
     1                  IEQWRK, REQWRK, ICKWRK, RCKWRK, ierr)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      PARAMETER (HUGE = 1.0D+33)
      DIMENSION IEQWRK(*), REQWRK(*), ICKWRK(*), RCKWRK(*)
      CHARACTER ATOM(*)*16, KSYM(*)*16
C
C----------------PROLOGUE FOR EQRUN--------------------------------
C
C    THIS SUBROUTINE DRIVES THE CHEMKIN INTERFACE TO THE STANJAN-3
C    EQUILIBRIUM CODE.  IT CAN ONLY BE CALLED AFTER SUBROUTINE
C    EQINIT IS CALLED TO SET UP THE INTERNAL WORKING AND STORAGE SPACE.
C    ALSO, THE INCLUDED SUBROUTINE SJTIHS **MUST** BE USED INSTEAD OF
C    THE SUBROUTINE OF THE SAME NAME PROVIDED IN THE STANJAN LIBRARY.
C
C  INPUT
C    NOP   - INTERGER DESIGNATING THE PROBLEM TYPE:
C            1.  SPECIFIED T AND P
C            2.  SPECIFIED T AND V
C            3.  SPECIFIED T AND S
C            4.  SPECIFIED P AND V
C            5.  SPECIFIED P AND H
C            6.  SPECIFIED P AND S
C            7.  SPECIFIED V AND U
C            8.  SPECIFIED V AND H
C            9.  SPECIFIED V AND S
C            10. CHAPMAN-JOUGUET DETONATION (H, S, V, T CONTAIN THE
C                UNBURNED STATE, AND TE IS THE BURNED ESTIMATE.
C    KMON  - INTEGER COTROLLING PRINTED OUTPUT FROM STANJAN.  FOR
C            NO RUN-TIME INFORMATION USE KMON=0.
C    LOUT  - OUTPUT UNIT ONTO WHICH STANJAN DIAGNOSTICS ARE WRITTEN.
C    NPHASE- NUMBER OF PHASES IN THE SYSTEM.  ***THESE OPTIONS ARE NOT
C            FULLY IMPLEMENTED IN THIS VERSION.  USE NPHASE=1.
C    KFRZ  - 0 EQUILIBRIUM COMPOSITION
C            1 FROZEN COMPOSITION WITH SAME T ALL PHASES
C            2 FROEN COMPOSITION WITH DIFFERENT PHASE TEMPERATURES
C            ***THE PHASE OPTIONS ARE NOT FULLY IMPLEMENTED IN THIS
C                VERSION --- USE KFRZ=0
C    REAC  - VECTOR OF REACTANT MOLE FRACTIONS, FROM WHICH THE ATOM
C            POPULATIONS ARE COMPUTED.  DIMENSION REAC(*) AT LEAST KK,
C            WHERE KK IS THE NUMBER OF SPECIES.
C    DCS   - VECTOR OF THE CONDENSED PHASE SPECIES DENSITIES.  ***THE
C            CONDENSE PHASE SPECIES OPTIONS ARE NOT FULLY IMPLEMENTED.
C            DCS(*) MUST BE IMENSIONED AT LEAST KK IN THE CALLING
C            PROGRAM, BUT ALL ENTRIES CAN BE 0.
C    ATOM  - ARRAY OF CHARACTER*16 NAMES OF THE ELEMENTS IN THE PROBLEM.
C            DECLARE AND DIMENSION CHAR*16 ATOM(MM), WHERE MM IS THE
C             NUMBER OF ELEMENTS.
C    KSYM  - ARRAY OF CHARACTER*16 NAMES OF THE SPECIES IN THE PROBLEM.
C            DECLARE AND DIMENSION CHAR*16 KSYM(KK).
C    IEQWRK- INTEGER EQUILIBRIUM WORK SPACE.
C            IEQWRK(*) MUST BE DIMENSIONED AT LEAST LENIEQ.
C    REQWRK- REAL EQUILIBRIUM WORK SPACE.
C            REQWRK(*) MUST BE DIMENSIONED AT LEAST LENREQ.
C    LENIEQ- ACTUAL DIMENSION DECLARED FOR IEQWRK(*)
C            LENIEQ MUST BE AT LEAST
C            22 + 14*MM + 4*NPHASE + 8*KK + 2*MM*KK
C            WHERE MM IS THE NUMBER OF ELEMENTS AND KK IS THE NUMBER OF
C            SPECIES.
C    LENREQ- ACTUAL DIMENSION DECLARED FOR REQWRK(*)
C            LENREQ MUST BE AT LEAST
C            24 + 16*MM + 12*MM*MM + 3*NPHASE*MM + 6*NPHASE
C                 + 18*KK + NWORK*NWORK + NWORK,  WHERE NWORK IS
C            MAX(2*MM, MM+NPHASE).
C    ICKWRK- INTEGER CHEMKIN WORK SPACE
C    RCKWRK - REAL CHEMKIN WORK SPACE.
C    LENRCK- ACTUAL DECLARED DIMENSION OF THE REAL CHEMKIN WORK SPACE.
C
C  INPUT AND OUTPUT
C    (ONLY THE PROPERTIES REQUIRED FOR THE GIVEN OPTION NEED TO BE GIVEN
C    P     - PRESSURE (DYNES/CM**2)
C    T     - TEMPERATURE (K)
C    H     - MIXTURE ENTHALPY (J/KG)
C    S     - MXTURE ENTROPY (J/KG-K)
C    U     - MIXTURE INTERNAL ENERGY (J/KG)
C    V     - MIXTURE SPECIFIC VOLUME (M**3/KG)
C    TE    - ESTIMATED BURNED TEMPERTURE FOR THE C-J DETONATION OPTION.
C    TP    - TP(N) IS THE TEMPERATURE OF THE N-TH PHASE.  ***THE PHASE
C            OPTIONS ARE NOT FULLY IMPLEMENTED IN THIS VERSION.
C
C  OUTPUT
C    C     - SOUND SPEED (M/S) IF THE OTION CALLS FOR COMPUTING IT
C    CDET  - C-J DETONATION WAVESPEED (M/S)
C    WM    - EQUILIBRIUM MEAN MOLECULAR WEIGHT (G/MOLE)
C    PMOL  - ARRAY OF RELATIVE MOLES OF THE N-TH PHASE. ***THIS IS NOT
C            IMPLEMENTED.
C    SMOL  - ARRAY OF RELATIVE MOLES OF THE K-TH SPECIES.  ***THIS IS
C            NOT IMPLEMENTED.
C    XP    - PHASE MOLE FRACTION OF THE K-TH SPECIES. ***THIS IS NOT
C            IMPLEMENTED.
C    XM    - MIXTURE MOLE FRACTION OF THE K-TH SPECIES.  DIMENSION XM(*)
C            AT LEAST KK.
C    Y     - MIXTURE MASS FRACTION.  DIMENSION Y(*) AT LEAST KK.
C
C-------------------END PROLOGUE FOR EQRUN---------------------------
C
C 80 pointers required by SJEQLB
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
C
C  20 additional pointers required by  SJTP
       COMMON /SJTPTR/
     ;   IoKFRZ,IoCVCJ,IoPATM,IoRGAS,IoP,IoT,IoH,IoS,IoU,IoV,
     ;   IoWM,IoTP,IoDCS,IoDHF0,IoWMS,IoHMH0,IoS0,IoWMP,IoXM,IoYM
C
C  10 additional pointers required by SJSET
       COMMON /SJSPTR/
     ;   IoKNFS,IoKSME,IoKUFL,IoNAF,IoNSF,IoJFS,IoNF,LoNF,IoITHL,IoITHM
C
C  20 additional pointers required by SJROP
       COMMON /SJRPTR/
     ; IoKDET,IoKRCT,IoKSND,IoKTRI,IoKTRT,IoI2,IoI3,IoNCON,IoTE,IoC,
     ; IoCDET,IoH1,IoP1,IoT1,IoV1,IoR5,IoR6,IoSMFA,IoSMFB,IoSMOZ
C
C--------------------------------------------------------------------
C      Species are loaded by phases:
C	   phase 1:  species  1, 2, ... , NSP(1)
C	   phase 2:  species  NSP(1)+1, ... , NSP(2)
C	   ...
C	   phase NP: species  NSP(NP-1)+1, ... , NSP(NP)
C
C	   CHEM(J): CHARACTER*16 name of species J, for J=1, ... ,NS
C
C      Atoms appearing in the species are numbered sequentially:
C	   ATOM(I): CHARACTER*16 name of the atom, for I=1, ..., NA
C
C      N(I,J) is the number of Ith atoms in a Jth molecule, which may
C      be negative for electrons in positive ions.
C
C      Species may be cross-referenced to a data file for property
C      look-up: JFS(J) is the file reference of species
C      (JFS le NSMAX).
C------------------------------------------------------------------
C    If calling SJROP:
C
C    Load the following into the work arrays:
C
C	   Mixture specification parameters:
C	       NA	   number of atom types in the system
C	       NP	   number of phases in the system
C	       NS	   number of species in the system
C	       NSP(M)	   number of species in Mth phase
C	       N(I,J)	   number of Ith atoms in molecule of Jth species
C	       PA(I)	   (relative) mols of Ith atoms
C
C	   Pointers to species data in the file:
C              JFS(J)	   index of Jth species in the species data file
C
C	   Species properties required for solution:
C	       DCS(J)	   density of Jth species, g/cc (0 for gas)
C	       DHF0(J)	   enthalpy of formation of Jth species, kcal/mol
C	       WMS(J)	   molal mass of Jth species, g/mol
C
C	   Properties specified (load two specified properties):
C	       P	   pressure, Pa    (trial value if not specified)
C	       T	   temperature, K  (trial value if not specified)
C	       H	   mixture enthalpy, J/kg
C	       S	   mixture entropy, J/kg-K
C	       U	   mixture internal energy, J/kg
C	       V	   mixture specific volume, m**3/kg
C
C	   Other property specifications for special options:
C	       TE	   estimate of burned gas T (C-J det. run only)
C	       TP(K)	   T of Kth reactant phase (reactants run only)
C
C	   Control parameters:
C	       KFRZ	 0 equilibrium composition
C			 1 frozen composition with same T all phases
C			 2 frozen composition with different phase temperatures
C	       KMON	   runtime monitor control (0 none)
C	       KUMO	   output unit for runtime monitor
C
C	   Select the option:
C	       NOP	   option selection:
C			 1 specified T and P
C			 2 specified T and V
C			 3 specified T and S
C			 4 specified P and V
C			 5 specified P and H
C			 6 specified P and S
C			 7 specified V and U
C			 8 specified V and H
C			 9 specified V and S
C			10 Chapman-Jouguet detonation
C		       (H,S,V,T contain unburned state,
C                        TE is burned estimate)
C
C       CALL SJROP (NAMAX,NSMAX,NIWORK,NRWORK,NSWORK,
C     ;		   ATOM,KSYM,IWORK,RWORK,SWORK,NOP)
C
C      Output in the work arrays:
C
C	   Calculated temperature-dependent properties of the species:
C	       G(J)	   g(T,P)/RT for Jth species
C	       HMH0(J)	   H(T)-H(298.15) for Jth species, kcal/mol
C	       S0(J)	   S0(T) for Jth species, cal/mol-K
C
C	   Properties of the entire mixture:
C	       C	   sound speed, m/s (if calculated)
C	       H	   mixture enthalpy, J/kg
C	       S	   mixture entropy, J/kg-K
C	       U	   mixture internal energy, J/kg
C	       V	   mixture specific volume, m**3/kg
C	       WM	   molal mass of mixture, g/mol
C
C	   Distributed properties of the mixture:
C	       PMOL(M)	   (relative) mols of the Mth phase
C	       SMOL(J)	   (relative) mols of Jth species
C	       WMP(M)	   molal mass of Mth phase, g/mol
C	       X(J)	   phase mol fraction of Jth species
C	       XM(J)	   mixture mol fraction Jth species
C	       YM(J)	   mixture mass fraction Jth species
C
C	   Detonation properties (if C-J option).
C	       CDET	   C-J detonation wavespeed, m/s (if calculated)
C	       H1	   Chap. Jouguet unburned enthalpy, J/kg
C	       P1	   Chap. Jouguet unburned pressure, Pa
C	       T1	   Chap. Jouguet unburned temperature, K
C	       V1	   Chap. Jouguet volume, m**3/kg
C	       T2	   Chap. Jouguet detonation temperature, K ????
C
C	   Flags:
C	       KERR	   error flag
C
C	   Other data of possible interest:
C	       NB	   number of independent atoms
C	       IB(K)	   atom index of Kth indpendent atom
C	       ELAM(K)	   element potential of Kth independent atom
C-------------------------------------------------------------------
C    Check KERR = 0 for successful calculation.
C         LOAD THE STANJAN COMMON BLOCKS
C
      COMMON /IPAR/ NIWORK, NICMP, NIKNT, NRWORK, NRKNT, NRTMP, NRA,
     1              NRADD,  MAXTP, NCP,   NCP1,   NCP2,  NCP2T,
     2              NPHASE, NSMS,  NWT,   NXCON,  NKCON, NAMAX,
     3              NX1,    NX2,   NY1,   NY2,    NT1,   NT2,
     4              NP1,    NP2,   NV1,   NV2,    NWM1,  NWM2,
     5              NS1,    NS2,   NU1,   NU2,    NH1,   NH2,
     6              NC1,    NC2,   NCDET, NTEST,  NPEST
      COMMON /RPAR/ ER2CAL, RU, RUC, T298 
C
      ierr = 0
      REQWRK(2) = HUGE
      CALL SJSPTS( NAMAX, NPHASE, KK, NIWORK, NRWORK, NRADD,
     1             REQWRK, LOUT)
C
C         SET THE RUN-TIME MONITOR
C
      IEQWRK(IoKMON) = KMON
      IF (KMON .NE. 0) THEN
         IEQWRK(IoKUMO) = LOUT
      ELSE
         IEQWRK(IoKUMO) = 40
      ENDIF
C
      DO 260 M = 1, MM
         REQWRK(IoPA+M) = 0.0D0
         DO 260 K = 1, KK
C           SET THE RELATIVE ATOM POPULATIONS FROM REAC(K)
            RCMP = dble (IEQWRK(NICMP + (K-1)*MM + M - 1))
            X    = REQWRK(NX1 + K - 1)
            REQWRK(IoPA+M) = REQWRK(IoPA+M) + X * RCMP
            IEQWRK(IoN+M+LoN*K) = IEQWRK(NICMP+(K-1)*MM+M-1)
  260 CONTINUE
C
       IEQWRK(IoNA) = NAMAX
       IEQWRK(IoNP) = NPHASE
       IEQWRK(IoNS) = KK
       DO 275 L = 1, NPHASE
          IEQWRK(IoNSP+L) = KK
275    CONTINUE
C
C      ITERATION LOOP FOR VARIABLE MEAN MOLECULAR WEIGHT
C
      ITER = 0
      TOTMOL = 1.0D0
310   CONTINUE
C
      DO 350 N = 1, NCON
         M = MM + N
         XCON = REQWRK(NXCON + N - 1)
         KCON = IEQWRK(NKCON + N - 1)
C
C        SET CONSTRAINTS ON INDIVIDUAL SPECIES MOLE FRACTIONS
C
         REQWRK(IoPA+M) = XCON * TOTMOL
C
C        SET "ATOM POPULATION" MATRIX FOR CONSTRAINTS
C
         DO 325 K = 1, KK
            IEQWRK(IoN+M+LoN*K) = 0.0D0
  325    CONTINUE
         IEQWRK(IoN+M+LoN*KCON) = 1.0D0
  350 CONTINUE
C
C     FROZEN COMPOSITION OPTION
C
      KFRZ = 0
      IEQWRK(IoKFRZ) = KFRZ
C
      DCS = 0.0D0
      CALL CKHML( T298, ICKWRK, RCKWRK, REQWRK(IoDHF0+1) )
      DO 500 K = 1, KK
C
C        SET THE SPECIES DATA POINTERS
C
         IEQWRK(IoJFS + K) = K
C
C        SET THE CONDENSED PHASE DENSITIES
C
         REQWRK(IoDCS + K) = DCS
C
C        SET ENTHALPY OF FORMATION (KCAL/MOLE)
C
         REQWRK(IoDHF0 + K) = REQWRK(IoDHF0 + K) * ER2CAL
C
C        SET THE MOLAR MASSES
C
         REQWRK(IoWMS + K) = REQWRK(NWT + K - 1)
500   CONTINUE
C
C         SET THE SELECTED PROPERTIES
C
      REQWRK(IoP) = REQWRK(NP1)
      IF ((NOP.EQ.2).OR.(NOP.EQ.3).OR.(NOP.EQ.7).OR.(NOP.EQ.9))
     1    REQWRK(IoP) = REQWRK(NPEST)
C
      REQWRK(IoT) = REQWRK(NT1)
      IF ((NOP.GE.4) .AND. (NOP.LE.9)) REQWRK(IoT) = REQWRK(NTEST)
C
      REQWRK(IoH) = REQWRK(NH1)
      REQWRK(IoS) = REQWRK(NS1)
      REQWRK(IoU) = REQWRK(NU1)
      REQWRK(IoV) = REQWRK(NV1)
      REQWRK(IoTE) = REQWRK(NTEST)
C
C         CALL STANJAN SJEQLB
C
       CALL SJROP( NAMAX, KK, NIWORK, NRWORK, NRADD,
     1             ATOM, KSYM, IEQWRK, REQWRK, REQWRK(NRWORK+1), NOP)
C
       IF (IEQWRK(IoKERR) .NE. 0) THEN
C           WRITE (LOUT,'(/A)') ' ERROR RETURNING FROM STANJAN !'
C           WRITE (LOUT,'(A,I3)') ' KERR =', IEQWRK(IoKERR)
C           STOP
C           write (lout,'(/A)') ' Unreacted values will be used'
            ierr = 1
           return
       ENDIF
C
C       CHECK CHANGE IN MEAN MOLECULAR WEIGHT
C
       TOTOLD = TOTMOL
       TOTMOL = 0.0D0
       DO 700 K = 1, KK
          TOTMOL = TOTMOL + REQWRK(IoSMOL+K)
700    CONTINUE
C
       DEL = ABS(TOTMOL - TOTOLD)/TOTMOL
       ITER = ITER + 1
       IF (ITER .GT. 20) THEN
           WRITE (6,*) 'ITERATION NOT CONVERGED!'
           WRITE (6,*) 'TOTMOL =', TOTMOL
C           STOP
           write (lout,'(/A)') ' Unreacted values will be used'
           ierr = 1
           return
       ENDIF
       IF (DEL .GT. 1.D-4) GO TO 310
C
C           RETRIEVE THE STANJAN RESULTS
C
      DO 800 K = 1, KK
         REQWRK(NX2 + K - 1) = REQWRK(IoXM+K)
         REQWRK(NY2 + K - 1) = REQWRK(IoYM+K)
800   CONTINUE
C
      REQWRK(NP2) = REQWRK(IoP)
      REQWRK(NT2) = REQWRK(IoT)
      REQWRK(NH2) = REQWRK(IoH)
      REQWRK(NU2) = REQWRK(IoU)
      REQWRK(NS2) = REQWRK(IoS)
      REQWRK(NV2) = REQWRK(IoV)
      REQWRK(NC2) = REQWRK(IoC)
      REQWRK(NWM2)= REQWRK(IoWM)
      REQWRK(NCDET) = REQWRK(IoCDET)
C
      RETURN
      END
C
C-------------------------------------------------------------------
C
      SUBROUTINE SJTIHS (NSMAX, NSWORK, SW, K, T, HMH0, S0)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
C      Gets HMH0 and S0 for the Kth file species at T.
C-------------------------------------------------------------------
C      Nomenclature:
C
C	   @ denotes a variable that must be loaded at call
C	   # denotes a variable available upon successful return
C
C      Variables in the argument list:
C @	   NSMAX       maximum number of species in system or file
C @	   NSWORK      dimension of work array SW
C @	   SWORK(I)    work array
C @	   K 	       species index in the data file
C @	   T	       temperature, K
C #	   HMH0        enthalpy above 298.15 K, kcal/mol
C #	   S0	       entropy, cal/mol-K
C-------------------------------------------------------------------
C
      DIMENSION SW(*)
      COMMON /IPAR/ NIWORK, NICMP, NIKNT, NRWORK, NRKNT, NRTMP, NRA,
     1              NRADD,  MAXTP, NCP,   NCP1,   NCP2,  NCP2T,
     2              NPHASE, NSMS,  NWT,   NXCON,  NKCON, NAMAX,
     3              NX1,    NX2,   NY1,   NY2,    NT1,   NT2,
     4              NP1,    NP2,   NV1,   NV2,    NWM1,  NWM2,
     5              NS1,    NS2,   NU1,   NU2,    NH1,   NH2,
     6              NC1,    NC2,   NCDET, NTEST,  NPEST
      COMMON /RPAR/ ER2CAL, RU, RUC, T298 
C
C         DETERMINE THE TEMPERATURE RANGE AT 298 AND T
C
      NTR = INT (SW(NRKNT + K - 1 - NRWORK)) - 1
      L0 = 1
      LT = 1
      DO 10 N = 2, NTR
         TEMP = SW(NRTMP + (K-1)*MAXTP + N - 1 - NRWORK)
         IF (T298 .GT. TEMP) L0=L0+1
         IF (T    .GT. TEMP) LT=LT+1
   10 CONTINUE
C
C         EVALUATE THE ENTHALPY AT T298 AND T
C
      NH = NRA + (L0-1)*NCP2 + (K-1)*NCP2T + NCP1 - 1 - NRWORK
      H298 = SW(NH) / T298
      NH = NRA + (LT-1)*NCP2 + (K-1)*NCP2T + NCP1 - 1 - NRWORK
      HMH0 = SW(NH) / T
      DO 100 N = 1, NCP
         NH = NRA + (L0-1)*NCP2 + (K-1)*NCP2T + N - 1 - NRWORK
         H298 = H298 + SW(NH) * T298**(N-1) / dble(N)
         NH = NRA + (LT-1)*NCP2 + (K-1)*NCP2T + N - 1 - NRWORK
         HMH0 = HMH0 + SW(NH) * T**(N-1) / dble(N)
100   CONTINUE
C
C         ENTHALPY ABOVE 298K (KCAL/MOL)
C
      HMH0 = (HMH0 * ER2CAL * RU * T) -
     1       (H298 * ER2CAL * RU * T298)
C
C         EVALUATE THE ENTROPY
C
      NS = NRA + (LT-1)*NCP2 + (K-1)*NCP2T + NCP2 - 1 - NRWORK
      S0 = SW(NS)
      NS = NRA + (LT-1)*NCP2 + (K-1)*NCP2T + 1 - 1 - NRWORK
      S0 = S0 + LOG(T) * SW(NS)
      DO 300 N = 2, NCP
         NS = NRA + (LT-1)*NCP2 + (K-1)*NCP2T + N - 1 - NRWORK
         S0 = S0 + SW(NS) * T**(N-1) / dble(N-1)
300   CONTINUE
C
C          ENTROPY (CAL/MOLE)
C
      S0 = S0 * RUC
C
      RETURN
      END
C
      SUBROUTINE EQSTART (LIN, LOUT, LENICK, LENRCK, LENCCK, LINKCK,
     1                    MDIM, KDIM, ICKWRK, RCKWRK, CCKWRK, MM, KK,
     2                    ATOM, KSYM, NOP, KMON, REAC, TEMP, TEST,
     3                    PRES, PEST, LCNTUE, NCON, KCON, XCON)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
C     Input:  LIN  - unit number for input data
C             LOUT - unit number for output messages
C             LENICK - length of CHEMKIN integer array
C             LENRCK - length of CHEMKIN real array
C             LENCCK - length of CHEMKIN character array
C             LINKCK - unit number for CHEMKIN linking file
C             MDIM   - dimension for atomic elements
C             KDIM   - dimension for molecular species
C     Output: ICKWRK - CHEMKIN integer array
C             RCKWRK - CHEMKIN real array
C             CCKWRK - CHEMKIN character array
C             MM     - total number of atomic elements
C             KK     - total number of molecular species
C             ATOM   - character array of element names
C             KSYM   - character array of species names
C             NOP    - integer equilibrium option number
C             KMON   - integer monitor flag
C             REAC   - real array of input mole fractions
C                      of the mixture
C             TEMP   - starting temperature for the problem
C             TEST   - estimated equilibrium temperature
C             PRES   - starting pressure for the problem
C             PEST   - estimated equilibrium pressure
C             LCNTUE - logical flag for continuation (initialize
C                      in calling program)
C             NCON   - integer number of species to be held constant
C             KCON   - integer array of index numbers of the species
C                      to be held constant
C             XCON   - real mole fractions input for the species to
C                      be held constant
C
      DIMENSION REAC(*), XCON(*), KCON(*), RCKWRK(*), ICKWRK(*)
      LOGICAL LCONH, LCONT, LCONP, LCONV, LCONS, LCONU, LCHAP,
     1        KERR, LCNTUE, IERR
      CHARACTER KEYWRD*4, LINE*76
      CHARACTER*(*) CCKWRK(*), ATOM(*), KSYM(*)
C
      KERR = .FALSE.
C
      IF (.NOT. LCNTUE) THEN
C
C            INITIALIZE VARIABLES
C
         DO 10 K = 1, KDIM
            REAC(K) = 0.0D0
10       CONTINUE
         TEMP = 0.0D0
         TEST = 0.0D0
         PRES = 0.0D0
         PEST = 0.0D0
         NCON = 0
         NOP  = 0
         KMON = 0
         LCONT = .FALSE.
         LCONP = .FALSE.
         LCONH = .FALSE.
         LCONU = .FALSE.
         LCONV = .FALSE.
         LCONS = .FALSE.
         LCHAP = .FALSE.
C
         CALL CKINIT (LENICK, LENRCK, LENCCK, LINKCK, LOUT,
     1                ICKWRK, RCKWRK, CCKWRK)
         CALL CKINDX (ICKWRK, RCKWRK, MM, KK, II, NFIT)
         IF (KDIM .GE. KK) THEN
            CALL CKSYMS (CCKWRK, LOUT, KSYM, IERR)
         ELSE
            WRITE (LOUT, *)
     1      ' ERROR...SPECIES DIMENSION MUST BE AT LEAST ',KK
            STOP
         ENDIF
         IF (MDIM .GE. MM) THEN
            CALL CKSYME (CCKWRK, LOUT, ATOM, IERR)
         ELSE
            WRITE (LOUT, *)
     1      ' ERROR...ELEMENT DIMENSION MUST BE AT LEAST ',MM
         ENDIF
      ENDIF
C
      LCNTUE = .FALSE.
C
      WRITE (LOUT,'(/A/)') '           KEYWORD INPUT '
C
90    CONTINUE
      KEYWRD = ' '
      LINE = ' '
      IERR = .FALSE.
      READ  (LIN,  7000) KEYWRD, LINE
      WRITE (LOUT, 8000) KEYWRD, LINE
      CALL UPCASE (KEYWRD)
C
C               IS THIS A KEYWORD COMMENT?
C
      IF (KEYWRD(1:1) .EQ. '.' .OR. KEYWRD(1:1) .EQ. '/' .OR.
     1    KEYWRD(1:1) .EQ. '!') GO TO 90
      IND = INDEX(LINE,'(')
      IF (IND .GT. 0) LINE(IND:) = ' '
      IND = INDEX(LINE,'!')
      IF (IND .GT. 0) LINE(IND:) = ' '
C
      IF (KEYWRD .EQ. 'DIAG') THEN
         KMON = 2
C
      ELSEIF (KEYWRD .EQ. 'CONH') THEN
C
C     Constant enthalpy (H) is used with constant volume (V) or
C     pressure (P), not with constant temperature (T), entropy (S),
C     internal energy (U), or Chapman-Jouguet:
C
         LCONH = .TRUE.
         LCONT = .FALSE.
         LCONS = .FALSE.
         LCONU = .FALSE.
         LCHAP = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'CONP') THEN
C
C     Constant pressure (P) is used with constant temperature (T),
C     volume (V), enthalpy (H) or entropy (S), not with internal
C     energy (U), or Chapman-Jouguet:
C
         LCONP = .TRUE.
         LCONU = .FALSE.
         LCHAP = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'CONT') THEN
C
C     Constant temperature (T) is used with constant pressure (P),
C     volume (V) or entropy (S), not with enthalpy (H), internal
C     energy (U), or Chapman-Jouguet:
C
         LCONT = .TRUE.
         LCONH = .FALSE.
         LCONU = .FALSE.
         LCHAP = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'CONX') THEN
C
C     Constant mole fraction (X)
        CALL CKSNUM (LINE, 1, LOUT, KSYM, KK, KSPEC, NVAL,
     1                VALUE, IERR)
        NCON = NCON + 1
        IF (IERR .OR. KSPEC.LE.0) THEN
           WRITE (LOUT,'(A)')
     1      ' ERROR READING DATA FOR KEYWORD '//KEYWRD
           KERR = .TRUE.
        ELSEIF (NCON .GT. KK-MM) THEN
           WRITE (LOUT,'(A,I3,A)')
     1      ' ERROR, CAN HAVE NO MORE THAN ', KK-MM, ' SPECIES'
           KERR = .TRUE.
        ELSE
           XCON(NCON) = VALUE
           KCON(NCON) = KSPEC
        ENDIF
C
      ELSEIF (KEYWRD .EQ. 'CONV') THEN
C
C     Constant volume (V) is used with constant temperature (T),
C     pressure (P), internal energy (U), enthalpy (H) or
C     entropy (S), but not with Chapman-Jouguet:
C
         LCONV = .TRUE.
         LCHAP = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'CONU') THEN
C
C     Costant internal energy (U) is used with constant volume (V)
C     only, not constant temperature (T), pressure (P),
C     enthalpy (H), or entropy (S), or with Chapman-Jouguet:
C
         LCONU = .TRUE.
         LCONT = .FALSE.
         LCONP = .FALSE.
         LCONH = .FALSE.
         LCONS = .FALSE.
         LCHAP = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'CONS') THEN
C
C     Constant entropy (S) is used with constant temperature (T),
C     pressure (P), or volume (V) only, not internal energy (U),
C     enthalpy (H), or Chapman-Jouguet:
C
         LCONS = .TRUE.
         LCONU = .FALSE.
         LCONH = .FALSE.
         LCHAP = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'CHAP') THEN
C
C     Chapman-Jouguet cancels any previous options:
C
         LCHAP = .TRUE.
         LCONT = .FALSE.
         LCONP = .FALSE.
         LCONH = .FALSE.
         LCONS = .FALSE.
         LCONU = .FALSE.
         LCONV = .FALSE.
C
      ELSEIF (KEYWRD .EQ. 'TEMP') THEN
C
C     Starting temperature (K)
C
         CALL CKXNUM (LINE, 1, LOUT, NVAL, TEMP, IERR)
         KERR = KERR.OR.IERR
C
      ELSEIF (KEYWRD .EQ. 'TEST') THEN
C
C     Estimate of equilibrium temperature (K)
C
         CALL CKXNUM (LINE, 1, LOUT, NVAL, TEST, IERR)
         KERR = KERR.OR.IERR
C
      ELSEIF (KEYWRD .EQ. 'PRES') THEN
C
C     Starting pressure (atm)
C
         CALL CKXNUM (LINE, 1, LOUT, NVAL, PRES, IERR)
         KERR = KERR.OR.IERR
         PRES = PRES
C
      ELSEIF (KEYWRD .EQ. 'PEST') THEN
C
C     Estimate of equilibrium pressure (atm)
C
         CALL CKXNUM (LINE, 1, LOUT, NVAL, PEST, IERR)
         KERR = KERR.OR.IERR
         PEST = PEST
C
      ELSEIF (KEYWRD .EQ. 'REAC') THEN
C
C     Reactant species
C
         CALL CKSNUM (LINE, 1, LOUT, KSYM, KK, KSPEC, NVAL,
     1                VALUE, IERR)
         IF (IERR .OR. KSPEC.LE.0) THEN
            WRITE (LOUT,'(A)')
     1      ' ERROR READING DATA FOR KEYWORD '//KEYWRD
            KERR = .TRUE.
         ELSE
            REAC(KSPEC) = VALUE
         ENDIF
C
      ELSEIF (KEYWRD .EQ. 'CNTN') THEN
C
C     More jobs to follow
C
         LCNTUE = .TRUE.
C
      ELSEIF (KEYWRD .EQ. 'END') THEN
         GO TO 1000
      ELSE
C
C     Unknown keyword
C
         WRITE (LOUT, *) ' Unrecognized input...',KEYWRD
C
      ENDIF
      GO TO 90
C
 1000 CONTINUE
C
C     Check for completeness
C
      IF (LCONT) THEN
         NREQ = 0
         IF (LCONP) THEN
            NOP = 1
            NREQ = NREQ + 1
         ENDIF
         IF (LCONV) THEN
            NOP = 2
            NREQ = NREQ + 1
         ENDIF
         IF (LCONS) THEN
            NOP = 3
            NREQ = NREQ + 1
         ENDIF
         IF (NREQ .GT. 1) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use only one of CONP, CONV, or CONS in',
     2      ' conjunction with CONT...'
         ELSEIF (NREQ .EQ. 0) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use one of CONP, CONV, or CONS in',
     2      ' conjunction with CONT...'
         ENDIF
      ENDIF
C
      IF (LCONP) THEN
         NREQ = 0
         IF (LCONT) THEN
            NOP = 1
            NREQ = NREQ + 1
         ENDIF
         IF (LCONV) THEN
            NOP = 4
            NREQ = NREQ + 1
         ENDIF
         IF (LCONH) THEN
            NOP = 5
            NREQ = NREQ + 1
         ENDIF
         IF (LCONS) THEN
            NOP = 6
            NREQ = NREQ + 1
         ENDIF
         IF (NREQ .GT. 1) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use only one of CONT, CONV, CONH or',
     2      ' CONS in conjunction with CONP...'
         ELSEIF (NREQ .EQ. 0) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use one of CONT, CONV, CONH or',
     2      ' CONS in conjunction with CONP...'
         ENDIF
      ENDIF
C
      IF (LCONV) THEN
         NREQ = 0
         IF (LCONT) THEN
            NOP = 2
            NREQ = NREQ + 1
         ENDIF
         IF (LCONP) THEN
            NOP = 4
            NREQ = NREQ + 1
         ENDIF
         IF (LCONU) THEN
            NOP = 7
            NREQ  = NREQ + 1
         ENDIF
         IF (LCONH) THEN
            NOP = 8
            NREQ = NREQ + 1
         ENDIF
         IF (LCONS) THEN
            NOP = 9
            NREQ = NREQ + 1
         ENDIF
         IF (NREQ .GT. 1) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use only one of CONT, CONU, CONH or',
     2      ' CONS in conjunction with CONV...'
         ELSEIF (NREQ .EQ. 0) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use one of CONT, CONU, CONH or',
     2      ' CONS in conjunction with CONV...'
         ENDIF
      ENDIF
C
      IF (LCONU) THEN
         IF (LCONV) THEN
            NOP = 7
         ELSE
            WRITE (LOUT, *) ' Error...must use CONV in conjunction',
     1                      ' with CONU...'
            KERR = .TRUE.
         ENDIF
      ENDIF
C
      IF (LCONH) THEN
         NREQ = 0
         IF (LCONP) THEN
            NOP = 5
            NREQ = NREQ + 1
         ENDIF
         IF (LCONV) THEN
            NOP = 8
            NREQ = NREQ + 1
         ENDIF
         IF (NREQ .GT. 1) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use only one of CONP or CONV in',
     2      ' conjunction with CONH...'
         ELSEIF (NREQ .EQ. 0) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use one of CONP or CONV in',
     2      ' conjunction with CONH...'
         ENDIF
      ENDIF
C
      IF (LCONS) THEN
         NREQ = 0
         IF (LCONT) THEN
            NOP = 3
            NREQ = NREQ + 1
         ENDIF
         IF (LCONP) THEN
            NOP = 6
            NREQ = NREQ + 1
         ENDIF
         IF (LCONV) THEN
            NOP = 9
            NREQ = NREQ + 1
         ENDIF
         IF (NREQ .GT. 1) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use only one of CONT, CONP or CONV in',
     2      ' conjunction with CONS...'
         ELSEIF (NREQ .EQ. 0) THEN
            KERR = .TRUE.
            WRITE (LOUT, *)
     1      ' Error...must use one of CONT, CONP, or CONV in',
     2      ' conjunction with CONS...'
         ENDIF
      ENDIF
C
      IF (LCHAP) THEN
         NOP = 10
         IF (LCONT.OR.LCONP.OR.LCONV.OR.LCONS.OR.LCONH.OR.LCONU)
     1   THEN
            KERR = .TRUE.
            WRITE (LOUT, *) ' Error...cannot use CHAP in conjunction',
     1      ' with other equilibrium type keywords...'
         ENDIF
      ENDIF
C
      IF (TEMP .LE. 0.0) THEN
         KERR = .TRUE.
         WRITE (LOUT, *) ' Error...no TEMP provided...'
      ENDIF
      IF (PRES .LE. 0.0) then
         KERR = .TRUE.
         WRITE (LOUT, *) ' Error...no PRES provided...'
      ENDIF
C
      IF ((NOP.EQ.2 .OR. NOP.EQ.3 .OR. NOP.EQ.7 .OR. NOP.EQ.8 .OR.
     1    NOP.EQ.9) .AND. PEST.LE.0.0D0) PEST = PRES
C
      IF (NOP.GT.3 .AND. TEST.LE.0.0D0) TEST = TEMP
C
      IF (KERR) THEN
         WRITE (LOUT, *) ' STOP due to errors in input..'
         STOP
      ENDIF
C
      WRITE (LOUT, *)
      IF (NOP .EQ. 1) THEN
         WRITE (LOUT, *) ' Constant temperature and pressure problem: '
      ELSEIF (NOP .EQ. 2) THEN
         WRITE (LOUT, *) ' Constant temperature and volume problem: '
      ELSEIF (NOP .EQ. 3) THEN
         WRITE (LOUT, *) ' Constant temperature and entropy problem: '
      ELSEIF (NOP .EQ. 4) THEN
         WRITE (LOUT, *) ' Constant pressure and volume problem: '
      ELSEIF (NOP .EQ. 5) THEN
         WRITE (LOUT, *) ' Constant pressure and enthalpy problem: '
      ELSEIF (NOP .EQ. 6) THEN
         WRITE (LOUT, *) ' Constant pressure and entropy problem: '
      ELSEIF (NOP .EQ. 7) THEN
         WRITE (LOUT, *) ' Constant volume and energy problem: '
      ELSEIF (NOP .EQ. 8) THEN
         WRITE (LOUT, *) ' Constant volume and enthalpy problem: '
      ELSEIF (NOP .EQ. 9) THEN
         WRITE (LOUT, *) ' Constant volume and entropy problem: '
      ELSEIF (NOP .EQ. 10) THEN
         WRITE (LOUT, *) ' Chapman=Jouguet detonation: '
      ENDIF
C
      RETURN
C
 7000 FORMAT (A4, A76)
 8000 FORMAT (1X, A4, A76)
      END
C
      SUBROUTINE EQLEN (MM, KK, NCON, NITOT, NRTOT)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      COMMON /IPAR/ NIWORK, NICMP, NIKNT, NRWORK, NRKNT, NRTMP, NRA,
     1              NRADD,  MAXTP, NCP,   NCP1,   NCP2,  NCP2T,
     2              NPHASE, NSMS,  NWT,   NXCON,  NKCON, NAMAX,
     3              NX1,    NX2,   NY1,   NY2,    NT1,   NT2,
     4              NP1,    NP2,   NV1,   NV2,    NWM1,  NWM2,
     5              NS1,    NS2,   NU1,   NU2,    NH1,   NH2,
     6              NC1,    NC2,   NCDET, NTEST,  NPEST
C
      NCP  = 5
      NCP1 = 6
      NCP2 = 7
      NCP2T= 14
      MAXTP = 3
C
C    If calling SJROP:
C
      NAMAX = MM + NCON
      NPHASE = 1
      NPMAX = NPHASE
      NSMAX = KK
      NWORK = MAX(2*NAMAX, NAMAX+NPMAX)
C
C    If using a species data file or calling SJSET:
      NIWORK = 22 + 14*NAMAX + 4*NPMAX + 8*NSMAX + 2*NAMAX*NSMAX
      NRWORK = 24 + 16*NAMAX + 12*NAMAX*NAMAX + 3*NPMAX*NAMAX + 6*NPMAX
     1            + 18*NSMAX + NWORK*NWORK + NWORK
C
C     pointers for additional integer arrays
C
      NICMP = NIWORK + 1
      NIKNT = NICMP + NAMAX * NSMAX
      NKCON = NIKNT + NSMAX
      NITOT = NKCON + NSMAX - 1
C
C     pointers for additional real arrays
C
      NRKNT = NRWORK + 1
      NRTMP = NRKNT + NSMAX
      NRA   = NRTMP + MAXTP * NSMAX
      NSMS  = NRA   + NCP2T * NSMAX
      NWT   = NSMS  + NSMAX
      NXCON = NWT   + NSMAX
      NX1   = NXCON + NSMAX
      NX2   = NX1   + NSMAX
      NY1   = NX2   + NSMAX
      NY2   = NY1   + NSMAX
      NT1   = NY2   + NSMAX
      NT2   = NT1   + 1
      NP1   = NT2   + 1
      NP2   = NP1   + 1
      NS1   = NP2   + 1
      NS2   = NS1   + 1
      NU1   = NS2   + 1
      NU2   = NU1   + 1
      NV1   = NU2   + 1
      NV2   = NV1   + 1
      NWM1  = NV2   + 1
      NWM2  = NWM1  + 1
      NH1   = NWM2  + 1
      NH2   = NH1   + 1
      NC1   = NH2   + 1
      NC2   = NC1   + 1
      NCDET = NC2   + 1
      NTEST = NCDET + 1
      NPEST = NTEST + 1
C
      NRTOT = NPEST + 1 - 1
      NRADD = NRTOT - NRWORK
C
      RETURN
      END
C
      SUBROUTINE EQPRNT (LOUT, LSAVE, LCNTUE, NOP, KK, KSYM, REQWRK)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION REQWRK(*)
      CHARACTER KSYM(*)*(*)
      LOGICAL LCNTUE
      COMMON /IPAR/ NIWORK, NICMP, NIKNT, NRWORK, NRKNT, NRTMP, NRA,
     1              NRADD,  MAXTP, NCP,   NCP1,   NCP2,  NCP2T,
     2              NPHASE, NSMS,  NWT,   NXCON,  NKCON, NAMAX,
     3              NX1,    NX2,   NY1,   NY2,    NT1,   NT2,
     4              NP1,    NP2,   NV1,   NV2,    NWM1,  NWM2,
     5              NS1,    NS2,   NU1,   NU2,    NH1,   NH2,
     6              NC1,    NC2,   NCDET, NTEST,  NPEST
C
      WRITE (LOUT, 7200)
     1 REQWRK(NP1), REQWRK(NP2), REQWRK(NT1), REQWRK(NT2),
     2 REQWRK(NV1), REQWRK(NV2), REQWRK(NH1), REQWRK(NH2),
     3 REQWRK(NU1), REQWRK(NU2), REQWRK(NS1), REQWRK(NS2),
     4 REQWRK(NWM1), REQWRK(NWM2)
C
      WRITE (LOUT, 7210)
      WRITE (LOUT, 7215) (KSYM(K), REQWRK(NX1 + K - 1),
     1                             REQWRK(NX2 + K - 1), K=1,KK)
C
      IF (NOP .EQ. 10)
     1   WRITE (LOUT, 7100) REQWRK(NC1), REQWRK(NC2), REQWRK(NCDET),
     2                      REQWRK(NCDET) / REQWRK(NC1)
C
      IF (LSAVE .GT. 0)
C
C     WRITE BINARY RECORD OF STARTING AND EQUILIBRIUM VALUES
C
     *WRITE (LSAVE) REQWRK(NT1), REQWRK(NP1), REQWRK(NU1), REQWRK(NH1),
     1              REQWRK(NS1), REQWRK(NV1), REQWRK(NC1),
     2              (REQWRK(NX1 + K - 1), K = 1, KK),
     3              REQWRK(NT2), REQWRK(NP2), REQWRK(NU2), REQWRK(NH2),
     4              REQWRK(NS2), REQWRK(NV2), REQWRK(NC2),
     5              (REQWRK(NX2 + K - 1), K = 1, KK)
C
C
      IF (LCNTUE) THEN
         WRITE (LOUT, *)
         WRITE (LOUT, *) '*************************',
     1                   'CONTINUING TO NEW PROBLEM',
     2                   '*************************'
         WRITE (LOUT, *)
      ENDIF
C
C      FORMATS
C
 7200 FORMAT (/1X,T20, 'INITIAL STATE:',T40,'EQUILIBRIUM STATE:',
     1        /1X,'P (atm)       ',T20,1PE14.4,T40,1PE14.4,
     2        /1X,'T (K)         ',T20,1PE14.4,T40,1PE14.4,
     3        /1X,'V (cm3/gm)    ',T20,1PE14.4,T40,1PE14.4,
     4        /1X,'H (erg/gm)    ',T20,1PE14.4,T40,1PE14.4,
     5        /1X,'U (erg/gm)    ',T20,1PE14.4,T40,1PE14.4,
     6        /1X,'S (erg/gm-K)  ',T20,1PE14.4,T40,1PE14.4,
     7        /1X,'W (gm/mole)   ',T20,1PE14.4,T40,1PE14.4)
 7210 FORMAT(//1X,'Mole Fractions')
 7215 FORMAT ( 1X, A16,            T20,1PE14.4,T40,1PE14.4)
 7100 FORMAT (/1X,'C-J DETONATION PROPERTIES',
     1        /1X,'C (cm/s)      ',T20,1PE14.4,T40,1PE14.4,
     2        /1X,'CDET (cm/s) ',T40, 1PE14.4,
     3        /1X,'MACH ',T40,F10.6 )
C
      RETURN
      END
C
      SUBROUTINE EQSOL (KK, REQWRK, X, Y, T, P, H, V, S, WM, C, CDET)
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION REQWRK(*), X(*), Y(*)
      COMMON /IPAR/ NIWORK, NICMP, NIKNT, NRWORK, NRKNT, NRTMP, NRA,
     1              NRADD,  MAXTP, NCP,   NCP1,   NCP2,  NCP2T,
     2              NPHASE, NSMS,  NWT,   NXCON,  NKCON, NAMAX,
     3              NX1,    NX2,   NY1,   NY2,    NT1,   NT2,
     4              NP1,    NP2,   NV1,   NV2,    NWM1,  NWM2,
     5              NS1,    NS2,   NU1,   NU2,    NH1,   NH2,
     6              NC1,    NC2,   NCDET, NTEST,  NPEST
C
C     RETURNS THE EQUILIBRIUM SOLUTION IN CGS UNITS
C
      DO 50 K = 1, KK
         X(K) = REQWRK(NX2 + K - 1)
         Y(K) = REQWRK(NY2 + K - 1)
   50 CONTINUE
      P = REQWRK(NP2)
      T = REQWRK(NT2)
      V = REQWRK(NV2)
      WM = REQWRK(NWM2)
      S = REQWRK(NS2)
      H = REQWRK(NH2)
      U = REQWRK(NU2)
      C = REQWRK(NC2)
      CDET = REQWRK(NCDET)
C
      RETURN
      END
C
      SUBROUTINE UPCASE (STR)
C
      CHARACTER LOWER(26)*1, UPPER(26)*1, STR*(*)
      DATA LOWER/'a','b','c','d','e','f','g','h','i','j','k','l',
     1           'm','n','o','p','q','r','s','t','u','v','w','x',
     2           'y','z'/
      DATA UPPER/'A','B','C','D','E','F','G','H','I','J','K','L',
     1           'M','N','O','P','Q','R','S','T','U','V','W','X',
     2           'Y','Z'/
      DO 10 L=1,LEN(STR)
         DO 10 N=1,26
            IF (STR(L:L) .EQ. LOWER(N)) STR(L:L) = UPPER(N)
   10 CONTINUE
      RETURN
      END
