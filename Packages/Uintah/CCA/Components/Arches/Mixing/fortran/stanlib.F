c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJEQLB(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW,KINIT)
c
c      Equilibrium solution by element potentials (and initialization).
c----------------------------------------------------------------------
c      Solves for the element potentials and phase mols that meet the
c      atomic constraints.  Uses the dual problem for
c
c           W = sum{PMOL(M)*[Z(M)-1]} - sum{ELAM(K)*PA(IB(K)}
c
c      The atomic constraints are satisfied when W is a minimum with
c      respect to variations in ELAMs at fixed PMOLs.  For any such
c      state the Z constraints are satisfied when this W = W* = Y is
c      maximized with respect to PMOLs.  This forms the basis for the
c      steepest descent/ascent algorithms used below. Newton-Raphson
c      iterations are used near the extremum states.
c
c      The basic element potential mol fraction generator
c
c             X(J) = exp[-G(J) + sum{ELAM(K)*N(K,J)}]
c
c      is used to calculate the phase mol fractions.
c
c      Works in three modes:
c
c          Mode 1: Steepest descent in ELAM space at fixed PMOLs
c                  followed by steepest ascent in PMOL space
c
c          Mode 2: Newton-Raphson iteration on ELAMs at fixed PMOLs
c                  followed by steepest ascent in PMOL space
c
c          Mode 3: Newton-Raphson iteration on ELAMs and PMOLs
c
c          Problems:
c
c              KERR = 1 solution failed due to singularity
c              KERR = 2 solution failed to converge
c
c      Initialization:
c
c          If KINIT = 1  at call, the initializer is called to set up
c          the problem and estimate the element potentials and phase
c          mols.
c
c          If KINIT = 0 at call, the current values are used to start.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c @        KINIT       initialization control (initialize if 1)
c
c      Variables in the integer work array IW:
c #        IB(K) = I   if the Kth independent atom is the Ith system
c                      atom
c #        JB(K) = J   if the Kth base is the Jth species
c #        KB(J)       0 if Jth species is not a base
c                      1 if the Jth species is a base
c                      4 if the Jth species is excluded
c #        KERR        error flag
c @        KMON        monitor control
c -        KPC(M)      0 if phase not active, 1 if active
c -        KPCX(M)     KPC for provisional new phase distribution
c #        KTRE        main pass counter
c @        KUMO        output unit for monitor
c #        MPA(L) = M  if Mth phase is the Lth active phase
c #        MPJ(J) = M  if the Jth species is in the Mth phase
c @        N(I,J)      number of Ith atoms in the Jth species
c @        NA          number of atom types in the system
c #        NB          number of independent atoms (basis species)
c @        NP          number of allowed phases
c @        NS          number of species
c @        NSP(M)      number of species in the Mth phase
c
c      Variables in the real work array RW:
c -        A(K,L)      work matrix
c #        CM(K,M)     conditioning matrix
c -        D(K,M)      sum{N(IB(K),J)*X(J)} for Mth phase
c -        DC(K,M)     conditioned D vector for Mth phase
c -        DLAM(K)     change in ELAM(K) when PMOLs held constant
c -        DLAY(K)     change in ELAM(K) associated with PMOLs changes
c -        DPML(M)     change in PMOL(M)
c -        E(K,M)      satisfies sum{QC(K,L)*E(L,M)} = - DC(K,M)
c #        ELAM(K)     element potential for the Kth independent atom
c @        FRND        roundoff factor
c @        G(J)        g(T,P)/RT for the Jth species
c -        HA(K)       sum{PMOL(M)*D(K,M)}-PA(IB(K))
c                      (zero for solution)
c -        HC(K)       conditioned H vector
c @        PA(I)       population of Ith atom
c #        PC(K)       conditioned  populations SUM{CM(K,L)*PA(IB(L))}
c #        PMOL(M)     mols of Mth phase
c -        Q(K,L)      sum{PMOL(M)*SUM{(Mth phase)
c                         N(IB(K),J)*N(IB(L),J)*X(J)}}
c -        QC(K,L)     conditioned Q matrix
c -        RL(K)       direction cosines for steepest descent in ELAM
c                         space
c -        RP(M)       direction cosines for steepest descent in PMOL
c                         space
c -        SMOL(J)     Jth species mols  (plus false species in SJISMP)
c                  (FINAL VALUES NOT COMPUTED BY SJEQLB!)
c -        W(K)        work vector
c #        X(J)        mol fraction of the Jth species
c -        Z(M)        sum{X(J)} over species in Mth phase
c
c      Variables used only internally:
c          AEZM        maximum error in Z(M) - 1 for an active phase
c          AEZMD       value of AEZM above which a mode downgrade is
c                      made
c          AEZMU       value of AEZM below which a mode upgrade is
c                         attempted
c          AEZMF       factor for error tolerance in mode 1
c          BETA2       sum{HA(K)*HA(K)} = Beta-squared (sum over bases)
c          BETA        -sqrt[BETA2] = dW/ds for steepest descent
c          CALL1       logical key; first or subsequent call
c          DLAMAX      maximum change allowed in ELAMs for
c                      Newton-Raphson
c          DS          path length in ELAM space
c          DSUP        DS below which upgrade from more 1 is attempted
c          DWDS        dW/ds on steepest descent path
c          DYDS        dW*/ds* on steepest ascent path
c          ERRH        allowed fractional error in any H
c          ERRHT       maximum tolerant  ERRH
c          ERRZ        allowed fractional error in Z
c          FDW         damping factor to suppress W valley oscillations
c          FDY         damping factor to suppress Y valley oscillations
c          FDPMA       maximum PMOL fractional change in W* ascent
c          FDPMC       fraction of PMOL change used for illegal phase
c                         demise
c          FDPMI3      maximum PMOL fractional (of PMMAX) increase in
c                         mode 3
c          FDPMD3      maximum PMOL fractional (of PMOL) decrease in
c                         mode 3
c          FDPMV       fraction of illegal vanishing change accepted
c          FDPYR       fraction for reduction of a large phase seeking
c                         absence
c          FDPYV       fraction of largest PMOL below which a phase may
c                         vanish
c          FDPY        maximum fraction of max PMOL for phase
c                      activation
c          KTRH        number of times the LAM changes have been halved
c          KTRM        number of passes made in the current mode
c          KTRMAX      maximum KTRE allowed
c          KTRY        Y (mols adjustment) descent counter
c          KUP         mode upgrade control
c          MODE        mode number
c          MS          matrix size in phase mols adjustment
c          PMMAX       maximum PMOL
c          REHM        maximum relative error found for any HA(K)
c          REHCM       maximum relative error found for any HC(K)
c          REHMD       value of REHM above which a mode downgrade is
c                      made
c          REHMU       value of REHM below which a mode upgrade is
c                         attempted
c          WF          sum{PMOL(M)*[Z(M)-1]} - sum{ELAM(K)*PA(IB(K)}
c          WP          WF value on previous pass
c          Y           WF at the minimum in ELAM at fixed PMOL;
c                         also called W*
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
       common /kcallblock/ kcall
c-----------------------------------------------------------------------
c    iteration limit
       PARAMETER (KTRMAX=200)
c    first call indicator
       SAVE
       DATA KCALL/1/

c-----------------------------------------------------------------------
c    set comparison constants in computation precision
       IF (KCALL.NE.0)  THEN
           KCALL = 0
c        error parameters for mode change
           AEZMD = 0.3D0
           AEZMU = 0.2D0
           REHMD = 0.4D0
           REHMU = 0.3D0
           DSUP  = 0.3D0
c        tolerant error parameters
           AEZMF = 1.0D-2
           ERRHT = 1.0D-3
c        final convergence error limits
           ERRZ = 1.0D-10
           ERRH = 1.0D-8
c        change limits
           DLAMAX = 2.0D0
           FDPMA = 0.5D0
           FDPMI3 = 0.3D0
           FDPMD3 = 0.3D0
c        W* ascent change limits
           FDPY = 0.3D0
           FDPYR = 0.8D0
           FDPYV = 0.02D0
           FDPMV = 0.8D0
           FDPMC = 0.9D0
c        constants
           ONE = 1.0D0
           ZERO = 0.0D0
       ENDIF
c
c-----------------------------------------------------------------------
c    get parameters
       NA = IW(IoNA)
       NP = IW(IoNP)
       NS = IW(IoNS)
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
c
c    monitor
       IF (KMON.GT.1)  THEN
           WRITE (KUMO,1)
1          FORMAT (/' Equilibrium solution monitor:'//
     ;              '   Atom      mols')
           WRITE (KUMO,2) (ATOM(I),RW(IoPA+I),I=1,NA)
2          FORMAT (4X,A,1PE14.4)
           WRITE (KUMO,3) (ATOM(I),I=1,NA)
3          FORMAT (/' Species      g/RT  ',20(1X,A2))
           DO 5 J=1,NS
              WRITE (KUMO,4) CHEM(J),RW(IoG+J),(IW(IoN+I+LoN*J),I=1,NA)
4             FORMAT (1X,A8,F10.3,(20I3))
5             CONTINUE
       ENDIF
c
c    initialization
       IF (KINIT.NE.0) THEN
c            set species phase cross-reference
               J2 = 0
               DO 7 M=1,NP
                   J1 = J2 + 1
                   J2 = J2 + IW(IoNSP+M)
                   DO 6 J=J1,J2
                       IW(IoMPJ+J) = M
6                      CONTINUE
7                  CONTINUE
c            estimate phase mols and element potentials
               CALL SJINIT(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW)
               IF (IW(IoKERR).NE.0)  RETURN
c            load NB
               NB = IW(IoNB)
           ELSE
c            estimate element potentials using current mol fractions
               K = 1
               CALL SJIESL(NSMAX,NIW,NRW,CHEM,IW,RW,K,DUMMY,IDUMMY)
               IF (IW(IoKERR).NE.0)  RETURN
           ENDIF
c
c ** initialize counters and controls for the element potential
c    solution
c
c    set counters
       IW(IoKTRE) = 0
       KTRM = 0
       KTRH = 0
       KTRY = 0
c    set to start in mode 3
       MODE = 3
       KUP = 0
c    set initial damping for steepest descent
       FDW = 1.0D0
       FDY = 1.0D0
c    set active phases
       DO 9 M=1,NP
           IF (RW(IoPMOL+M).NE.ZERO)  THEN
                    IW(IoKPC+M) = 1
                ELSE
                    IW(IoKPC+M) = 0
                ENDIF
9          CONTINUE
c
c **** Loop point for ELAM and PMOL adjustments  *****************
c
c    check for convergence failure
10     IF (IW(IoKTRE).GT.KTRMAX) THEN
           IW(IoKERR) = 2
           RETURN
           ENDIF
c
c    increment counters
       IW(IoKTRE) = IW(IoKTRE) + 1
       KTRM = KTRM + 1
c
c ** compute mol fractions and Z  for each phase and W
c
c    initialize
       AEZM = 0
       MAEZM = 0
       WF = 0
c    phase contributions
       J2 = 0
       DO 19  M=1,NP
           J1 = J2 + 1
           J2 = J2 + IW(IoNSP+M)
           IoZM = IoZ + M
           RW(IoZM) = 0.0D0
c        species contributions
           DO 17 J=J1,J2
c            check for inclusion
               IF (IW(IoKB+J).NE.4)  THEN
                   SUM = - RW(IoG+J)
c                element potential contributions
                   DO 15 K=1,NB
                       I = IW(IoIB+K)
                       SUM = SUM + RW(IoELAM+K)*IW(IoN+I+LoN*J)
15                     CONTINUE
                   RW(IoX+J) = SJUEXP(SUM)
                   RW(IoZM) = RW(IoZM) + RW(IoX+J)
                   ENDIF
17             CONTINUE
           WF = WF + RW(IoPMOL+M)*RW(IoZM)
c
c        compute Z error
           AEZ = RW(IoZM) - 1.0D0
c        check vs maximum
           IF (IW(IoKPC+M).NE.0)  THEN
c                 the phase is active
                   IF (ABS(AEZ).GT.AEZM)  THEN
                       AEZM = ABS(AEZ)
                       MAEZM = M
                       ENDIF
               ELSE
c                 if Z ge 1 the phase should be active
                   IF (AEZ.GE.ZERO)  AEZM = AEZ
               ENDIF
c
19         CONTINUE
c
c ** calculate  D,DC, HA, HC, and QC and evaluate the relative errors
c
c    initialize
       REHM = 0.0D0
       REHCM = 0.0D0
       DO 29 K =1,NB
c        population term in WF
           IK = IW(IoIB+K)
           WF = WF - RW(IoELAM+K)*RW(IoPA+IK)
c        initialize
           IoHAK = IoHA + K
           IoHCK = IoHC + K
           IoPAIK = IoPA+IK
           IoPCK = IoPC + K
           RW(IoHAK) = - RW(IoPAIK)
           RW(IoHCK) = - RW(IoPCK)
           TERMX = ABS(RW(IoPAIK))
           TERMCX = ABS(RW(IoPCK))
           DO 21 L=1,NB
               RW(IoQC+K+LoQC*L) = 0.0D0
21             CONTINUE
c        phase contributions
           J2 = 0
           DSUM = 0.0D0
           DO 27 M=1,NP
               J1 = J2 + 1
               J2 = J2 + IW(IoNSP+M)
               IoDKM = IoD + K + LoD*M
               IoDCKM = IoDC + K + LoDC*M
               RW(IoDKM) = 0.0D0
               RW(IoDCKM) = 0.0D0
               TDM = 0.0D0
               TDCM = 0.0D0
c            species contributions
               DO 25 J=J1,J2
c                check for inclusion
                   IF (IW(IoKB+J).NE.4)  THEN
c                    contributions to D and H and major terms in H
                       TERM = IW(IoN+IK+LoN*J)*RW(IoX+J)
                       CALL SJUMAX(TERM,TDM)
                       RW(IoDKM) = RW(IoDKM) + TERM
c                    check for active phase
                       IF (IW(IoKPC+M).NE.0)  THEN
                           TERM = TERM*RW(IoPMOL+M)
                           CALL SJUMAX(TERM,TERMX)
                           RW(IoHAK) = RW(IoHAK) + TERM
                           ENDIF
c                    contributions to conditioned D, H, and Q
                       IF ((IW(IoKB+J).NE.1).OR.(IW(IoJB+K).EQ.J)) THEN
                           IF (IW(IoKB+J).EQ.1)  THEN
c                                species J is the Kth base species
                                   TERM = 1.0D0
                               ELSE
c                                species J is not a base
                                   TERM = 0.0D0
                                   DO 23 L=1,NB
                                       IL = IW(IoIB+L)
                                       TERM = TERM + RW(IoCM+K+LoCM*L)*
     ;                                               IW(IoN+IL+LoN*J)
23                                     CONTINUE
                               ENDIF
                           TERM = TERM*RW(IoX+J)
                           CALL SJUMAX(TERM,TDCM)
                           RW(IoDCKM) = RW(IoDCKM) + TERM
                           IF (IW(IoKPC+M).NE.0)  THEN
                               TERM = TERM*RW(IoPMOL+M)
                               CALL SJUMAX(TERM,TERMCX)
                               RW(IoHCK) = RW(IoHCK) + TERM
                               DO 24 L=1,NB
                                   IL = IW(IoIB+L)
                                   IoQCKL = IoQC + K + LoQC*L
                                   RW(IoQCKL) = RW(IoQCKL) +
     ;                                 TERM*IW(IoN+IL+LoN*J)
24                                 CONTINUE
                               ENDIF
                           ENDIF
                       ENDIF
25                 CONTINUE
               CALL SJURND(RW(IoDKM),TDM)
               CALL SJURND(RW(IoDCKM),TDCM)
27             CONTINUE
c
c        compute relative errors in HA(K) and compare vs maximum
           IF (TERMX.NE.ZERO)  THEN
                   REH = ABS(RW(IoHAK))/TERMX
               ELSE
                   REH = 0.0D0
               ENDIF
           CALL SJUMAX(REH,REHM)
c
c        compute relative errors in HC(K) and compare vs maximum
           IF (TERMCX.NE.ZERO)  THEN
                   REHC = ABS(RW(IoHCK))/TERMCX
               ELSE
                   REHC = 0.0D0
               ENDIF
           CALL SJUMAX(REHC,REHCM)
c
29         CONTINUE
c
c ** monitor
c
c    instructional monitor
       IF (KMON.GT.1)  THEN
           WRITE (KUMO,60) IW(IoKTRE),WF
60         FORMAT (/' Equilibrium solution pass ',I3,
     ;              ';   dual function W =',1PE20.12)
           WRITE (KUMO,61) (M,RW(IoPMOL+M),RW(IoZ+M),M=1,NP)
61         FORMAT  ('   phase ',I2,' mols =',1PE15.8,
     ;              ';  mol fraction sum Z =',E20.12)
           DO 63 K=1,NB
               I = IW(IoIB+K)
               WRITE (KUMO,62) ATOM(I),RW(IoELAM+K),RW(IoHA+K)
62             FORMAT  ('   element potential for ',A2,' =',1PE20.12,
     ;                  '; population error =',E11.3)
63             CONTINUE
           WRITE (KUMO,64) REHCM
64         FORMAT('   maximum conditioned population relative error =',
     ;             1PE12.4)
           DO 69 J1=1,NS,6
               J2 = J1 + 5
               IF (J2.GT.NS) J2 = NS
               WRITE (KUMO,65) (CHEM(J),J=J1,J2)
65             FORMAT  (6X,6(4X,A8))
               WRITE (KUMO,67) (RW(IoX+J),J=J1,J2)
67             FORMAT  ('    X:',6E12.5)
69             CONTINUE
           ENDIF
c
c **** initial convergence check (superaccurate convergence)
c
       IF ((AEZM.LT.RW(IoFRND)).AND.(REHCM.LT.RW(IoFRND))) GOTO 800
c
c **** mode selection/revision ********
c
c    set error limit for constraint convergence
c    (allow more error if Zs are not close)
       ERRHX  = ERRH + AEZM*AEZMF
       IF (ERRHX.GT.ERRHT)  ERRHX = ERRHT
c
c    branch on current mode
       GOTO (110,120,130),  MODE
c
c ** mode 1:  Steepest descent of W in ELAM space at fixed PMOL
c             followed by steepest descent of Y in PMOL space
c
110    IF (KTRM.GT.1)  THEN
c        previous pass also in mode 1; check for W increase
           IF (WF.GT.WP)  THEN
c           WF increased:  check mode trial count
               IF (KTRH.GT.10)  THEN
c                not getting anywhere:  give up
                   IW(IoKERR) = 2
                   RETURN
                   ENDIF
c            cut ELAM changes in half and try again
               DO 117 K=1,NB
                   IoDLAK = IoDLAM + K
                   RW(IoDLAK) = 0.5D0*RW(IoDLAK)
                   RW(IoELAM+K) = RW(IoELAM+K) - RW(IoDLAK)
117                CONTINUE
               KTRH = KTRH + 1
               KUP = 2
               IF (KMON.GT.1)  WRITE (KUMO,118)
118            FORMAT (' W overshoot;',
     ;                 '  halving element potential changes.')
               GOTO 10
               ENDIF
           ENDIF
       KTRH = 0
c
c    check for upgrade to mode 2 or 3
       IF (KTRM.GT.KUP)  THEN
           IF (DS.LT.DSUP)  THEN
               IF (AEZM.LT.AEZMU)  THEN
                       MODE = 3
                   ELSE
                       MODE = 2
                   ENDIF
               KTRM = 1
               KUP = 0
               ENDIF
           ENDIF
       GOTO 200
c
c **  mode 2:  Newton-Raphson iteration for LAM followed by
c               steepest descent of Y in PMOL space
c
120    IF (KTRM.GT.1)  THEN
c        previous pass was in mode 2; W should have decreased
           IF (WF.GT.WP)  THEN
c            W increased: check numerical limit
               IF (REHM.LE.ERRH)  GOTO 126
c            use last good values in mode 1
               DO 121 K=1,NB
                   RW(IoELAM+K) = RW(IoELAM+K) - RW(IoDLAM+K)
121                CONTINUE
c            set for a fresh descent
               MODE = 1
               KTRM = 0
               KUP = 2
               IF (KMON.GT.1) WRITE (KUMO,122)
122            FORMAT (' W increased: going back to last good point.')
               GOTO 10
               ENDIF
           ENDIF
c
c    check for upgrade to mode 3
126    IF (KTRM.GT.KUP)  THEN
           IF (((REHM.LT.REHMU).AND.(AEZM.LT.AEZMU)).OR.
     ;        (REHM.LT.ERRHX))  THEN
c            set to try mode 3
               MODE = 3
               KTRM = 1
               KUP = 0
               GOTO 200
               ENDIF
           ENDIF
c
c    check for change to mode 1
       IF (REHM.GT.REHMD)  THEN
c        not close enough for mode 2; set for mode 1
           MODE = 1
           KTRM = 1
           KUP = 1
           GOTO 200
           ENDIF
c
c    continue in mode 2
       GOTO 200
c
c ** mode 3:   Newton-Raphson in ELAM and PMOL
c
c    check for errors too large for mode 3
130    IF ((AEZM.GT.AEZMD).OR.(REHM.GT.REHMD))  THEN
c        stay with mode 3 if only phase mols need adjustment
           IF (REHM.LT.ERRH)  GOTO 200
c        errors large; go to mode 2
           MODE = 2
           KTRM = 0
           KTRY = 0
           KUP = 2
           GOTO 120
           ENDIF
c
c **** calculations after a good pass
c
c    save WF from this pass
200    WP = WF
c
c    HA convergence test for mode 1 or 2
250    IF (MODE.NE.3)  THEN
c      if HA error is less than adjusted error bound ERRHX,
c          go adjust PMOL
           IF (REHM.LT.ERRHX)  GOTO 600
           ENDIF
c
c    branch on mode
       GOTO (400,500,700),  MODE
c
c **** LAM adjustments in mode 1
c
c    valley check
400    IF (KTRM.GT.1)  THEN
c        compute dW/ds at the new point on the old path
           DWDSN = 0
           DO 407 K=1,NB
               DWDSN = DWDSN + RW(IoRL+K)*RW(IoHA+K)
407            CONTINUE
c        check for a valley
           IF (DWDS*DWDSN.LT.ZERO)  THEN
c            interpolate in the valley
               TERM =  - ABS(DWDSN/(DWDSN - DWDS))
               DO 409 K=1,NB
                   RW(IoELAM+K) = RW(IoELAM+K) + TERM*RW(IoDLAM+K)
409                CONTINUE
c            monitor
               IF (KMON.GT.1)  WRITE (KUMO,410)
410            FORMAT (' W valley interpolation')
c            set to take a fresh descent
               KTRM = 0
               GOTO 10
               ENDIF
           ENDIF
c
c    determine the dW/ds = beta
       BETA2 = 0.0D0
       DO 419 K=1,NB
           BETA2 = BETA2 + RW(IoHA+K)*RW(IoHA+K)
419        CONTINUE
c
c    we are at the minimum W point if BETA2 = 0
       IF (BETA2.EQ.ZERO)  GOTO 600
       DWDS = - SQRT(BETA2)
c
c    determine sum{Q(K,L)*HA(K)*HA(L)}/BETA2
       RBETA = 1.0D0/DWDS
       QHSOB2 = 0.0D0
c    phase contributions
       J2 = 0
       DO 429 M=1,NP
           J1 = J2 + 1
           J2 = J2 + IW(IoNSP+M)
c        check for active phase
           IF (IW(IoKPC+M).NE.0)  THEN
c            species contributions
               PSUM = 0.0D0
               DO 427 J=J1,J2
c               check for inclusion
                   IF (IW(IoKB+J).NE.4)  THEN
                       SUM = 0.0D0
                       DO 425 K=1,NB
                           I = IW(IoIB+K)
                           SUM  = SUM + IW(IoN+I+LoN*J)*RW(IoHA+K)
425                        CONTINUE
c                    dividing by Beta here helps prevent overflow
                       SUM = SUM*RBETA
                       PSUM = PSUM + SUM*SUM*RW(IoX+J)
                       ENDIF
427                CONTINUE
               QHSOB2 = QHSOB2 + PSUM*RW(IoPMOL+M)
               ENDIF
429        CONTINUE
c
c    estimate ds to the minimum point
       IF (QHSOB2.LE.ZERO)  THEN
c        something is wrong;  quit
           IW(IoKERR) = 1
           RETURN
           ENDIF
       DS = - FDW*DWDS/QHSOB2
c
c    limit changes
       IF (DS.GT.DLAMAX)  THEN
           DS = DLAMAX
           ENDIF
c
c    set direction cosines, changes, and new values
       FDW = 1.0D0
       DO 439 K=1,NB
           RLX = RW(IoHA+K)*RBETA
c        check for oscillation
           IF (KTRM.GT.1) THEN
               IF (RLX*RW(IoRL+K).LT.ZERO)  THEN
c                oscillating; reset damping factor
                   FDWX = ABS(1.0D0 - 0.9D0*ABS(RLX))
                   IF (FDWX.LT.FDW)  FDW = FDWX
                   ENDIF
               ENDIF
           RW(IoRL+K) = RLX
           RW(IoDLAM+K) = RLX*DS
           RW(IoELAM+K) = RW(IoELAM+K) + RW(IoDLAM+K)
439        CONTINUE
c
c    monitor
       IF (KMON.GT.1)  THEN
           WRITE (KUMO,442)
442        FORMAT (' Element potentials adjusted by',
     ;             ' steepest descent in W')
           ENDIF
c
       GOTO 10
c
c **** LAM adjustments in mode 2
c
c    calculate the LAM changes
500    DO  529 K=1,NB
c        load the rhs = - (conditioned H vector) in W
           RW(IoW+K) = - RW(IoHC+K)
c        load the conditioned Q matrix in A
           DO 527 L=1,NB
               RW(IoA+K+LoA*L) = RW(IoQC+K+LoQC*L)
527            CONTINUE
529        CONTINUE
c    singularity check/repair
       CALL SJECKA(NSMAX,NIW,NRW,CHEM,IW,RW,NB)
c    solve to get DLAM
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
c    check for trouble
       IF (IW(IoKERR).NE.0)  THEN
c        singular system; go try mode 1
           IF (KMON.GT.1)  WRITE (KUMO,530)
530        FORMAT (' Singular matrix in Newton-Raphson',
     ;             ' adjustment of element potentials')
           MODE = 1
           KUP = 2
           KTRM = 0
           GOTO 250
           ENDIF
c
c    check for excessive changes
       DO 539 K=1,NB
           IF (ABS(RW(IoW+K)).GT.DLAMAX)  THEN
c            changes too large
               IF (KMON.GT.1)  THEN
                   WRITE (KUMO,532)
532                FORMAT (' Attempted Newton-Raphson for',
     ;               ' element potentials; changes too large.'/
     ;               ' Re-estimating element potentials',
     ;               ' from rescaled mol fractions.')
                   ENDIF
c            rescale mol fractions
               DO 537 J=1,NS
                   M = IW(IoMPJ+J)
                   RW(IoX+J) = RW(IoX+J)/RW(IoZ+M)
537                CONTINUE
c            re-estimate element potentials
               L = 1
               CALL SJIESL(NSMAX,NIW,NRW,CHEM,IW,RW,L,DUMMY,IDUMMY)
               IF (IW(IoKERR).NE.0)  RETURN
c            restart in mode 1
               MODE = 1
               KTRM = 0
               KUP = 1
               GOTO 10
               ENDIF
539        CONTINUE
c
c    make the changes
       DO 549 K=1,NB
           RW(IoDLAM+K) = RW(IoW+K)
           RW(IoELAM+K) = RW(IoELAM+K) + RW(IoDLAM+K)
549        CONTINUE
c
       IF (KMON.GT.1)  WRITE (KUMO,552)
552    FORMAT (' Element potentials adjusted by Newton-Raphson')
c
c      go do another pass
       GOTO 10
c
c      Y denotes W* =  Wmin, the maximum of which we seek in phase mols
c                      space
c
c    advance pass counter
600    KTRY = KTRY + 1
c
c    ridge interpolation check
       IF (KTRY.GT.1)  THEN
c        compute dW*/ds* at the new point on the old path
           DYDSN = 0.0D0
           DO 601 M=1,NP
               IF (IW(IoKPC+M).NE.0)
     ;            DYDSN = DYDSN + (RW(IoZ+M) - 1.0D0)*RW(IoRP+M)
601            CONTINUE
c        check for a ridge
           IF (DYDSN*DYDS.LT.ZERO)  THEN
c            interpolate for the ridge
               TERM = - ABS(DYDSN/(DYDSN - DYDS))
               DO 605 M=1,NP
                   RW(IoPMOL+M) = RW(IoPMOL+M) + TERM*RW(IoDPML+M)
605                CONTINUE
               DO 607 K=1,NB
                   RW(IoELAM+K) = RW(IoELAM+K) + TERM*RW(IoDLAY+K)
607                CONTINUE
c            monitor
               IF (KMON.GT.1)  WRITE (KUMO,608)
608            FORMAT (' W* ridge interpolation')
c            set for a fresh ascent in W*
               KTRY = 0
               KTRM = 0
               GOTO 10
               ENDIF
           ENDIF
c
c    calculate the E vectors for each phase
       DO 617 M=1,NP
           DO 612 K=1,NB
c            load the rhs =  - (conditioned D) in W
               RW(IoW+K) = - RW(IoDC+K+LoDC*M)
c            load the conditioned Q matrix in A
               DO 611 L=1,NB
                   RW(IoA+K+LoA*L) = RW(IoQC+K+LoQC*L)
611                CONTINUE
612            CONTINUE
c        singularity check/repair
           CALL SJECKA(NSMAX,NIW,NRW,CHEM,IW,RW,NB)
c        solve to get E
           CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
c        check for problems
           IF (IW(IoKERR).NE.0)  THEN
c            set E zero and keep trying
               DO 613 K=1,NB
                   RW(IoW+K) = 0.0D0
613                CONTINUE
               ENDIF
c        load the E vector
           DO 615 K=1,NB
               RW(IoE+K+LoE*M) = RW(IoW+K)
615            CONTINUE
617        CONTINUE
c
c    activate all phases and get maximum phase mols
       PMMAX = 0.0D0
       DO 619 M=1,NP
           CALL SJUMAX(RW(IoPMOL+M),PMMAX)
           IW(IoKPC+M) = 1
619        CONTINUE
c
c    compute the cosine coefficients in W
620    SUM = 0.0D0
       DO 629 M=1,NP
           IoWM = IoW+M
c        check for active phase
           IF (IW(IoKPC+M).NE.0)  THEN
               RW(IoWM) = (RW(IoZ+M) - 1.0D0)
c            check for an empty phase with W < 0
               IF ((RW(IoPMOL+M).EQ.ZERO).AND.(RW(IoWM).LT.ZERO)) THEN
c                set this phase inactive and start over
                   IW(IoKPC+M) = 0
                   GOTO 620
                   ENDIF
               SUM = SUM + RW(IoWM)*RW(IoWM)
               ENDIF
629        CONTINUE
c
c    compute dY/ds on path of steepest ascent
       DYDS =  SQRT(SUM)
       IF (DYDS.EQ.ZERO)  THEN
c        check for convergence
           IF (REHCM.LT.ERRH) GOTO 800
           IW(IoKERR) = 1
           RETURN
           ENDIF
c
c    compute direction cosines and select damping factor FDYF
       RDYDS = 1.0D0/DYDS
       FDYF = 1.0D0
       DO 631 M=1,NP
           IF (IW(IoKPC+M).NE.0)  THEN
c                active phase
                   RX = RW(IoW+M)*RDYDS
c                set damping
                   IF (KTRY.GT.1)  THEN
                       IF (RX*RW(IoRP+M).LT.ZERO)  THEN
                           FDYFX = 1.0D0 - 0.8D0*ABS(RX)
                           IF (FDYFX.LT.FDYF)  FDYF = FDYFX
                           ENDIF
                       ENDIF
                   RW(IoRP+M) = RX
               ELSE
c                inactive phase
                   RW(IoRP+M) = 0.0D0
               ENDIF
631        CONTINUE
c
c  set damping coefficient FDY
       IF (FDYF.EQ.ONE)  THEN
c            reduce the damping if no longer needed
               IF (FDY.LT.ONE) THEN
                   FDY = 1.4D0*FDY
                   IF (FDY.GT.ONE)  FDY = 1.0D0
                   ENDIF
           ELSE
               IF (KTRY.EQ.1)  THEN
c                    set undamped at the start of an ascent
                       FDY = 1.0D0
                   ELSE
c                    increase damping if more is needed
                       FDY = FDY*FDYF
                   ENDIF
           ENDIF
c
c    estimate the distance to the maximum point
       D2YDS2 = 0.0D0
       DO 639 L=1,NP
c        check for active phase
           IF (IW(IoKPC+L).NE.0)  THEN
c            phase L active
               SUM = 0.0D0
               DO 637 M=1,NP
c                check for active phase
                   IF (IW(IoKPC+M).NE.0)  THEN
c                    Mth phase active:  compute A(L,M)
                       ALM = 0.0D0
                       DO 635 K=1,NB
                           ALM = ALM + RW(IoD+K+LoD*L)*RW(IoE+K+LoE*M)
635                        CONTINUE
                       SUM = SUM + ALM*RW(IoRP+M)
                       ENDIF
637                CONTINUE
               D2YDS2 = D2YDS2 + SUM*RW(IoRP+L)
               ENDIF
639        CONTINUE
c
c    check for completion
       IF (D2YDS2.GE.ZERO)  THEN
c            set a reasonable change
               DSP = PMMAX
           ELSE
c            set distance as (damped) estimate to maximum point
               DSP = - FDY*DYDS/D2YDS2
           ENDIF
       DSP0 = DSP
c
c    limit phase mols change
       TERMY = FDPY*PMMAX
       DO 641 M=1,NP
           IF (IW(IoKPC+M).NE.0)  THEN
c            active phase
               IoPMOM = IoPMOL + M
               TERM = RW(IoRP+M)*DSP
               IF ((RW(IoPMOM)+TERM).LE.RW(IoFRND)*RW(IoPMOM))  THEN
c                    phase attempting to vanish
                       IF (RW(IoPMOM).LT.FDPYV*PMMAX)  THEN
c                            size step to make PMOL vanish
                               DSP = - RW(IoPMOM)/RW(IoRP+M)
                           ELSE
c                            size step to reduce PMOL by a factor
                               DSP = - FDPYR*RW(IoPMOM)/RW(IoRP+M)
                           ENDIF
                   ELSE
c                    phase not attempting to vanish
                       IF (RW(IoPMOM).NE.ZERO)  THEN
c                            existing phase
                               TERMX = FDPMA*RW(IoPMOM)
                               IF (ABS(TERM).GT.TERMX)  THEN
                                  DSP = DSP*ABS(TERMX/TERM)
                                  ENDIF
                           ELSE
c                            non-existing phase
                               IF (ABS(TERM).GT.TERMY)  THEN
                                  DSP = DSP*ABS(TERMY/TERM)
                                  ENDIF
                           ENDIF
                   ENDIF
               ENDIF
641        CONTINUE
c
c    check for completion
       IF (DSP.EQ.ZERO)  GOTO 800
c
c    make the phase mol changes and prepare for possible rebasing
       KPMC = 0
       KRED = 0
       DO 649 M=1,NP
           IW(IoKPCX+M) = IW(IoKPC+M)
           IoPMOM = IoPMOL + M
           IoDPMM = IoDPML + M
           IF (IW(IoKPC+M).NE.0)  THEN
c               active phase
c                 set the change
                   RW(IoDPMM) = RW(IoRP+M)*DSP
c                 no termination on phase activation
                   IF (RW(IoPMOM).EQ.ZERO)  THEN
                       KPMC = 1
                       KRED = 1
                       ENDIF
c                 make the change with roundoff control
                   RW(IoPMOM) = RW(IoPMOM) + RW(IoDPMM)
                   IF (RW(IoPMOM).LT.RW(IoFRND)*ABS(RW(IoDPMM)))
     ;              THEN
                       RW(IoPMOM) = 0.0D0
                       KPMC = 1
                       KRED = 1
                       IW(IoKPCX+M) = 0
                       ENDIF
c                 relative error assessment
                   IF (IW(IoKPCX+M).NE.ZERO)  THEN
c                       be tolerant with sparse phases
                          TERM =ERRH*(10.D0*LOG(PMMAX/RW(IoPMOM))+1.0D0)
                          IF (ABS(RW(IoDPMM)).GT.TERM*RW(IoPMOM))
     ;                        KPMC = 1
                       ELSE
c                        can not terminate on a phase deactivation
                           KPMC = 1
                       ENDIF
               ELSE
c               inactive phase
                   RW(IoDPMM) = 0.0D0
               ENDIF
649        CONTINUE
c
c    check for phase activation/demise
       IF (KRED.NE.0)  THEN
c        check and rebase if ok
           CALL SJERB(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW)
           IF (IW(IoKERR).NE.0)  THEN
c            unacceptable changes; reduce and go
               TERM = 1.0D0 - FDPMC
               DO 657 M=1,NP
                   IoPMOM = IoPMOL + M
                   IoDPMM = IoDPML + M
                   RW(IoPMOM) = RW(IoPMOM) - TERM*RW(IoDPMM)
                   RW(IoDPMM) = FDPMC*RW(IoDPMM)
657                CONTINUE
               ENDIF
           ENDIF
c
c    estimate the associated changes in ELAMs
       DLX = 0.0D0
       DO 665 K = 1,NB
           IoDLYK = IoDLAY + K
           RW(IoDLYK) = 0
           DO 663 M=1,NP
               IF (IW(IoKPC+M).EQ.0)  GOTO 663
               RW(IoDLYK) = RW(IoDLYK) + RW(IoE+K+LoE*M)*RW(IoDPML+M)
663            CONTINUE
           CALL SJUMAX(RW(IoDLYK),DLX)
665        CONTINUE
c
c    make the ELAM changes if not excessive
       IF (DLX.LT.DLAMAX)  THEN
               DO 667 K=1,NB
                   RW(IoELAM+K) = RW(IoELAM+K) + RW(IoDLAY+K)
667                CONTINUE
           ELSE
               DO 669 K=1,NB
                   RW(IoDLAY+K) = 0.0D0
669                CONTINUE
           ENDIF
c
c    reset active phase indicator
       DO 679 M=1,NP
           IW(IoKPC+M) = IW(IoKPCX+M)
679        CONTINUE
c
c    monitor
       IF (KMON.GT.1)  THEN
           WRITE (KUMO,680)
680        FORMAT (' phase mols adjusted by steepest ascent in W*')
           ENDIF
c
c    convergence check
       IF ((REHCM.LT.ERRH).AND.(AEZM.LT.ERRZ))  THEN
           IF ((DLX.LT.ERRH).AND.(KPMC.EQ.0))  GOTO 800
           ENDIF
c
c    start a new W descent
       KTRM = 0
       KUP = 0
       GOTO 10
c
c **** ELAM and PMOL adjustments in mode 3
c
c    if constraints accurately satisfied but phasemols not, go adjust
c       in Mode 2
700    IF (REHCM.LT.RW(IoFRND))  THEN
           MODE = 2
           KUP = 0
           KTRY = 0
           GOTO 600
           ENDIF
c
c    set up the active phase list and get maximum phase mols
       NPA = 0
       PMMAX = 0.0D0
       DO 701 M=1,NP
           IF ((RW(IoPMOL+M).GT.ZERO).OR.(RW(IoZ+M).GE.ONE))  THEN
c                set phase active
                   IW(IoKPC+M) = 1
                   NPA = NPA + 1
                   IW(IoMPA+NPA) = M
                   CALL SJUMAX(RW(IoPMOL+M),PMMAX)
               ELSE
c                set phase inactive
                   IW(IoKPC+M) = 0
               ENDIF
701        CONTINUE
c
c    set matrix size for solution for ELAM and PMOL
       MS = NB + NPA
c    load the matrix
       DO 707 K=1,NB
c        load the rhs for rows 1-NB
           RW(IoW+K) = - RW(IoHC+K)
c        load the last NPA columns and rows
           DO 703 M1=1,NPA
               L = NB + M1
               M = IW(IoMPA+M1)
               RW(IoA+K+LoA*L) = RW(IoDC+K+LoDC*M)
               RW(IoA+L+LoA*K) = RW(IoD+K+LoD*M)
703            CONTINUE
c        load the conditioned Q matrix in the first NB columns
c        (and rows)
           DO 705 L=1,NB
               RW(IoA+K+LoA*L) = RW(IoQC+K+LoQC*L)
705            CONTINUE
707        CONTINUE
c    load the last NPA x NPA block and rhs
       DO  709 M1=1,NPA
           L = NB + M1
           M = IW(IoMPA+M1)
           RW(IoW+L) = 1.0D0 - RW(IoZ+M)
           DO 708 M2=1,NPA
               K = NB + M2
               RW(IoA+K+LoA*L) = 0.0D0
708            CONTINUE
709        CONTINUE
c
c    singularity check/repair
       CALL SJECKA(NSMAX,NIW,NRW,CHEM,IW,RW,MS)
c
c    solve to get changes in ELAMs and PMOL
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),MS,IW(IoKERR))
c    check for problems
       IF (IW(IoKERR).NE.0)  THEN
c        singular matrix; try mode 2
           IF (KMON.GT.1)  WRITE (KUMO,710)
710        FORMAT (' Singular matrix in full Newton-Raphson')
           MODE = 2
           KUP = 2
           KTRM = 0
           GOTO 250
           ENDIF
c
c -- check for excessive change
       DLX = 0.0D0
c    check maximum LAM changes
       DO 713 K=1,NB
           IF (ABS(RW(IoW+K)).GT.DLAMAX)  GOTO 750
           CALL SJUMAX(RW(IoW+K),DLX)
713        CONTINUE
c
c    check phase changes
       DO 733 M1=1,NPA
           K = NB + M1
           M = IW(IoMPA+M1)
           IoWK = IoW + K
           IF (RW(IoPMOL+M).EQ.ZERO)  THEN
c                phase does not exist; permit no decrease
                   IF (RW(IoWK).LT.ZERO)  GOTO 750
c                allow a reasonable increase
                   TEST = FDPMI3*PMMAX
               ELSE
c                phase exists
                   IF (RW(IoWK).GT.ZERO) THEN
c                        allow a large increase
                           TEST = FDPMI3*PMMAX
                       ELSE
c                        allow a reasonable decrease
                           TEST = FDPMD3*RW(IoPMOL+M)
                       ENDIF
               ENDIF
           IF (ABS(RW(IoWK)).GE.TEST)   GOTO 750
733        CONTINUE
c
c -- make the changes and check for phase convergence
       DPMX = 0
       DO 741 K=1,NB
           RW(IoDLAY+K) = RW(IoW+K)
           RW(IoELAM+K) = RW(IoELAM+K) + RW(IoDLAY+K)
741        CONTINUE
       K = NB
       KPMC = 0
       DO 747 M=1,NP
           IoPMOM = IoPMOL + M
           IoDPMM = IoDPML + M
           IF (IW(IoKPC+M).NE.0)  THEN
c                active phase
                   K = K + 1
                   RW(IoDPMM) = RW(IoW+K)
c                prevent termination on phase activation
                   IF (RW(IoPMOM).EQ.ZERO)  KPMC = 1
c                make the change
                   RW(IoPMOM) = RW(IoPMOM) + RW(IoDPMM)
c                relative error assessment; tolerant with sparse phases
                   IF (RW(IoPMOM).NE.ZERO)  THEN
                          TERM =ERRH*(10.D0*LOG(PMMAX/RW(IoPMOM))+1.0D0)
                          IF (ABS(RW(IoDPMM)).GT.TERM*RW(IoPMOM))
     ;                         KPMC = 1
                       ELSE
c                        hopefully never get here
                           KPMC = 1
                       ENDIF
               ELSE
c                inactive phase
                   RW(IoDPMM) = 0.0D0
               ENDIF
747        CONTINUE
c
       IF (KMON.GT.1)  WRITE (KUMO,748)
748    FORMAT (' Element potentials and phase mols adjusted by',
     ;         ' Newton-Raphson')
c
c *** convergence check ***
c    must have small errors
       IF ((REHCM.LT.ERRH).AND.(AEZM.LT.ERRZ))  THEN
c        and be making small changes
           IF ((DLX.LT.ERRH).AND.(KPMC.EQ.0))  GOTO 800
           ENDIF
c
c    continue the iteration as if a fresh W* ascent
       KTRY = 0
       GOTO 10
c
c -- changes too large; reset for mode 2 and a fresh W* ascent
750    MODE = 2
       KTRM = 0
       KTRY = 0
       KUP = 2
       IF (KMON.GT.1)  THEN
           WRITE (KUMO,752)
752        FORMAT (' Attempted full Newton-Raphson; changes excessive')
           ENDIF
       GOTO 250
c
c ---- exit calculations
c
c    normal return
800    IW(IoKERR) = 0
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJECKA(NSMAX,NIW,NRW,CHEM,IW,RW,MS)
c
c      Checks the first NB rows of matrix A of dimension MS used in
c      SJEQLB to see if it is singular (because of an absent base).
c      Replaces bad rows with the condition that the base mol fraction
c      not change.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c          IW(I)       integer work array
c          RW(I)       REAL work array
c @        CHEM(J)     CHARACTER*16 name of Jth species
c @        MS          matrix size (row count)
c
c      Variables in the integer work array IW:
c @        IB(K) = I   if the Kth independent atom is the Ith system
c                      atom
c @        JB(K) = J   if the Kth base is the Jth species
c @        N(I,J)      number of Ith atoms in Jth molecule
c @        NB          number of bases
c
c      Variables in the real work array RW:
c @#       A(K,L)      work matrix
c @#       W(K)        work vector
c
c      Variables used only internally:
c          KCHK        check control
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16     CHEM
c-----------------------------------------------------------------------
       DIMENSION   CHEM(NSMAX),IW(NIW),RW(NRW),IEPTR(80)
c-----------------------------------------------------------------------
       COMMON /SJEPTR/ IEPTR
       EQUIVALENCE (IoIB,IEPTR(9)),(IoJB,IEPTR(11)),
     ;   (IoNB,IEPTR(6)),(IoN,IEPTR(28)), (LoN,IEPTR(29)),
     ;   (IoA,IEPTR(36)),(LoA,IEPTR(37)),(IoW,IEPTR(76))
c-----------------------------------------------------------------------
c    set comparison constant
       ZERO = 0.0D0
c
c    get parameters
       NB = IW(IoNB)
c
c    check the matrix
       DO 99 K=1,NB
           KCHK = 0
           DO 91 L=1,MS
               IF (RW(IoA+K+LoA*L).NE.ZERO)  KCHK = 1
91             CONTINUE
           IF (KCHK.EQ.0)  THEN
c            replace the row to prevent singularities
c            this treatment holds X(JB(K)) fixed in ELAM adjustments
               J = IW(IoJB+K)
               RW(IoW+K) = 0.0D0
               DO 95 L=1,NB
                   I = IW(IoIB+L)
                   RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J)
95                 CONTINUE
               IF (MS.GT.NB)  THEN
                   L1 = NB + 1
                   DO 97 L=L1,MS
                       RW(IoA+K+LoA*L) = 0.0D0
97                     CONTINUE
                   ENDIF
               ENDIF
99         CONTINUE
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJINIT(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW)
c
c      Initialization routine
c-----------------------------------------------------------------------
c      Identifies the linearly independent atoms in the allowed species
c      set and distributes the atom populations to a set of system
c      base species, which are expected to be the dominant species.
c      This is accomplished by an approximate Gibbs function
c      minimization in which the ln(X) term is neglected, giving rise
c      to a linear minimization problem solved by the standard simplex
c      process (SJISMP).
c      There will be one base species for each linearly independent
c      atom.  A specification of impossible populations is identified
c      by SJISMP, in which case the initialization is terminated.
c
c      The simplex multipliers emerging at the end of this process are
c      element potentials corresponding to the prescribed mol fractions
c      of the base species species. Using these potentials, the mol
c      fractions estimated for all other species are less than unity.
c      Thus, they form a first guess for the element potentials.
c
c      The main analysis loop begins with a set of trial mols from
c      which the mol fractions and phase mols are calculated (SJIPMX).
c
c      The element potentials are then estimated (SJIESL) with
c      iteration until a consistent set has been found for the given
c      basis set.
c
c      The conditioned population equations are examined in SJIESL to
c      see if any base is dominated by another species not a balancer.
c      If there is dominance, then a base change is made (SJISMP) and
c      the element potentials are reestimated using SJIESL.
c
c      Once a consistent set of basis species and potentials have been
c      found, the mol fractions of balancing species are estimated, and
c      atoms are redistributed to seek these target mol fractions
c      (SJISRD).
c
c      These processes are repeated until sufficient consistency is
c      obtained.
c-----------------------------------------------------------------------
c      On return:
c
c          Successful initialization:
c
c              KERR = 0
c
c          Problems encountered:
c
c              KERR = 1 program problems (singularities)
c                     2 unable to initialize
c                     3 impossible populations
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c
c      Variables in the integer work array IW:
c #        IB(K) = I   if the Ith atom is the Kth independent system
c                      atom
c #        JB(K) = J   if the Jth system species is the Kth base
c                      species
c -        JBA(K) = J  JB(K) after last redistribution
c -        JBB(K) = J  JB(K) for the best basis
c -        JBO(K) = J  JB(K) before redistribution
c #        KB(J)       basis control (SJISMP)
c @                 -1 if the Jth species is a base freed for SJISRD
c                      call
c                    0 if the Jth species is not a base
c                    1 if the Jth species is a base
c                    2 if the Jth species is a balancing species
c                    4 if the Jth species is excluded
c          KBA(J)      KB(J) after last redistribution
c -        KBB(J)      KB(J) for the best basis
c -        KBO(J)      KB(J) before the last distribution
c #        KERR        error flag
c @        KMON        monitor control
c @        KUMO        output unit for monitor
c @        N(I,J)      number of Ith atoms in Jth system species
c @        NA          number of atom types
c #        NB          number of base species (independent atoms)
c @        NP          number of phases
c @        NS          number of species
c @        NSP(M)      number of species in the Mth phase
c
c      Variables in the real work array RW:
c -        ELMB(K)     ELAM(K) for best basis
c #        ELAM(K)     element potential for the Kth independent system
c                         atom
c @        G(J)        g(T,P)/RT  for the Jth species
c @        HUGE        large machine number
c @        PA(I)       population of the Ith atoms
c #        PC(K)       Kth conditioned population
c #        PMOL(M)     mols of Mth phase
c -        SMLA(J)     SMOLS(J) after last redistribution
c -        SMLB(J)     SMOLS(J) for best basis
c -        SMLO(J)     SMOLS(J) for before redistribution
c #        SMOL(J)     mols of Jth species
c #        X(J)        mols fraction of the Jth species (in its phase)
c -        XO(J)       X(J) of last distribution
c
c      Variables used only internally:
c          DOM         maximum dominance of a base by a secondary
c                      species
c          DOMB        DOM for best basis
c          KCCPC       conditioning matrix calculation control
c          KEX30       exit control
c          KRED        redistribution control (0=don't, 1=maybe)
c          KTRB        basis change iteration counter
c          KTRR        redistribution counter
c          XRMIN       fractional change in X for initiation
c                      convergence
c          ZERO        0 in calculation precision
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
       LOGICAL*2       SJICKR
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C      LOGICAL SJICKR
C*****END precision > single
C
       CHARACTER*16   ATOM, ELECT, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
c-----------------------------------------------------------------------
c    Electron designation
       DATA ELECT /'E-'/
c-----------------------------------------------------------------------
c    set comparison constants in calculation precision
       ZERO = 0.0D0
       ONE = 1.0D0
c
c    get parameters
       NA = IW(IoNA)
       NP = IW(IoNP)
       NS = IW(IoNS)
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
c
c    monitor
       IF (KMON.GT.1)  WRITE (KUMO,1)
1      FORMAT (/' Initialization:')
c
c    clear X and allow all species
       DO 3 J=1,NS
           RW(IoX+J) = 0.D0
           IW(IoKB+J) = 0
3          CONTINUE
c
c    check for zero atom populations and exclude such species
       DO 9 I=1,NA
           IF (RW(IoPA+I).EQ.ZERO)  THEN
               IF (ATOM(I).NE.ELECT)  THEN
                   IF (KMON.GT.1)  WRITE (KUMO,4) ATOM(I)
4                  FORMAT (/' Eliminating species containing ',
     ;                       'absent ',A,' atoms:')
                   DO 7 J=1,NS
                       IF (IW(IoN+I+LoN*J).NE.0)  THEN
                           IF (KMON.GT.1)  WRITE (KUMO,6) CHEM(J)
6                          FORMAT (5X,A)
                           IW(IoKB+J) = 4
                           ENDIF
7                      CONTINUE
                   ENDIF
               ENDIF
9          CONTINUE
c
c ** simplex initialization
c
c    simplex distribution
10     L = 0
       CALL SJISMP(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW,L)
       IF (IW(IoKERR).NE.0)  RETURN
c    check for no basis for anything
       NB = IW(IoNB)
       IF (NB.EQ.0)  THEN
           IW(IoKERR) = 3
           RETURN
           ENDIF
c
c    initialize counters and controls
       KTRR = 0
       KEX30 = 1
       KCCPC = 0
c
c ** loop point after redistribution
c
c    set rebasing counter
20     KTRB = 0
       DOMB = RW(IoHUGE)
c
c ** loop point after rebasing
c
30     KTRB = KTRB + 1
c
c    calculate the conditioning matrix and conditioned populations
       CALL SJICPC(NIW,NRW,IW,RW,KCCPC)
       IF (IW(IoKERR).NE.0)  RETURN
       KCCPC = 1
c
c    calculate phase mols and mol fractions
       CALL SJIPMX(NSMAX,NIW,NRW,CHEM,IW,RW)
c
c    select exit
       GOTO (32,60,90), KEX30
c
c    estimate ELAMs and dominance
32     L = 0
       CALL SJIESL(NSMAX,NIW,NRW,CHEM,IW,RW,L,DOM,JW)
c
c    check for species exclusion
       IF (IW(IoKERR).EQ.4)  GOTO 10
c    check for ESTEP error
       IF (IW(IoKERR).NE.0)  RETURN
c
c    check for needed rebasing
       IF (JW.NE.0)  THEN
c        check vs best basis
           IF (DOM.LT.DOMB)  THEN
c            save base as best
               DO 33 J=1,NS
                   IW(IoKBB+J) = IW(IoKB+J)
                   RW(IoSMOB+J) = RW(IoSMOL+J)
33                 CONTINUE
               DO 35 K=1,NB
                   IW(IoJBB+K) = IW(IoJB+K)
                   RW(IoELMB+K) = RW(IoELAM+K)
35                 CONTINUE
               DOMB = DOM
               ENDIF
c        check rebasing count
           IF (KTRB.GT.NB)  THEN
c            excessive rebasing attempts; use best
               GOTO 50
               ENDIF
c        install JW as a base
           L = JW + 10
           CALL SJISMP(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW,L)
           IF (IW(IoKERR).NE.0)  RETURN
c        check for repeat of the best basis
           IF (KTRB.GT.1)  THEN
               DO 37 J=1,NS
                   L = IW(IoKBB+J)
                   IF (L.LT.0) L = - L
                   IF ((IW(IoKB+J).EQ.1).AND.(L.NE.1))  GOTO 40
37                 CONTINUE
c            rebasing cycling
               GOTO 50
               ENDIF
c
c       try the new basis
40         KEX30 = 1
           GOTO 30
c
c       reinstall the best basis
50         DO 51 J=1,NS
               IW(IoKB+J) = IW(IoKBB+J)
               RW(IoSMOL+J) = RW(IoSMOB+J)
51             CONTINUE
           DO 53 K=1,NB
               RW(IoELAM+K) = RW(IoELMB+K)
               IW(IoJB+K) = IW(IoJBB+K)
53             CONTINUE
           DOM = DOMB
           KEX30 = 2
           GOTO 30
c
           ENDIF
c
c ** redistribution check
c
c    set controls
60     KRED = 0
       KEX30 = 1
c
c    set targets for balancers
       DO 65 J=1,NS
           IF (IW(IoKB+J).EQ.2)  THEN
c        set redistribution to balancer
c            flag possible redistribution
               IF (KRED.EQ.0)  KRED = 1
c            estimate mol fraction
               SUM = - RW(IoG+J)
               DO 63 K=1,NB
                   I = IW(IoIB+K)
                   SUM = SUM + RW(IoELAM+K)*IW(IoN+I+LoN*J)
63               CONTINUE
                RW(IoX+J) = SJUEXP(SUM)
c            check for valid value
               IF (RW(IoX+J).GE.ONE)  THEN
c                abort redistribution
                   IF (KMON.GT.1)  WRITE (KUMO,64) CHEM(J)
64                 FORMAT (/' Redistribution to x>1 for ',A,
     ;                      ' not allowed; attempting solution.')
                   GOTO 90
                   ENDIF
               ENDIF
65         CONTINUE
c
c    check for no targets
       IF (KRED.EQ.0)  THEN
           IF (KMON.GT.1)  WRITE (KUMO,66)
66         FORMAT (/' Initializer distribution converged.')
           GOTO 90
           ENDIF
c
c    check for repeating redistribution
       IF (KTRR.GT.0)  THEN
c        check repeat of the last distribution
       IF (SJICKR(NIW,NRW,IW,RW))  THEN
c            repeating
               IF (KMON.GT.1)  WRITE (KUMO,66)
c            check for rebasing since the last redistribution
               IF (KTRB.NE.1)  THEN
c                rebased on this pass; restore last redistribution
                   DO 67 J=1,NS
                       IW(IoKB+J) = IW(IoKBA+J)
                       RW(IoSMOL+J) = RW(IoSMOA+J)
67                     CONTINUE
                   DO 69 K=1,NB
                       IW(IoJB+K) = IW(IoJBA+K)
                       RW(IoELAM+K) = RW(IoELMA+K)
69                     CONTINUE
c                go restore phase mols, conditioning
                   KEX30 = 3
                   GOTO 30
                   ENDIF
c            go exit
               GOTO 90
               ENDIF
           ENDIF
c
c ** redistribution
c
c    save distribution
70     DO 71 J=1,NS
           RW(IoXO+J) = RW(IoX+J)
           IW(IoKBO+J) = IW(IoKB+J)
           RW(IoSMOO+J) = RW(IoSMOL+J)
71         CONTINUE
       DO 73 K=1,NB
           IW(IoJBO+K) = IW(IoJB+K)
73         CONTINUE
c
c    monitor
       IF (KMON.GT.1)  THEN
           DO 75 J=1,NS
               IF (IW(IoKB+J).EQ.2) WRITE (KUMO,74) RW(IoX+J),CHEM(J)
74                 FORMAT (/' Redistributing atoms seeking',
     ;                      ' mol fraction =',E16.8,' for ',A)
75             CONTINUE
           ENDIF
c
c    redistribute
       CALL SJISRD(NAMAX,NIW,NRW,IW,RW)
c    check for failed redistribution
       IF (IW(IoKERR).NE.0) THEN
c        attempt solution with distribution before attempted
c        redistribution
           DO 81 J=1,NS
               RW(IoSMOL+J) = RW(IoSMOO+J)
               IW(IoKB+J) = IW(IoKBO+J)
81             CONTINUE
           DO 83 K=1,NB
               IW(IoJB+K) = IW(IoJBO+K)
83             CONTINUE
           IF (KMON.GT.1)  WRITE (KUMO,84)
84         FORMAT (/' Unable to redistribute; attempting solution.')
           GOTO 90
           ENDIF
c
c    check count
       KTRR = KTRR + 1
       IF (KTRR.GT.NB*NP)  THEN
           IF (KMON.GT.1)  WRITE (KUMO,86)
86         FORMAT (/' Excessive iterations in initializer;',
     ;              ' attempting solution.')
           GOTO 90
           ENDIF
c
c    save redistribution
       DO 87 K=1,NB
           IW(IoJBA+K) = IW(IoJB+K)
           RW(IoELMA+K) = RW(IoELAM+K)
87         CONTINUE
       DO 89 J=1,NS
           IW(IoKBA+J) = IW(IoKB+J)
           RW(IoSMOA+J) = RW(IoSMOL+J)
89         CONTINUE
c
c    go try new distribution
       GOTO 20
c
c ** exit
c
c    clear flags
90     DO 91 J=1,NS
           IoKBJ = IoKB+J
           IF (IW(IoKBJ).LT.0)  IW(IoKBJ) = - IW(IoKBJ)
           IF (IW(IoKBJ).EQ.2)  IW(IoKBJ) = 0
91         CONTINUE
c
c    monitor
       If (KMON.GT.1)  THEN
           WRITE (KUMO,92)
92         FORMAT (/' Independent species used as basis set:')
           DO 95 K=1,NB
               J = IW(IoJB+K)
               WRITE (KUMO,94) CHEM(J)
94             FORMAT (5X,A)
95             CONTINUE
           ENDIF
c
c    normal return
       IW(IoKERR) = 0
       RETURN
c
       END
c-----------------------------------------------------------------------
c
C*****precision > double
       LOGICAL*2 FUNCTION SJICKR(NIW,NRW,IW,RW)
C*****END precision > double
C*****precision > single
C       LOGICAL FUNCTION SJICKR(NIW,NRW,IW,RW)
C*****END precision > single
c
c      Tests to see if species, bases, and targets are the same as
c      for previous case KBO,XO to within XRMIN < X/XO < 1/XRMIN
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c          IW(I)       integer work array
c          RW(I)       REAL work array
c
c      Variables in the integer work array IW:
c @        JB(K) = J   if the Jth species is the Kth base species
c @        KB(J)       basis control
c                   -1 if the Jth species is a base freed for SJISRD
c                      call
c                    0 if the Jth species is not a base
c                    1 if the Jth species is a base
c                    2 if the Jth species is a balancing species
c                    4 if the Jth species is excluded
c @        KBO(J)      old KB(J)
c @        KMON        monitor control
c @        KUMO        output unit for monitor
c @        NS          number of species
c
c      Variables in the real work array RW:
c @        X(J)        mol fraction of Jth species (here targets)
c @        XO(J)       old X(J)
c-----------------------------------------------------------------------
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
c-----------------------------------------------------------------------
       DIMENSION   IW(NIW),RW(NRW),IEPTR(80)
c-----------------------------------------------------------------------
c   pointers
       COMMON /SJEPTR/ IEPTR
       EQUIVALENCE (IoJB,IEPTR(11)),(IoKB,IEPTR(18)),(IoKBO,IEPTR(22)),
     ;   (IoKMON,IEPTR(2)),(IoKUMO,IEPTR(4)),(IoNS,IEPTR(8)),
     ;   (IoX,IEPTR(77)),(IoXO,IEPTR(78))
c-----------------------------------------------------------------------
c    get parameters
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
       NS = IW(IoNS)
c
c    set range constants in calculation precision
       XRMIN = 0.8D0
       XRMAX = 1.2D0
       ZERO = 0.0D0
c
c    compare
       DO 9 J=1,NS
c       is species included this time (KB = -1,1,2 ne 0,4)?
           L = IW(IoKB+J)
           M = IW(IoKBO+J)
           IF ((L.NE.0).AND.(L.NE.4))  THEN
c                included; must repeat if omitted before (KBO=0,4)
                   IF ((M.EQ.0).OR.(M.EQ.4))  GOTO 70
               ELSE
c        not included; must repeat if included before (KBO=-1,1,2)
                   IF ((M.NE.0).AND.(M.NE.4))  GOTO 70
               ENDIF
c        check for target (balancing species)
           IF (L.EQ.2)  THEN
               IF (RW(IoX+J).EQ.ZERO)  GOTO 70
c            check target for essential repeat
               TERM = ABS(RW(IoXO+J)/RW(IoX+J))
               IF ((TERM.LT.XRMIN).OR.(TERM.GT.XRMAX)) GOTO 70
               ENDIF
9          CONTINUE
c    same
       SJICKR = .TRUE.
       RETURN
c
c    not same
70     SJICKR = .FALSE.
       RETURN
c
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJIPMX(NSMAX,NIW,NRW,CHEM,IW,RW)
c
c      Calculates phase mols and mol fractions from a set of mols.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c
c      Variables in the integer work array IW:
c @        KMON        monitor control
c @        KUMO        output unit for monitor
c @        NP          number of phases
c @        NS          number of species
c @        NSP(M)      number of species in Mth phase
c
c      Variables in the real work array RW:
c #        PMOL(M)     total mols in the Mth phase
c @        SMOL(J)     number of mols of the Jth species
c #        X(J)        mol fraction of the Jth species (in its phase)
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16 CHEM
c-----------------------------------------------------------------------
       DIMENSION   CHEM(NSMAX),IW(NIW),RW(NRW),IEPTR(80)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/IEPTR
       EQUIVALENCE (IoKMON,IEPTR(2)),(IoKUMO,IEPTR(4)),(IoNP,IEPTR(7)),
     ;   (IoNS,IEPTR(8)),(IoNSP,IEPTR(30)),
     ;   (IoPMOL,IEPTR(62)),(IoSMOL,IEPTR(74)),(IoX,IEPTR(77))
c-----------------------------------------------------------------------
c    comparison constant in machine precision
C*****precision > single
C      ZERO = 0.0E0
C*****END precision > single
C*****precision > double
      ZERO = 0.0D0
C*****END precision > double
c
c    get parameters
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
       NP = IW(IoNP)
       NS = IW(IoNS)
c
c    analyze all phases
       J2=0
       DO 15 M=1,NP
c        analyze phase M
           J1 = J2 + 1
           J2 = J2 + IW(IoNSP+M)
c        compute total mols
           IoPMOM = IoPMOL + M
           RW(IoPMOM) = 0.0D0
           DO 11 J = J1,J2
               RW(IoPMOM) = RW(IoPMOM) + RW(IoSMOL+J)
11             CONTINUE
c        compute mol fractions
           IF (RW(IoPMOM).NE.ZERO)  THEN
                   DO 13 J=J1,J2
                       RW(IoX+J) = RW(IoSMOL+J)/RW(IoPMOM)
13                     CONTINUE
               ELSE
                   DO 14 J=J1,J2
                       RW(IoX+J) = 0.0D0
14                     CONTINUE
               ENDIF
15         CONTINUE
c
c    monitor
       IF (KMON.GT.1)  THEN
           WRITE (KUMO,20)
20         FORMAT (/' Estimated distribution:')
           WRITE (KUMO,21) (M,RW(IoPMOL+M),M=1,NP)
21         FORMAT ('   Phase ',I2,' mols =',1PE12.5)
           DO 29 J1=1,NS,6
               J2 = J1 + 5
               IF (J2.GT.NS)  J2 = NS
               WRITE (KUMO,22) (CHEM(J),J=J1,J2)
22             FORMAT  (6X,6(4X,A8))
               WRITE (KUMO,23) (RW(IoSMOL+J),J=J1,J2)
23             FORMAT  ('   mols:',6(1PE12.5))
               WRITE (KUMO,24) (RW(IoX+J),J=J1,J2)
24             FORMAT  ('      X:',6E12.5)
29             CONTINUE
           ENDIF
c
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJICPC(NIW,NRW,IW,RW,KOP)
c
c      Computes the conditioning matrix and conditioned populations
c-----------------------------------------------------------------------
c      The conditioning matrix satisfies
c
c          sum{CM(M,K)*N(I(K),J(L))} = delta(M,L)
c            independent atoms K
c-----------------------------------------------------------------------
c
c      For KOP = 0  makes the calculation and saves JB(K) in JBX(K)
c
c      For KOP > 0  only makes calculation if JB() ne JBX()
c
c      On return:
c
c          Successful:
c              CM and PC loaded
c              KERR = 0
c
c          Problems:
c              KERR = 1
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c          IW(I)       integer work array
c          RW(I)       REAL work array
c @        KOP         option control (see above)
c
c      Variables in the integer work array IW:
c @        IB(K) = I   if the Ith atom is the Kth independent atom
c @        JB(K) = J   if the Jth species is the Kth base species
c @#       JBX(K)      JB of the previous call
c #        KERR        error flag
c @        N(I,J)      number if Ith atoms in the Jth species
c @        NB          number of independent atoms
c
c      Variables in the real work array RW:
c -        A(I,J)      work array
c #        CM(I,J)     conditioning matrix
c @        PA(I)       population of I atoms
c #        PC(K)       Kth conditioned population
c -        W(I)        work array
c
c      Variables used only internally:
c @#       NBX         NB at last call
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   IW(NIW),RW(NRW),IEPTR(80)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/ IEPTR
       EQUIVALENCE (IoIB,IEPTR(9)),(IoJB,IEPTR(11)),(IoJBX,IEPTR(16)),
     ;  (IoKERR,IEPTR(1)),
     ;  (IoN,IEPTR(28)),(LoN,IEPTR(29)),(IoNB,IEPTR(6)),
     ;  (IoA,IEPTR(36)),(LoA,IEPTR(37)),
     ;  (IoCM,IEPTR(41)),(LoCM,IEPTR(42)),
     ;  (IoPA,IEPTR(60)),(IoPC,IEPTR(61)),(IoW,IEPTR(76))
c-----------------------------------------------------------------------
c    get parameters
       NB = IW(IoNB)
c
c    check option
       IF (KOP.GT.0)  THEN
           DO 5 K=1,NB
               IF (IW(IoJB+K).NE.IW(IoJBX+K))  GOTO 10
5              CONTINUE
c        redundant computation
           IW(IoKERR) = 0
           RETURN
           ENDIF
c
c -- Calculate C
c
10      NBX = NB
        DO 59 M=1,NB
c        set up the matrix and rhs
           DO 39 L=1,NB
c            identify the base species
               J = IW(IoJB+L)
c            save base species
               IW(IoJBX+L) = J
c            right hand side
               IF (L.EQ.M)  THEN
                       RW(IoW+L) = 1.0D0
                   ELSE
                       RW(IoW+L) = 0.0D0
                   ENDIF
c            matrix
               DO 29 K=1,NB
c                identify the atom
                   I = IW(IoIB+K)
c                set the coefficient
                   RW(IoA+L+LoA*K) = IW(IoN+I+LoN*J)
29                 CONTINUE
39             CONTINUE
c
c        solve for CM(M,K) for K=1,NB
           CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
           IF (IW(IoKERR).NE.0)  RETURN
c
c        set the Mth row of CM
           DO 49 K=1,NB
               RW(IoCM+M+LoCM*K) = RW(IoW+K)
49             CONTINUE
59         CONTINUE
c
c    compute conditioned populations
       DO 69 K=1,NB
           IoPCK = IoPC + K
           RW(IoPCK) = 0.0D0
           TERMX = 0.0D0
           DO 67 L=1,NB
               IL = IW(IoIB+L)
               TERM = RW(IoCM+K+LoCM*L)*RW(IoPA+IL)
               CALL SJUMAX(TERM,TERMX)
               RW(IoPCK) = RW(IoPCK) + TERM
67             CONTINUE
           CALL SJURND(RW(IoPCK),TERMX)
69         CONTINUE
c
c -- Normal return
c
       IW(IoKERR) = 0
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE  SJISMP(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW,KOP)
c
c      Simplex initializer;  also used in rebasing
c-----------------------------------------------------------------------
c      Estimates the initial mol distribution by minimizing
c
c              sum{F(J)*SMOL(J)}
c                real species
c
c      subject to the constraints
c
c              sum{N(I,J)*SMOL(J)} = PA(I)   for all independent atoms
c                 real species
c
c          SMOL(J)  ge 0    all species
c
c      For KOP = 0:    (first initializer call)
c
c          start with false species distribution
c          F(J) = G(J)
c          SLAM(K) returned as simplex multiplier SMUL(K)
c            (element potential assuming X(J) = 1 for each base)
c
c      For KOP = 1:    (rebasing during phase adjustment in SJEQEP)
c
c          start with false species distribution
c          F(J) = G(J) - ln PMOL(phase of J)
c            (this gives better basing with large phase mols
c            differences)
c
c      For KOP = 10 + JN:   (rebasing during initialization)
c
c          start with current bases and mols
c          install JN as a base and then exit
c
c      Works only with the linearly independent atoms,  1,...,K,...,NB.
c
c      At call:
c
c              NB = number of atom types
c              IB(I) = I (all atoms assumed independent)
c              KB(J) = 0 real species not excluded
c                      4 excluded species
c              PA(I) = system population of the Ith atom
c
c      On return:
c
c              KERR = 1 if the problem is singular
c              KERR = 3 if the populations are impossible
c
c                      or
c
c              KERR = 0 (populations possible)
c              NB = number of independent atoms
c              IB(K) = Kth independent atom  1 <= k <= nb
c                      dependent atoms for K > NB
c              JB(K) = Kth system base
c              KB(J) = 1 for base species
c              SMOL(J) set for the extremizing distribution
c
c      Two round approach:
c
c          The first round distributes atoms from false species to the
c          real species by minimizing the moles of false species
c          subject to the constraints:
c
c              Minimize   sum{abs(SMOL(J)} (constraints as above)
c                          false species
c
c          False species J = NS + K is a monatomic species consisting
c          of one IB(K) atom.
c
c          If false species with SMOL ne 0 remain at the end of
c          round 1, the input populations were impossible
c          (error return).
c
c          If a false species J remains in the basis set, but with
c          SMOL(J) =0, then the corresponding atom is not independent.
c          NB and IB(K) are adjusted accordingly before round 2.
c
c          The second round performs the approximate Gibbs
c          minimization.
c
c      Special consideration for electrons:
c
c          The electron population will be negative if the system
c          carries a positive charge (a negative amount of electrons).
c          Therefore, it may be necessary to work with negative SMOL
c          of false species in round 1.  The program is designed with
c          this in mind.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c
c      Variables in the integer work array IW:
c #        IB(K) = I   if the Ith atom is the Kth independent atom
c -        IBO(K)      saved IB
c #        JB(K) = J   if the Jth species is the Kth base species
c #        KB(J) = 0   species J is not a base
c                  1   species J is a base
c                  2   species J may not be used as a base
c                  4   species J is excluded
c #        KERR        error flag (see above)
c @        KOP         run control (see above)
c @        KMON        monitor control
c @        KUMO        output unit for monitor
c -        LB2(K)      temporary storage
c @        MPJ(J)      phase of species J
c @        N(I,J)      number of Ith atoms in a species-J molecule
c @        NA          number of atom types in the system
c #        NB          number of bases (independent atoms)
c @        NP          number of phases allowed
c @        NS          total number of species
c
c      Variables in the real work array RW:
c -        A(I,J)      work matrix
c #        ELAM(K)     element potential for Kth independent atom
c -        F(J)        minimizing function; modified G(J)
c @        FRND        roundoff number
c @        G(J)        g(T,P)/RT for the Jth species
c @        HUGE        largest machine number
c @        PA(I)       population of the Ith atom type
c @        PC(K)       Kth conditioned population
c @        PMOL(M)     mols in the Mth phase
c -        SMOL(J)     mols of Jth species or false species
c -        SMUL(K)     simplex Lagrange multiplier for the Kth
c                         independent atom
c -        W(I)        work vector
c
c      Variables used only internally:
c          SJ          simplex descent strength
c          DS          path length in MOLS space
c          DSJ         path length for elimination of Jth species
c          JE          species to be eliminated
c          JN          species to become a base
c          KPASS       pass index
c          KPMAX       maximum number of passes
c          KROUND      round index
c          NT          total number of species involved in the simplex
c                         round
c          PMMIN       minimum phase mols
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW)
c-----------------------------------------------------------------------
c   pointers
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
c-----------------------------------------------------------------------
c    extract constants
       NA = IW(IoNA)
       NP = IW(IoNP)
       NS = IW(IoNS)
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
       ZERO = 0
c
c    check option
c
       IF (KOP.GT.10)  THEN
c        set for prescribed rebasing
           NB = IW(IoNB)
           DO 1 J=1,NS
               RW(IoSMOL+J) = 0.0D0
               IoKBJ= IoKB + J
               IF (IW(IoKBJ).EQ.2)  IW(IoKBJ) = 0
               IF (IW(IoKBJ).LT.0)  IW(IoKBJ) = 1
1              CONTINUE
           DO 3 K=1,NB
               J = IW(IoJB+K)
               RW(IoSMOL+J) = RW(IoPC+K)
3              CONTINUE
           JN = KOP - 10
           GOTO 50
           ENDIF
c
       PMMIN = RW(IoHUGE)
       IF (KOP.EQ.1)  THEN
c        find the smallest populated phase mols
           DO 7 M=1,NP
               IF ((RW(IoPMOL+M).GT.ZERO).AND.(RW(IoPMOL+M).LT.PMMIN))
     ;             PMMIN = RW(IoPMOL+M)
7              CONTINUE
           ENDIF
c
c    set to start with false species
       KROUND = 1
       NT = NA + NS
c    allow all atoms
       NB = NA
       DO 9 I=1,NA
           IW(IoIB+I) = I
9          CONTINUE
c    set false species as the basis
       DO 11 K = 1,NB
           J = NS + K
           I = IW(IoIB+K)
           RW(IoSMOL+J) = RW(IoPA+I)
           IW(IoJB+K) = J
           IW(IoKB+J) = 1
11         CONTINUE
       DO 13 J = 1,NS
           IF (IW(IoKB+J).NE.4)  IW(IoKB+J) = 0
           RW(IoSMOL+J) = 0.0D0
           IoFJ = IoF + J
           RW(IoFJ) = RW(IoG+J)
c        check for rebasing call from SJESEP
           IF (KOP.EQ.1)  THEN
c           add phase correction
               M = IW(IoMPJ+J)
               IF (RW(IoPMOL+M).GT.ZERO)  THEN
                       RW(IoFJ) = RW(IoFJ) - LOG(RW(IoPMOL+M))
                   ELSE
                       RW(IoFJ) = RW(IoFJ) - LOG(PMMIN)
                   ENDIF
               ENDIF
13         CONTINUE
c
       KPMAX = 2*NT
       KPASS = 0
c
c -- Round loop point
c
c -- Simplex loop point; Determine the simplex multipliers.
c
20     KPASS = KPASS + 1
c
c    check pass count
       IF (KPASS.GT.KPMAX)  THEN
           IF (KMON.GT.0)  WRITE (KUMO,26)
26         FORMAT (/' Too many passes in SJISMP')
           IW(IoKERR) = 1
           RETURN
           ENDIF
c
c    set up the matrix and rhs
       DO 29 K=1,NB
           J = IW(IoJB+K)
c
c        right hand sides are loaded in SMUL(K)
           IF (KROUND.EQ.1)  THEN
c                round 1; false species distribution
                   IF (J.GT.NS)  THEN
c                        false species
                           IF (RW(IoSMOL+J).GE.ZERO)  THEN
                                   RW(IoSMUL+K) = 1.0D0
                               ELSE
                                   RW(IoSMUL+K) = - 1.0D0
                               ENDIF
                       ELSE
c                        real species
                           RW(IoSMUL+K) = 0.0D0
                       ENDIF
               ELSE
c                round 2; approximate Gibbs minimization
                   RW(IoSMUL+K) = RW(IoF+J)
               ENDIF
c
c        matrix coefficients A(K,L)
           DO 27 L=1,NB
               I = IW(IoIB+L)
               IF (J.GT.NS)  THEN
c                    false species
                       IF (L.EQ.J-NS)  THEN
                               RW(IoA+K+LoA*L) = 1.0D0
                           ELSE
                               RW(IoA+K+LoA*L) = 0.0D0
                           ENDIF
                   ELSE
c                    real species
                       RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J)
                   ENDIF
27             CONTINUE
29         CONTINUE
c
c    solve to put the multipliers in SMUL
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoSMUL+1),NB,IW(IoKERR))
       IF (IW(IoKERR).NE.0)  RETURN
c
c -- select the new basis member JN
c
       SMAX = 0.0D0
       JN = 0
       DO 49 J=1,NT
           IF (IW(IoKB+J).EQ.0)  THEN
c            species J is a trial new base
               IF (KROUND.EQ.1)  THEN
c                    round 1; false species distribution
                       IF (J.GT.NS)  THEN
c                            trial base is false species
                               SJ = - 1.0D0
                           ELSE
c                            trial base is a real species
                               SJ = 0.0D0
                           ENDIF
                   ELSE
c                    round 2; approximate Gibbs minimization
                       SJ = - RW(IoF+J)
                   ENDIF
               TERMX = ABS(SJ)
c
c            add in the simplex multiplier terms
               DO 39 K=1,NB
                   I = IW(IoIB+K)
                   IF (J.GT.NS)  THEN
c                        trial base is a false species
                           IF (K.EQ.J-NS)  THEN
                                   AKL = 1.0D0
                               ELSE
                                   AKL = 0.0D0
                               ENDIF
                       ELSE
c                        trial base is a real species
                           AKL = IW(IoN+I+LoN*J)
                       ENDIF
                   TERM = RW(IoSMUL+K)*AKL
                   SJ = SJ + TERM
                   CALL SJUMAX(TERM,TERMX)
39                 CONTINUE
               CALL SJURND(SJ,TERMX)
               IF (SJ.GT.SMAX)  THEN
c                a better candidate found
                   SMAX = SJ
                   JN = J
                   ENDIF
               ENDIF
49         CONTINUE
c
c -- Check for completion (JN=0)
c
       IF (JN.EQ.0)  GOTO 100
c
c -- Determine the directions of change as JN becomes a basis member.
c
c          W(K) = dSMOL(Kth old basis)/dSMOL(new basis)
c
c    load matrix in A and rhs in W
50     DO  79  K=1,NB
           I = IW(IoIB+K)
c
c        load the RHS
           IF (JN.GT.NS)  THEN
c                new base is a false species
                   IF (K.EQ.JN-NS)  THEN
                           RW(IoW+K) = - 1.0D0
                       ELSE
                           RW(IoW+K) = 0.0D0
                       ENDIF
               ELSE
c                new base is a real species
                   RW(IoW+K) = - IW(IoN+I+LoN*JN)
               ENDIF
c
c        load the matrix
           DO 69 L=1,NB
               J = IW(IoJB+L)
               IF (J.GT.NS) THEN
c                    false species
                       IF (K.EQ.J-NS)  THEN
                               RW(IoA+K+LoA*L) = 1.0D0
                           ELSE
                               RW(IoA+K+LoA*L) = 0.0D0
                           ENDIF
                   ELSE
c                    real species
                       RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J)
                   ENDIF
69             CONTINUE
c
79         CONTINUE
c
c    solve to put the directions in V
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
       IF (IW(IoKERR).NE.0)  RETURN
c
c -- Determine which of the old bases JE will be eliminated
c
       JE = 0
       DS = RW(IoHUGE)
       DO  89  K=1,NB
           J = IW(IoJB+K)
           IF (((RW(IoW+K).LT.ZERO).AND.(RW(IoSMOL+J).GE.ZERO)).OR.
     ;         ((RW(IoW+K).GT.ZERO).AND.(RW(IoSMOL+J).LT.ZERO)))  THEN
c            the mols of J move towards zero on the path
               DSJ = - RW(IoSMOL+J)/RW(IoW+K)
               IF (ABS(DSJ-DS).LE.RW(IoFRND)*DS)  DSJ = DS
               IF ((DSJ.LT.DS).OR.((DSJ.EQ.DS).AND.(J.GT.NS)))  THEN
c                base J is a better replacement candidate
                   DS = DSJ
                   JE = J
                   KE = K
                   ENDIF
               ENDIF
89         CONTINUE
c
c -- check for no elimination
c
       IF (JE.EQ.0)  GOTO 100
c
c -- Make the changes
c
c    modify the old basis set
       DO 91 K=1,NB
           J = IW(IoJB+K)
           IoSMOJ = IoSMOL + J
           TERM = RW(IoSMOJ)
           RW(IoSMOJ) = TERM + RW(IoW+K)*DS
           CALL SJURND(RW(IoSMOJ),TERM)
91         CONTINUE
       RW(IoSMOL+JE) = 0.0D0
c
c    modify the new base
       RW(IoSMOL+JN) = DS
       IW(IoKB+JN) = 1
       IW(IoJB+KE) = JN
c
c    remove old base from base list
       IW(IoKB+JE) = 0
c
c    check for rebasing
       IF (KOP.GT.10)  THEN
           IW(IoKERR) = 0
           RETURN
           ENDIF
c
c -- check for end of false species distribution
c
       IF (KROUND.EQ.1)  THEN
           DO 93 K=1,NB
               IF (IW(IoJB+K).GT.NS)  GOTO 20
93             CONTINUE
c
           GOTO 100
           ENDIF
c
c ---- end of a simplex pass
c
       GOTO 20
c
c ** end of simplex process
c
c    branch on round
100    IF (KROUND.EQ.1)  THEN
c
c -- end of false species distribution
c
c        identify the basis species for the system
           K = 0
           DO 105 L=1,NB
               J = IW(IoJB+L)
                   IF (J.LE.NS)  THEN
c                        real species is a base
                           K = K + 1
                           IW(IoJB+K) = J
                       ELSE
c                        false species remains; ok if zero moles
                           IF (RW(IoSMOL+J).NE.ZERO)  THEN
c                              impossible populations
                               IW(IoKERR) = 3
                               RETURN
                               ENDIF
                       ENDIF
105               CONTINUE
c
c        identify the independent atoms
           K = 0
           DO 109 L=1,NA
               IW(IoLB2+L) = IW(IoIB+L)
               J = NS + L
               IF (IW(IoKB+J).EQ.0)  THEN
                   K = K + 1
                   IW(IoIB+K) = IW(IoIB+L)
                   ENDIF
109            CONTINUE
c
c        set the revised basis size
           NB = K
           IW(IoNB) = NB
c
c        set the dependent atoms
           K = NB
           IF (NB.LT.NA)  THEN
               DO 119 L = 1,NA
                   J = NS + L
                   IF (IW(IoKB+J).NE.0)  THEN
                       K = K + 1
                       IW(IoIB+K) = IW(IoLB2+L)
                       ENDIF
119                CONTINUE
               ENDIF
c
c        go do the approximate Gibbs minimization
           KROUND = 2
           NT = NS
           KPASS = 0
           GOTO 20
c
       ENDIF
c
c -- end of round 2
c
c    check option
       IF (KOP.NE.1)  THEN
c        set simplex multiplier as ELAM estimate
           DO 129 K=1,NB
               RW(IoELAM+K) = RW(IoSMUL+K)
129            CONTINUE
           ENDIF
c
c    normal return
       IW(IoKERR) = 0
       RETURN
c
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
       SUBROUTINE SJISRD(NAMAX,NIW,NRW,IW,RW)
c
c      Simplex species redistribution.
c-----------------------------------------------------------------------
c      Species group notation:
c
c          smf denotes the species for which a target mol fraction
c          is specified.  These are balancing species.  The only other
c          allowed species are the basis species identified in SJISMP.
c
c          bo denotes basis species from SJISMP.
c
c          bx denotes detargeted balancing species.
c
c      SJISRD loads up the smf species as close to their target
c      mol fractions as possible by maximizing
c
c              sum{SMOL(J)}    sum over smf species
c
c          subject to the constraints
c
c              sum{N(I,J)*SMOL(J)} = PA(I) for all independent atoms
c
c              SMOL(J)  ge  0    all bo&bx species
c
c              [X(J)*PMOL(phase of J) - SMOL(J)] ge 0 for smf species
c
c              X(J) is the target mol fraction for the smf species J.
c
c          Works only with the linearly independent atoms,
c             1,...,K,...,NB.
c
c          Only allocates atoms to the smf and bo&bx species.
c-----------------------------------------------------------------------
c          The simplex process is carried out using variables Y(J)
c          defined by
c
c           for bo and bx species:
c
c              Y(J) = SMOL(J)
c
c              Y(J) = - SMOL(J) + X(J)*PMOL(phase of J) smf species
c
c           for smf species:
c
c              Y(Jsmf)= - SMOL(Jsmf)
c                        + X(Jsmf)*[sum{SMOL(smf species in the phase)}
c                                + sum{Y(bo&bx species in the phase)}]
c
c          This is a system of simultaneous linear algebraic equations
c          relating the NSMF  values of SMOL(smf species) to NSMF
c          values of Y(smf species) and to the Y(bo&bx species):
c
c           - SMOL(Jsmf) + X(Jsmf)*sum{SMOL(smf species in the phase)}
c                  =  Y(Jsmf) - X(Jsmf)*sum{Y(bo&bx species in the
c                                            phase)}
c
c           or, using the summation convention,
c
c           A(I,K)*SMOL(JSMF(K)) = Y(JSMF(I))
c                     + B(I,L)*Y(JBO(L))
c
c          where JSMF(K) denotes the J of the Kth smf species,
c          and JBO(L) denotes the J of the Lth bo&bx species.
c
c          The inversion of this gives
c
c          SMOL(JSMF(I)) = Q(I,K)*[Y(JSMF(K)) + B(K,L)*Y(JBO(L))]
c
c          Hence, the function to be minimized in the Simplex process
c          is
c          - sum{Q(I,K)*[B(K,L)*Y(JBO(L)) + Y(JSMF(K))]} all
c            smf species.
c
c          which we recast as
c
c            sum{F(M))*Y(M)} = min       M = 1,...,NS2
c
c          The atomic constraints become
c
c            N(I,JSMF(K))*Q(K,L)*[Y(JSMF(L)) + B(L,M)*Y(JBO(M))]
c
c               + N(I,JBO(K))*Y(JBO(K)) = PA(I)   I=IB(K), K=1,...,NB
c
c          which we recast as
c
c            sum{R(I,L)*Y(L)} = PA(I)     I = IB(K), K=1,NB
c
c          plus the constraints
c
c             Y(M) ge 0  smf and bo species.
c
c          For numerical precision we multiply by the conditioning
c          matrix CM(K,L) and instead work with the conditioned R and
c          the conditioned populations:
c
c            sum{RC(K,L)*Y(L)} = PC(K)      K=1,NB
c
c          The smf species denoted by JSMF above are carried as the
c          first species in JS2, followed by the bo, then the bx,
c          species, then the false species used during initialization.
c
c          At call there are no bx species (all balancers are given
c          targets).  At the conclusion of the maximization described
c          above, any smf species in a phase with no mols is made a bx
c          species, and a second run is then made.  This allows the
c          phase to become populated while maintaining the smf
c          constraints in populated phases.
c
c          At the conclusion, the system base control KB reset to
c          SJISRD bases.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c          IW(I)       integer work array
c          RW(I)       REAL work array
c
c      Variables in the integer work array IW:
c @        IB(K) = I   if the Ith atom is the Kth independent system
c                      atom
c @        JB(K) = J   if the Jth species is the Kth basis
c -        JS2(K) = J  if the Jth species is the Kth SJISRD species
c @        KB(J)       run-start basis control for species J:
c                   -3 for bx species not a SJISRD base at run start
c                   -2 for smf species not a SJISRD base at run start
c                   -1 for bo species not a SJISRD base at run start
c                    1 for bo species a SJISRD base at run sratr
c                    2 for smf species a SJISRD base at run start
c                    3 for bx species a SJISRD base at run start
c                    4 for excluded species
c                    0 for all other species
c @-       KB2(K)      basis control for the Kth SJISRD species
c                   -3 SJISRD y(K) is a bx species not in use as
c                      a y-base
c                   -2 SJISRD y(K) is a smf species not in use as a
c                                  y-base
c                   -1 SJISRD y(K) is a bo species not in use as a
c                      y-base
c                    1 SJISRD y(K) is a bo species in use as a y-base
c                    2 SJISRD y(K) is a smf species in use as a y-base
c                    3 SJISRD y(k) is a bx species in use as a y-base
c @        KMON        monitor control
c @        KUMO        output unit for monitor
c -        LB2(K) = L  if the Lth SJISRD y is the Kth SJISRD basis
c @        MPJ(J) = M  if species J is in phase M
c @        N(I,J)      number of Ith atoms in Jth-species molecule
c @        NB          number of independent bases
c @        NP          number of phases
c @        NS          number of species
c
c      Variables in the real work array RW:
c -        A(I,J)      work array
c -        B(K,L)      bo species influence array (see above)
c @        CM(K,M)     conditioning matrix
c -        EEQN(K)     relative error in the Kth change equation
c -        F(K)        minimizing coefficient; see above
c @        FRND        roundoff number
c @        HUGE        largest machine number
c @#       PC(K)       conditioned population of Kth basis atoms
c @#       PMOL(M)     mols in phase M
c -        Q(I,J)      matrix inverse (y influence on mols)
c -        RC(I,L)     effective N(I,L) of y pseudospecies
c                         (conditioned R)
c @#       SMOL(J)     mols of the Jth species
c -        SMUL(I)     simplex lagrange multiplier for the Ith variable
c -        W(I)        work vector
c @        X(J)        target mol fractions (J for smf species)
c -        Y(L)        Lth element of the solution vector
c
c      Variables used only internally:
c -        NBOBX        total number of bo + bx species
c -        KPASS        pass counter
c -        KPMAX        maximum allowed KPASS
c -        KRUN         run-through counter
c -        NS2          number of SJISRD species
c -        NSMF         number of smf species
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   IW(NIW),RW(NRW)
c-----------------------------------------------------------------------
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
c-----------------------------------------------------------------------
c    integers for compares
       DATA   M1,M2,M3/-1,-2,-3/
c-----------------------------------------------------------------------
c    constants
       FERRF = 10.0D0*RW(IoFRND)
       ZERO = 0.0D0
c
c    get parameters
       NB = IW(IoNB)
       NP = IW(IoNP)
       NS = IW(IoNS)
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
c
c    set control
       KRUN = 1
c
c    set up JSMF and JS2 for smf species
1      NSMF = 0
       DO 2 J=1,NS
           L = IW(IoKB+J)
           IF ((L.EQ.2).OR.(L.EQ.M2))  THEN
               NSMF = NSMF + 1
               IW(IoJS2+NSMF) = J
               IW(IoKB2+NSMF) = L
               ENDIF
2          CONTINUE
c
c    set NS2, JS2, and KB2 for bo species
       NS2 = NSMF
       DO 3 K=1,NB
           J = IW(IoJB+K)
           NS2 = NS2 + 1
           IW(IoJS2+NS2) = J
           IW(IoKB2+NS2) = IW(IoKB+J)
3          CONTINUE
c    set NS2, JS2, and KB2 for bx species
       DO 4 J=1,NS
           L = IW(IoKB+J)
           IF ((L.EQ.3).OR.(L.EQ.M3))  THEN
               NS2 = NS2 + 1
               IW(IoJS2+NS2) = J
               IW(IoKB2+NS2) = L
               ENDIF
4          CONTINUE
       NBOBX = NS2 - NSMF
c
c    check for too many specified species
       IF (NSMF.GT.2*NAMAX) THEN
           IW(IoKERR) = 1
           RETURN
           ENDIF
c
c  set up the A array in B
       DO 9 K=1,NSMF
           JK = IW(IoJS2+K)
           MJK = IW(IoMPJ+JK)
           DO 8 L=1,NSMF
               JL = IW(IoJS2+L)
               MJL = IW(IoMPJ+JL)
               IoBKL = IoB + K + LoB*L
               IF (MJK.EQ.MJL)  THEN
                       RW(IoBKL) = RW(IoX+JK)
                   ELSE
                       RW(IoBKL) = 0.0D0
                   ENDIF
8              CONTINUE
           IoBKK = IoB + K + LoB*K
           RW(IoBKK) = RW(IoBKK) - 1.0D0
9          CONTINUE
c
c        compute the matrix Q
       DO 19 K=1,NSMF
c        load the matrix and rhs
           DO 15 L=1,NSMF
               RW(IoW+L) = 0.0D0
               DO 13 M=1,NSMF
                   RW(IoA+L+LoA*M) = RW(IoB+L+LoB*M)
13                 CONTINUE
15             CONTINUE
           RW(IoW+K) = 1.0D0
c        solve
           CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NSMF,IW(IoKERR))
           IF (IW(IoKERR).NE.0)  RETURN
c        transfer the column
           DO 17 L =1,NSMF
               RW(IoQ+L+LoQ*K) = RW(IoW+L)
17             CONTINUE
19         CONTINUE
c
c    compute the B array
       IF (NBOBX.GT.0)  THEN
           DO 25 K=1,NSMF
               JK = IW(IoJS2+K)
               MJK = IW(IoMPJ+JK)
               DO 23 L=1,NBOBX
                   LX = NSMF + L
                   JL = IW(IoJS2+LX)
                   MJL = IW(IoMPJ+JL)
                   IF (MJK.EQ.MJL)  THEN
                           RW(IoB+K+LoB*L) = - RW(IoX+JK)
                       ELSE
                           RW(IoB+K+LoB*L) = 0.0D0
                       ENDIF
23                 CONTINUE
25             CONTINUE
           ENDIF
c
c    compute the F vector
       DO 39 K=1,NS2
           IoFK = IoF+K
           RW(IoFK) = 0.0D0
           IF (K.LE.NSMF)  THEN
c                smf species
                   DO 31 I=1,NSMF
                       RW(IoFK) = RW(IoFK) - RW(IoQ+I+LoQ*K)
31                     CONTINUE
               ELSE
c                bo&bx species
                   L = K - NSMF
                   DO 35 I=1,NSMF
                       DO 33 M=1,NSMF
                          RW(IoFK) = RW(IoFK) -
     ;                      RW(IoQ+I+LoQ*M)*RW(IoB+M+LoB*L)
33                        CONTINUE
35                     CONTINUE
               ENDIF
39         CONTINUE
c
c    compute the conditioned R array
       DO 59 K=1,NB
           DO 57 L=1,NS2
               JL = IW(IoJS2+L)
               IoRCKL = IoRC + K + LoRC*L
               RW(IoRCKL) = 0.0D0
c            branch on species L type
               IF (L.LE.NSMF)  THEN
c                    smf species
                       DO 45 K1=1,NSMF
                           J1 = IW(IoJS2+K1)
c                        check for a contribution
                           IF ((IW(IoKB+J1).NE.1)
     ;                      .OR.(IW(IoJB+K).EQ.J1)) THEN
                               IF (IW(IoKB+J1).EQ.1)  THEN
c                                    species J1 is the Kth base
                                       SUM1 = 1.0D0
                                   ELSE
c                                    species J1 is not a base
                                       SUM1 = 0.0D0
                                       DO 43 K2=1,NB
                                         I2 = IW(IoIB+K2)
                                         SUM1 = SUM1 +
     ;                                    RW(IoCM+K+LoCM*K2)*
     ;                                      IW(IoN+I2+LoN*J1)
43                                       CONTINUE
                                   ENDIF
                               RW(IoRCKL) = RW(IoRCKL)
     ;                           + SUM1*RW(IoQ+K1+LoQ*L)
                               ENDIF
45                         CONTINUE
                   ELSE
c                    bo&bx species
                       LX = L - NSMF
                       DO 55 K1=1,NSMF
                           SUM1 = 0.0D0
                           DO 53 K2=1,NSMF
                               J2 = IW(IoJS2+K2)
                               IF ((IW(IoKB+J2).NE.1)
     ;                          .OR.(IW(IoJB+K).EQ.J2))  THEN
c                               check for a contribution
                                 IF (IW(IoKB+J2).EQ.1)  THEN
c                                     species J2 is the Kth base
                                       SUM2 = 1.0D0
                                    ELSE
c                                     species J2 is not a base
                                       SUM2 = 0.0D0
                                       DO 51 K3=1,NB
                                         I3 = IW(IoIB+K3)
                                         SUM2 = SUM2 +
     ;                                    RW(IoCM+K+LoCM*K3)*
     ;                                      IW(IoN+I3+LoN*J2)
51                                       CONTINUE
                                    ENDIF
                                 SUM1 = SUM1 + SUM2*RW(IoQ+K2+LoQ*K1)
                                 ENDIF
53                             CONTINUE
                           RW(IoRCKL) = RW(IoRCKL) +
     ;                       SUM1*RW(IoB+K1+LoB*LX)
55                         CONTINUE
c
c                    check type of L species
                       IF (IW(IoKB+JL).EQ.1)  THEN
c                            L is a base; add 1 if its the Kth base
                               IF (L-NSMF.EQ.K)  RW(IoRCKL) =
     ;                          RW(IoRCKL) + 1.0D0
                           ELSE
c                            L is a bx species
                               DO 56 K1=1,NB
                                   I1 = IW(IoIB+K1)
                                   RW(IoRCKL) = RW(IoRCKL)
     ;                            + RW(IoCM+K+LoCM*K1)*
     ;                                IW(IoN+I1+LoN*JL)
56                                 CONTINUE
                           ENDIF
                   ENDIF
57             CONTINUE
59         CONTINUE
c
c -- set initial basis
c
c    clear all mols
       DO 61 J=1,NS
           RW(IoSMOL+J) = 0.0D0
61         CONTINUE
c    set basis mols = conditioned populations
       DO 63 K=1,NB
           J = IW(IoJB+K)
           RW(IoSMOL+J) = RW(IoPC+K)
63         CONTINUE
c
c    set Y
70     DO 71 K=1,NSMF
           J = IW(IoJS2+K)
           M = IW(IoMPJ+J)
           RW(IoY+K) = - RW(IoSMOL+J) + RW(IoX+J)*RW(IoPMOL+M)
71         CONTINUE
       DO 73 L=1,NBOBX
           K = NSMF + L
           J = IW(IoJS2+K)
           RW(IoY+K) = RW(IoSMOL+J)
73         CONTINUE
c
c    set basis pointer
       K = 0
       DO 75 L=1,NS2
           IF (IW(IoKB2+L).GT.0) THEN
               K = K + 1
               IW(IoLB2+K) = L
               ENDIF
75         CONTINUE
c
c    initializations
       KPASS = 0
       KPMAX = 2*NS2
c
c -- Simplex loop point; Determine the simplex multipliers.
c
100    KPASS = KPASS + 1
c
c    check pass count
       IF (KPASS.GT.KPMAX)  THEN
           IF (KMON.GT.0)  WRITE (KUMO,106)
106        FORMAT (/' Too many passes in SJISRD')
           IW(IoKERR) = 1
           RETURN
           ENDIF
c
c    set up the matrix and rhs
       DO 129 K=1,NB
           LK = IW(IoLB2+K)
           RW(IoSMUL+K) = RW(IoF+LK)
           DO 119 L=1,NB
               RW(IoA+K+LoA*L) = RW(IoRC+L+LoRC*LK)
119            CONTINUE
129        CONTINUE
c
c      solve to put the multipliers in SMUL
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoSMUL+1),NB,IW(IoKERR))
       IF (IW(IoKERR).NE.0)  RETURN
c
c ---- select the new basis member LN
c
       SMAX = 0.0D0
       LN = 0
       DO 169 L=1,NS2
           IF (IW(IoKB2+L).LT.0)  THEN
c            SJISRD species L is a trial base base
               KBN = - IW(IoKB2+L)
               SL = - RW(IoF+L)
               TERMX = ABS(SL)
c
c            add in the Simplex multiplier terms
156            DO 167 K=1,NB
                   AKL = RW(IoRC+K+LoRC*L)
                   TERM = RW(IoSMUL+K)*AKL
                   SL = SL + TERM
                   CALL SJUMAX(TERM,TERMX)
167                CONTINUE
               CALL SJURND(SL,TERMX)
               IF (SL.GT.SMAX)  THEN
c                a better candidate found
                   SMAX = SL
                   LN = L
                   ENDIF
               ENDIF
c
169        CONTINUE
c
c -- Check for completion (LN=0)
c
       IF (LN.EQ.0)  GOTO 400
c
c -- Determine the directions of change as LN becomes a basis member.
c
c          W(K) = dY(Kth old basis)/dY(new basis)
c
       KBN = - IW(IoKB2+LN)
c
c    load matrix in A and rhs in W
       DO  179  K=1,NB
           RW(IoW+K) = - RW(IoRC+K+LoRC*LN)
176        DO 177 L=1,NB
               LL = IW(IoLB2+L)
               RW(IoA+K+LoA*L) = RW(IoRC+K+LoRC*LL)
177            CONTINUE
179        CONTINUE
c
c      solve to put the directions in W
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
       IF (IW(IoKERR).NE.0)  RETURN
c
c -- Determine which of the old bases KE will be eliminated
c
       LE = 0
       DS = RW(IoHUGE)
       DO 183 K=1,NB
           LK = IW(IoLB2+K)
           IoWK = IoW + K
           IoYLK = IoY + LK
           IF (((RW(IoWK).LT.ZERO).AND.(RW(IoYLK).GE.ZERO)).OR.
     ;         ((RW(IoWK).GT.ZERO).AND.(RW(IoYLK).LT.ZERO)))  THEN
c            the y of LK move towards zero on the path
               DSL = - RW(IoYLK)/RW(IoWK)
               IF (ABS(DSL-DS).LE.RW(IoFRND)*DS)  DSL = DS
               IF ((DSL.LT.DS).OR.((DSL.EQ.DS).AND.(LK.GT.NS2)))  THEN
c                base LK is a better replacement candidate
                   DS = DSL
                   LE = LK
                   KE = K
                   ENDIF
               ENDIF
183        CONTINUE
c
c -- check for elimination
c
       IF (LE.EQ.0)  GOTO 400
c
c -- solve for the new bases (rather than for changes, for better
c       accuracy)
c
c    remove the old base
       RW(IoY+LE) = 0
       IW(IoKB2+LE) = - IW(IoKB2+LE)
c
c    set the new base
       RW(IoY+LN) = DS
       IW(IoKB2+LN) = - IW(IoKB2+LN)
       IW(IoLB2+KE) = LN
c
c    set up the equation system
       DO 209 K=1,NB
c        load rhs
           RW(IoW+K) = RW(IoPC+K)
c        load matrix
           DO 205 M=1,NB
               LM = IW(IoLB2+M)
               RW(IoA+K+LoA*M) = RW(IoRC+K+LoRC*LM)
205            CONTINUE
209        CONTINUE
c    solve
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
c    check for failure
       IF (IW(IoKERR).NE.0)  RETURN
c    trim and load the solution
       DO 229 K=1,NB
           LK = IW(IoLB2+K)
           IoWK = IoW + K
           IoYLK = IoY + LK
c        zero drop fixup
           IF (((RW(IoWK).LT.ZERO).AND.(RW(IoYLK).GE.ZERO)).OR.
     ;     ((RW(IoWK).GT.ZERO).AND.(RW(IoYLK).LT.ZERO))) RW(IoWK) = 0
           RW(IoYLK) = RW(IoWK)
229        CONTINUE
c
c    check for accuracy of constraint satisfaction
       KFIX = 0
       DO 249 K=1,NB
c        compute equation error in SUM
           SUM = RW(IoPC+K)
           TERMX = ABS(SUM)
           DO 245 L=1,NB
               LL = IW(IoLB2+L)
               AX = RW(IoRC+K+LoRC*LL)
               TERM = AX*RW(IoY+LL)
               CALL SJUMAX(TERM,TERMX)
               SUM = SUM - TERM
245            CONTINUE
c        check relative errors
           IF (TERMX.NE.ZERO)  SUM = ABS(SUM/TERMX)
c        save error
           RW(IoEEQN+K) = SUM
c        check error vs limits
           IF (SUM.GT.FERRF)  KFIX = 1
249        CONTINUE
c
c    skip refinement if not necessary
       IF (KFIX.EQ.0)  GOTO 300
c
c -- Solution refinement
c
c    Solve for the populations of the other bases using the new base
c    value in a shortened set of conditional equations. The most
c    accurately solved equation (KE) is dropped.
c
c    find the most accurately solved equation
250    SUM = RW(IoHUGE)
       KE = 0
       DO 251 K=1,NB
           IF (RW(IoEEQN+K).LT.SUM)  THEN
               SUM = RW(IoEEQN+K)
               KE = K
               ENDIF
251        CONTINUE
c
c    check for inability to fix
       IF (KE.EQ.0)  THEN
           GOTO 300
           ENDIF
c
c    set up the reduced equation system
       K1 = 0
       DO 269 K=1,NB
           IF (K.NE.KE)  THEN
c            load rhs
               K1 = K1 + 1
               RW(IoW+K1) = RW(IoPC+K) -
     ;                  RW(IoRC+K+LoRC*LN)*RW(IoY+LN)
c            load matrix
               M1 = 0
               DO 267 M=1,NB
                   LM = IW(IoLB2+M)
                   IF (LM.NE.LN)  THEN
                       M1 = M1 + 1
                       RW(IoA+K1+LoA*M1) = RW(IoRC+K+LoRC*LM)
                       ENDIF
267                CONTINUE
               ENDIF
269        CONTINUE
c    solve
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),K1,IW(IoKERR))
c    check for failure
        IF (IW(IoKERR).NE.0)  THEN
c        try another equation
           RW(IoEEQN+KE) = RW(IoHUGE)
           GOTO 250
           ENDIF
c
c    transfer the solution
       K1 = 0
       DO 279 K=1,NB
           LK = IW(IoLB2+K)
           IF (LK.NE.LN)  THEN
               K1 = K1 + 1
               RW(IoY+LK) = RW(IoW+K1)
               ENDIF
279        CONTINUE
c
c -- end of pass
c
300   GOTO 100
c
c *** end of run
c
c   compute the mols of smf species
400    DO 409 K=1,NSMF
           J = IW(IoJS2+K)
           RW(IoSMOL+J) = 0
           TERMX = 0
           DO 407 L=1,NSMF
               TERM = RW(IoQ+K+LoQ*L)*RW(IoY+L)
               CALL SJUMAX(TERM,TERMX)
               RW(IoSMOL+J) = RW(IoSMOL+J) + TERM
               SUM = 0
               DO 405 M=1,NB
                   MX = NSMF + M
                   SUM = SUM + RW(IoB+L+LoB*M)*RW(IoY+MX)
405                CONTINUE
               TERM = RW(IoQ+K+LoQ*L)*SUM
               RW(IoSMOL+J) = RW(IoSMOL+J) + TERM
               CALL SJUMAX(TERM,TERMX)
407            CONTINUE
           CALL SJURND(RW(IoSMOL+J),TERMX)
409        CONTINUE
c
c    set mols of bo&bx species
       DO 411 K=1,NBOBX
           KX = NSMF + K
           J = IW(IoJS2+KX)
           L = NSMF + K
           RW(IoSMOL+J) = RW(IoY+L)
411        CONTINUE
c
c    compute phase mols
       DO 413 M=1,NP
           RW(IoPMOL+M) = 0.0D0
413        CONTINUE
       DO 415 K=1,NS2
           J = IW(IoJS2+K)
           M = IW(IoMPJ+J)
           RW(IoPMOL+M) = RW(IoPMOL+M) + RW(IoSMOL+J)
415        CONTINUE
c
c    check run
       IF (KRUN.EQ.1)  THEN
c       check for balancing species in an absent phase
           KSMF = 0
           DO 429 K=1,NSMF
               JK = IW(IoJS2+K)
               IF (RW(IoSMOL+JK).EQ.ZERO)  THEN
c                    balancer has zero mols; check phase
                       DO 427 L=1,NS2
                           JL = IW(IoJS2+L)
                           IF (IW(IoMPJ+JL).EQ.IW(IoMPJ+JK))  THEN
                               IF (RW(IoSMOL+JL).NE.ZERO)  THEN
c                                phase populated ; keep as smf
                                   KSMF = 1
                                   GOTO 429
                                   ENDIF
                               ENDIF
427                        CONTINUE
c                    balancer is in an empty phase; set as bx
                       IF (IW(IoKB2+K).LT.0)  THEN
c                            species is not a current SJISRD base
                               IW(IoKB2+K) = -3
                           ELSE
c                            species is a current SJISRD base
                               IW(IoKB2+K) = 3
                           ENDIF
                       KRUN = 2
                   ELSE
c                    balancer is smf species in a populated phase
                       KSMF = 1
                   ENDIF
429            CONTINUE
c
c        go do a second run if needed and feasible
           IF ((KRUN.EQ.2).AND.(KSMF.EQ.1))  THEN
c            set run-start basis controls
               DO 439 K=1,NS2
                   J = IW(IoJS2+K)
                   IW(IoKB+J) = IW(IoKB2+K)
439                CONTINUE
               GOTO 1
               ENDIF
           ENDIF
c
c ** exit
c
c    reset system basis as SJISRD basis
       NB = 0
       DO 459 L=1,NS2
           J = IW(IoJS2+L)
           IF (IW(IoKB2+L).GT.0)  THEN
c                the species is a system base
                   NB = NB + 1
                   IW(IoJB+NB) = J
                   IW(IoKB+J) = 1
               ELSE
c                the species is not a systems base
                   IW(IoKB+J) = 0
               ENDIF
459        CONTINUE
c
c    exit
       IW(IoKERR) = 0
       RETURN
c
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJIESL(NSMAX,NIW,NRW,CHEM,IW,RW,
     ;                   KOP,DOM,JW)
c
c      Estimates the element potentials ELAM and dominance DOM by
c      species JW.  JW is returned zero if no base is dominated.
c-----------------------------------------------------------------------
c  * For KOP = 0:
c
c    For each base K:
c
c      If PC(K) ne 0 for a base species, the equation used is:
c
c          sum{ELAM(I)*N(I,J)} = G(J) + ln X(J)
c
c      where J denotes the Kth base species.
c
c      If PC(K) = 0 , then an equation relating the ELAMs is
c                     (approximately)
c
c          X(JB) + B*X(JBZ) = 0
c
c      where JBZ is the balancing species. This produces the equation
c      used:
c
c     - G(J) + sum{N(I,J)*ELAM(I)} = ln(-B) - G(JBZ)
c                                             + sum{N(I,JBZ)*ELAM(I)}
c
c      This equation is used if the base and its balancer are in the
c      same phase, or in different phases that are both populated.
c
c      If the base phase is empty, the equation used is X(JB) = 1.
c
c      If a base with non-zero conditioned population is dominated by
c      another species in its equation with a negative coefficient,
c      then the base is treated as if it had a zero conditioned
c      population.
c
c      The calculation is iterated to check balancer selection.
c-----------------------------------------------------------------------
c  * For KOP = 1:
c
c    The mol fraction of each base species is used in the ELAM
c    estimates.
c-----------------------------------------------------------------------
c      On return:
c
c        If solution found:
c
c          KERR = 0
c
c        If there are solution singularities:
c          KERR = 1
c          ELAMs unchanged
c
c        If unbalanceable species are bases:
c          KERR = 4
c          KB(J) = 4 for bases to be removed
c          ELAMs unchanged
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c @        KOP         option control
c #        DOM         largest dominance (see SJIJBZ)
c #        JW          worst dominating species (0 if none)
c
c      Variables in the integer work array IW:
c @        IB(K) = I   if the Kth base atom is the Ith atom
c @        JB(K) = J   if the Kth base species is the Jth species
c #        JBAL(K)     JBZ for the Kth base species
c @        KB(J)       basis control
c                    0 if the Jth species is not a base
c                   -1 if the Jth species is a base freed for
c                      SJISRD call
c                    1 if the Jth species is a base
c  #                 2 if the Jth species is a balancing species
c  #                 4 if the Jth species is excluded
c #        KERR        return error flag (0 ok, >0 problems)
c @        KMON        monitor control
c @        KUMO        output unit for monitor
c @        NB          number of basis species
c @        N(I,J)      number of Ith atoms in the Jth species
c @        NS          number of species
c @        MPJ(J)      phase of species J
c
c      Variables in the real work array RW:
c -        A(K)        work array
c #        BBAL(K)     BZ for Kth base species
c @        CM(K,L)     conditioning matrix
c @        G(J)        g(T,P)/RT for the Jth species
c @        HUGE        largest machine number
c #        ELAM(K)      element potential for the Kth base atom
c @        MOLS(J)     mols of the Jth species
c @        PC(K)       Kth conditioned population
c @        PMOL(M)     mols in phase M
c -        W(K,L)      work array
c @        X(J)        mol fraction of Jth species
c
c      Variables used only internally:
c          B           balacing ceofficient
c          BZ          balancing coefficient
c          JBZ         balancing species
c          KBAB        bad base flag
c          KGO         exit check flag
c          KTRL        iteration counter
c          MJ          phase of the Jth species
c          MJBZ        balancer phase
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16 CHEM
c-----------------------------------------------------------------------
       DIMENSION   CHEM(NSMAX),IW(NIW),RW(NRW)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
c-----------------------------------------------------------------------
c    set constants
        ZERO = 0.0D0
c
c    get parameters
        NB = IW(IoNB)
        NS = Iw(IoNS)
        KUMO = IW(IoKUMO)
        KMON = IW(IoKMON)
c
c  ** reentry point after exchange of base and balancer
c
c    set false balancers to trigger first run
2      DO 5 K=1,NB
           IW(IoJBAL+K) = -1
5          CONTINUE
c
c    set counter
       KTRL = 0
       KBAD = 0
c
c  **  balancer iteration loop point
c
10     KTRL = KTRL + 1
c
       IF (KOP.EQ.0)  THEN
c        initialize dominance
           DOM = - RW(IoHUGE)
           JW= 0
c        set up balancers and check for run
           KGO = 0
           DO 13 K=1,NB
c            get balancer and dominator
                CALL SJIJBZ(NIW,NRW,IW,RW,
     ;                      K,JBZ,RW(IoBBAL+K),JBS,DOMX)
c            check for populated base
               IF (RW(IoPC+K).NE.ZERO)  THEN
c                    check for dominance by balancer
                       IF ((JBS.NE.0).AND.(JBS.EQ.JBZ))  THEN
c                            treat as if zero conditioned population
                               JBALX = JBZ
                               JBS = 0
                           ELSE
c                            treat with specified mol fraction
                               JBALX = 0
                           ENDIF
                   ELSE
c                    set balancer
                       JBALX = JBZ
c                    check for no balancer
                       IF (JBZ.EQ.0)  THEN
c                        set flag
                           J = IW(IoJB+K)
                           IW(IoKB+J) = 4
                           KBAD = 1
                           IF (KMON.GT.1)  WRITE (KUMO,12) CHEM(J)
12                         FORMAT (/'  Absent species: ',A)
                           ENDIF
                   ENDIF
c            compare to current dominance
               IF (JBS.NE.JBZ)  THEN
                   IF (DOMX.GT.DOM)  THEN
c                    this is the new worst case
                       DOM = DOMX
                       IF (DOM.GT.ZERO)  JW = JBS
                       ENDIF
                   ENDIF
c            set controls
               IF (JBALX.NE.IW(IoJBAL+K))  KGO = 1
               IW(IoJBAL+K) = JBALX
13             CONTINUE
c
c        check for excluded species
           IF (KBAD.NE.0)  THEN
               IW(IoKERR) = 4
               RETURN
               ENDIF
c
c        check for converged solution
           IF ((KGO.EQ.0).OR.(KTRL.GT.NB)) GOTO 90
           ENDIF
c
c    clear flags
       DO 15 J=1,NS
           IoKBJ = IoKB + J
           IF (IW(IoKBJ).LT.0)  IW(IoKBJ) = - IW(IoKBJ)
           IF (IW(IoKBJ).EQ.2)  IW(IoKBJ) = 0
15         CONTINUE
c
c    form the rhs and matrix
       DO 39 K=1,NB
c
           IoWK = IoW + K
c        identify base species and its phase
           J = IW(IoJB+K)
           MJ = IW(IoMPJ+J)
c
c        determine appropriate equation for this base
           IF ((KOP.EQ.1).OR.(IW(IoJBAL+K).EQ.0)) THEN
c                set to produce the base mol fraction
                   RW(IoWK) = RW(IoG+J)
                   IF (RW(IoX+J).GT.ZERO)  RW(IoWK) =
     ;                 RW(IoWK) + LOG(RW(IoX+J))
                   DO 17 L=1,NB
                       I = IW(IoIB+L)
                       RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J)
17                     CONTINUE
c                go monitor
                   GOTO 39
c
               ELSE
c
c                identify balancer and its phase
                   JBZ = IW(IoJBAL+K)
                   BZ = RW(IoBBAL+K)
                   MJBZ = IW(IoMPJ+JBZ)
c
c                check for a balance within the phase
                   IF (MJBZ.EQ.MJ)  THEN
c                    set for balance equation within the phase
                       RW(IoW+K) = RW(IoG+J) - RW(IoG+JBZ) + LOG(-BZ)
                       DO 21 L=1,NB
                           I = IW(IoIB+L)
                           RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J)
     ;                       - IW(IoN+I+LoN*JBZ)
21                         CONTINUE
c                    go set balancer commitment
                       GOTO 30
                       ENDIF
c
c                the base and balancer are in different phases
c                 check for zero mols in the base phase
                   IF (RW(IoPMOL+MJ).EQ.ZERO)  THEN
c                   the base phase is empty; set X = 1 for the base
                       RW(IoW+K) = RW(IoG+J)
                       DO 23 L=1,NB
                           I = IW(IoIB+L)
                           RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J)
23                         CONTINUE
c                    go set balancer commitment
                       GOTO 30
                       ENDIF
c
c                the base phase is populated
c                 check for zero mols in the balancer phase
                   IF (RW(IoPMOL+MJBZ).EQ.ZERO)  THEN
c                   the balancer phase is empty
c                    replace the base by the balancer
c                     (the balancer will then be in a populated
c                     phase and can have a target X<1 for
c                     redistribution)
                       IW(IoJB+K) = JBZ
                       IW(IoKB+JBZ) = 1
                       IW(IoKB+J) = 0
c                    recalculate the conditioning matrix and
c                     conditioned populations
                       L = 0
                       CALL SJICPC(NIW,NRW,IW,RW,L)
c                    go redo for the new base set
                       GOTO 2
                       ENDIF
c
c                balance between populated phases
                   B = BZ*RW(IoPMOL+MJBZ)/RW(IoPMOL+MJ)
                   RW(IoW+K) = RW(IoG+J) - RW(IoG+JBZ)  + LOG(-B)
                   DO 27 L=1,NB
                       I = IW(IoIB+L)
                       RW(IoA+K+LoA*L) = IW(IoN+I+LoN*J) -
     ;                   IW(IoN+I+LoN*JBZ)
27                     CONTINUE
c                go set balancer commitment
                   GOTO 30
c
c                balancer commitment; check for previous commitment
30                 IF (IW(IoKB+JBZ).NE.2)  THEN
c                    tag the balancer
                       IW(IoKB+JBZ) = 2
c                    free the base
                       IW(IoKB+J) = -1
                       ENDIF
c
c            end of equation selection
               ENDIF
c
39         CONTINUE
c
c    solve to get the element potentials into ELAM
       CALL SJULES(LoA,RW(IoA+1+LoA),RW(IoW+1),NB,IW(IoKERR))
       IF (IW(IoKERR).NE.0)  RETURN
c
c    put solution into ELAM
       DO 43 K=1,NB
           RW(IoELAM+K) = RW(IoW+K)
43         CONTINUE
c
c    check for iteration
       IF (KOP.EQ.0)  GOTO 10
c
c    normal return
90     IW(IoKERR) = 0
       RETURN
c
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE  SJIJBZ(NIW,NRW,IW,RW,L,JBZ,BZ,JBS,DOM)
c
c
c      For given ELAM and PMOL, finds the principal balancing species
c      JBZ associated with the Lth base species, returning coefficient
c      in BZ.  Also finds the dominator JBS and its dominance DOM.
c-----------------------------------------------------------------------
c      The atomic constraints are
c
c        sum{PMOL(M)*sum{N(I,J)*X(J)}} = PA(I)      I = i,...,NA
c                all gas species
c
c      The conditioning operation removes all base species but one from
c      each of the conditioned equations:
c
c        sum{PMOL(M)*[X(JB(L))+sum{sum{CM(L,K)*N(IB(K),J)*X(J)}}]
c           = PC(L)
c         M sum over phases
c             J sum over non-base species in phase M
c             K sum over base species in phase M
c      where
c
c          PC(M) = sum{CM(L,K)*PA(IB(K))}
c                   K sum over base species in phase M
c
c      If PC(M) le 0, the mol fraction X must be balanced by terms
c      provided by secondary species.  The balance requires a negative
c      coefficient
c
c          BZ = sum{CM(L,K)*N(IB(K),J)}
c                K sum over base species in phase M
c
c      The secondary species J having a negative coefficient and the
c      smallest G(J) is identified as the base balancing species
c      JBZ(L).  The coefficient is included in the selection.
c
c      The balance used in the element potential estimation is
c
c      PMOL(MPJ(JB(L)))*X(JB(L)) +
c      PMOL(MPJ(JBZB(L)))*BZ(L)*X(JBZ(L)) = 0
c
c      If no principal secondary species is identified,
c      JBZ is returned 0.
c
c      The dominance is the ln of the ratio of the dominator term in
c      the conditioned population equation to the base term. It will
c      be negative for an undominated base.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c          IW(I)       integer work array
c          RW(I)       REAL work array
c @        L           base index
c #        JBZ         secondary species balancing base L (0 if none)
c #        BZ          balancing coefficient
c #        JBS         secondary species dominating base L (0 if none)
c #        DOM         dominance of base by dominator
c
c      Variables in the integer work array IW:
c @        IB(K) = I   if the Ith atom is the Kth independent atom
c @        JB(K) = J   if the Jth species is the Kth base species
c @        KB(J)       basis control
c                    0 if the Jth species is not a base
c                   -1 if the Jth species is a base freed for
c                      SIMP2 call
c                    1 if the Jth species is a base
c                    2 if the Jth species is a balancing species
c                    4 if the Jth species is excluded
c @        MPJ(J) = M  if species J is in phase M
c @        N(I,J)      number of the Ith atoms in the Jth species
c @        NB          number of bases
c @        NS          number of species
c
c      Variables in the real work array RW:
c @        CM(M,K)     conditioning matrix
c @        ELAM(K)     estimated element potential of the Kth
c                      independent atom
c @        FRND        roundoff number
c @        G(J)        g(T,P)/RT for the Jth species
c @        HUGE        largest machine number
c @        PMOL(M)     mols of phase M
c
c      Variables used only internally:
c          BESTZ       balancer test function
c          BESTS       dominator test function
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   IW(NIW),RW(NRW),IEPTR(80)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/ IEPTR
       EQUIVALENCE (IoIB,IEPTR(9)),(IoJB,IEPTR(11)),(IoKB,IEPTR(18)),
     ;   (IoMPJ,IEPTR(27)),
     ;   (IoN,IEPTR(28)),(LoN,IEPTR(29)),(IoNB,IEPTR(6)),
     ;   (IoNS,IEPTR(8)),(IoCM,IEPTR(41)),(LoCM,IEPTR(42)),
     ;   (IoELAM,IEPTR(53)),(IoFRND,IEPTR(31)),(IoG,IEPTR(57)),
     ;   (IoHUGE,IEPTR(32)),(IoPMOL,IEPTR(62))
c-----------------------------------------------------------------------
c    set constants
        ZERO = 0.0D0
        ONE = 1.0D0
c
c    recover parameters
        NB = IW(IoNB)
        NS = Iw(IoNS)
c
c    get species and phase
       JL = IW(IoJB+L)
       ML = IW(IoMPJ+JL)
c
c    initialize
       JBZ = 0
       BZ = 0.0D0
       JBS = 0
       BESTZ = - RW(IoHUGE)
       BESTS = - RW(IoHUGE)
c
c    examine all species
       DO 19 J=1,NS
c     check for an included non-base species
           IF ((IW(IoKB+J).EQ.0).OR.(IW(IoKB+J).EQ.2))  THEN
c          non-base species; compute the conditioned coefficient
               SUM = 0
               TERMX = 0
               DO 11 K=1,NB
                   I = IW(IoIB+K)
                   TERM = RW(IoCM+L+LoCM*K)*IW(IoN+I+LoN*J)
                   SUM = SUM + TERM
                   CALL SJUMAX(TERM,TERMX)
11                 CONTINUE
               CALL SJURND(SUM,TERMX)
               IF (SUM.EQ.ZERO)  GOTO 19
c            get phase of test species
               MJ = IW(IoMPJ+J)
c            initiate the exponent argument
               TEST = - RW(IoG+J) + LOG(ABS(SUM))
c            incorporate phase information if known
               IF ((RW(IoPMOL+MJ).NE.ZERO).AND.(RW(IoPMOL+ML).NE.ZERO))
     ;           TEST = TEST + LOG(RW(IoPMOL+MJ)/RW(IoPMOL+ML))
c            incorporate the ELAM terms
               DO 13 K1=1,NB
                   I1 = IW(IoIB+K1)
                   TEST = TEST + RW(IoELAM+K1)*IW(IoN+I1+LoN*J)
13                 CONTINUE
c            check sum
               IF (SUM.LT.ZERO)  THEN
c                potential balancer;  check vs current balancer
                   IF (TEST.GT.BESTZ)  THEN
c                    this is a better choice
                           BESTZ = TEST
                           JBZ = J
                           BZ = SUM
                           ENDIF
                    ENDIF
c            no domination decision if phase empty
               IF (RW(IoPMOL+MJ).NE.ZERO)  THEN
c                check vs current dominator
                   IF (TEST.GT.BESTS)  THEN
c                    this is a better choice
                       BESTS = TEST
                       JBS = J
                       ENDIF
                   ENDIF
               ENDIF
19         CONTINUE
c
c    check for dominator
       IF (JBS.NE.0)  THEN
c            add in the base contributions to dominance
               DOM =  BESTS + RW(IoG+JL)
               DO 23 K1=1,NB
                   I1 = IW(IoIB+K1)
                   DOM = DOM - RW(IoELAM+K1)*IW(IoN+I1+LoN*JL)
23                 CONTINUE
               CALL SJURND(DOM,ONE)
               IF (DOM.LE.ZERO)  JBS = 0
           ELSE
c            set large negative dominance
               DOM = - RW(IoHUGE)
           ENDIF
c
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJERB(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW)
c
c      Rebases the system following a change in the active phases.
c      Checks for permissibility of the demise of phase(s).
c-----------------------------------------------------------------------
c      At call:
c
c          KPCX contains the proposed active phase list
c
c      On return:
c
c          If successful, KERR = 0 and the base system is reset.
c
c          KERR = 1 if the phase change is unfeasible.
c-----------------------------------------------------------------------
c      Sets up a provisional system excluding the species in the absent
c      phase(s). Uses SJISMP to find the base set for this system.  If
c      the populations are impossible in this subset of species, or if
c      the number of bases are changed, the rebasing is not allowed.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       real work array
c
c      Variables in the integer work array IW:
c @#       IB(K) = I   if the Ith atom is the Kth independent system
c                      atom
c -        IBO(K) = I  saved IB
c @#       JB(K) = J   if the Jth system species is the Kth base
c                      species
c -        JBO(K)      saved JB
c @#       KB(J)       basis control
c -        KBO(J)      saved KB
c #        KERR        error flag
c @        KPCX(K)     trial phase control; 0 if phase empty, 1 if
c                         populated
c @        N(I,J)      number of Ith atoms in Jth species
c @        NA          number of atom types
c @        NB          number of base species
c @        NP          number of phases
c @        NS          number of species
c @        NSP(M)      number of species in Mth phase
c
c      Variables used only internally:
c -        NBO         NB for current system
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
c-----------------------------------------------------------------------
c    get parameters
       NP = IW(IoNP)
       NS = IW(IoNS)
c
c    save the present base parameters and load trials
       NBO = IW(IoNB)
       DO 9 K=1,NBO
           IW(IoIBO+K) = IW(IoIB+K)
           IW(IoJBO+K) = IW(IoJB+K)
9          CONTINUE
       DO 19 J=1,NS
           IW(IoKBO+J) = IW(IoKB+J)
19         CONTINUE
c
c    set basis control for rebasing
       J2 = 0
       DO 39 M=1,NP
           J1 =J2 + 1
           J2 = J2 + IW(IoNSP+M)
           DO 29 J=J1,J2
               IF (IW(IoKPCX+M).NE.0)  THEN
c                   phase will remain allow all non-excluded species
                       IF (IW(IoKB+J).NE.4)  IW(IoKB+J) = 0
                   ELSE
c                    phase will be absent
                       IW(IoKB+J) = 4
                   ENDIF
29             CONTINUE
39         CONTINUE
c
c    obtain the provisional base (include phase considerations)
       K = 1
       CALL SJISMP(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW,K)
       IF (IW(IoKERR).NE.0)  RETURN
       NB = IW(IoNB)
c
c    check for illegal rebasing
       IF ((IW(IoKERR).NE.0).OR.(IW(IoNB).NE.NBO))  THEN
c        rebasing failed; restore old bases
           IW(IoNB) = NBO
           DO 43 K=1,NB
               IW(IoIB+K) = IW(IoIBO+K)
               IW(IoJB+K) = IW(IoJBO+K)
43             CONTINUE
           DO 49 J=1,NS
               IW(IoKB+J) = IW(IoKBO+J)
49             CONTINUE
           IW(IoKERR) = 1
           RETURN
           ENDIF
c
c    set the base controls
50     DO 59 J=1,NS
c        remove false exclusion flags
           IF ((IW(IoKB+J).EQ.4).AND.(IW(IoKBO+J).NE.4)) IW(IoKB+J) = 0
59         CONTINUE
c
c    calculate conditioning matrix and conditioned populations
       L = 0
       CALL SJICPC(NIW,NRW,IW,RW,L)
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
C*****precision > double
       DOUBLE PRECISION FUNCTION SJUEXP(X)
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C       REAL FUNCTION SJUEXP(X)
C       IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
c
c      Special DEXP routine.
c      A call must be made before use with X = HUGE to set HUGE.
c-----------------------------------------------------------------------
c      Nomenclature:
c          HUGE        nearly-largest machine number
c-----------------------------------------------------------------------
       SAVE
       DATA K/0/
c-----------------------------------------------------------------------
c    check for first call
       IF (K.EQ.0) THEN
           HUGE = X
           XMAX = LOG(HUGE)
           XMIN = - XMAX
           K = 1
           SJUEXP = HUGE
           RETURN
           ENDIF
c
       IF (X.GT.XMAX) THEN
           SJUEXP = HUGE
           RETURN
           ENDIF
       IF (X.LT.XMIN)  THEN
           SJUEXP = 0.0D0
           RETURN
           ENDIF
       SJUEXP = EXP(X)
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJULES(NDIM,A,V,N,IERR)
c
c          Double precision linear algebraic equation solver.
c
c          Solves A(I,J)*X(J) = Y(I)  I = 1,...,N
c
c          FROUND is a round-off test factor assuming at least 15 digit
c             accuracy
c
c          Uses Gaussian elimination with row normalization and
c          selection.
c
c      At entry:
c          NDIM is the dimension of the A and V arrays
c          Y is in V
c
c      On return:
c
c          Solution ok:
c              IERR = 0
c              X is in V
c              A is destroyed
c
c          Singular matrix or bad dimensioning:
c              IERR = 1
c              A and V are destroyed
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   A(NDIM,NDIM),V(NDIM)
c-----------------------------------------------------------------------
C*****precision > single
C      DATA FROUND/1.E-15/, ZERO/0.0E0/
C*****END precision > single
C
C*****precision > double
       DATA FROUND/1.D-15/, ZERO/0.D0/
C*****END precision > double
C
c-----------------------------------------------------------------------
c -- Check dimensions
c
       IF (N.GT.NDIM)  THEN
c        dimensioning error: treat as program error in EQUIL calls
           IERR = 1
           RETURN
           ENDIF
c
c -- form the lower triagonal system
c
       IF (N.GT.1)  THEN
           NM1 = N - 1
           DO 199 K=1,NM1
c            eliminate the last column in the upper M x M matrix
               M = N - K + 1
               MM1 = M - 1
c
c            normalize each row on its largest emement
c
               DO 119 I=1,M
                   C = 0.0D0
                   DO 111 J=1,M
                       CX = ABS(A(I,J))
                       IF (CX.GT.C)  C = CX
111                    CONTINUE
c                check for singular matrix
                   IF (C.EQ.ZERO)  THEN
                       IERR = 1
                       RETURN
                       ENDIF
                   C = 1.0D0/C
                   DO 113 J=1,M
                       A(I,J) = A(I,J)*C
113                     CONTINUE
                   V(I) = V(I)*C
119                CONTINUE
c            find the best row IX to eliminate in column M
               C = 0.0D0
               DO 121 I=1,M
                   CX=ABS(A(I,M))
                   IF (CX.GT.C)  THEN
                       C = CX
                       IX = I
                       ENDIF
121                CONTINUE
c            check for singular matrix
               IF (C.EQ.ZERO)  THEN
                   IERR = 1
                   RETURN
                   ENDIF
               IF (M.NE.IX)  THEN
c                  switch rows M and IX
                   C = V(M)
                   V(M) = V(IX)
                   V(IX) = C
                   DO 123 J=1,M
                       C = A(M,J)
                       A(M,J) = A(IX,J)
                       A(IX,J) = C
123                    CONTINUE
                   ENDIF
c
c        eliminate last column using the lowest row in the M x M matrix
c
c            check for singular matrix
               IF (A(M,M).EQ.ZERO)  THEN
                   IERR = 1
                   RETURN
                   ENDIF
c            column loop
               DO 139 I=1,MM1
c                check for column entry
                   IF (A(I,M).NE.ZERO)  THEN
c                    eliminate
                       C = A(I,M)/A(M,M)
                       D = V(I) - C*V(M)
                       IF (ABS(D).LT.FROUND*ABS(V(I)))  D = 0.0D0
                       V(I) = D
                       DO 131 J=1,MM1
                           D = A(I,J) - C*A(M,J)
                           IF (ABS(D).LT.FROUND*ABS(A(I,J)))   D = 0.0D0
                           A(I,J) = D
131                        CONTINUE
                       ENDIF
139                CONTINUE
199            CONTINUE
           ENDIF
c
c -- compute the back solution
c
c    check for singular matrix
       IF (A(1,1).EQ.ZERO) THEN
           IERR = 1
           RETURN
           ENDIF
c    calculate X(1)
       V(1) =  V(1)/A(1,1)
       IF (N.GT.1)  THEN
           DO 229 I=2,N
c            calculate X(I)
               IM1 = I - 1
               C = V(I)
               TERMB = ABS(C)
               DO 219 J=1,IM1
                   TERM = A(I,J)*V(J)
                   C = C - TERM
                   IF (ABS(TERM).GT.TERMB)  TERMB = ABS(TERM)
219                CONTINUE
               IF (ABS(C).LT.FROUND*TERMB)  C = 0.0D0
               V(I) = C/A(I,I)
229            CONTINUE
           ENDIF
c
c -- normal exit
c
       IERR = 0
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJUMAX(A,B)
c
c      Sets B = max{|A|,B}
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
C
C       C = ABS(A)
C       IF (C.GT.B)  B = C
C
       B = MAX (B, ABS(A))
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJURND(A,B)
c
c      Rounds A to zero if |A|< FRND*B.
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DATA    FRND/1.D-13/
       IF (ABS(A).LT.FRND*B)  A = 0.0D0
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJEPTS(NAMAX,NPMAX,NSMAX,NIW,NRW,NSW,RW,KU)
c
c      Sets SJEQLB pointers and checks specified work array dimensions.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in argument list
c
c          NAMAX       maximum number of atom types
c          NPMAX       maximum number of phases
c          NSMAX       maximum number of species
c          NIW         dimension of work array IW
c          NRW         dimension of work array RW
c          RW(I)       REAL work array
c          KU          output unit for error message
c-----------------------------------------------------------------------
c    Targets:
c      NW = max{2*NA,NA+NP}
c      NIW = 8 + 14*NA + 4*NP + 5*NS + NA*NS
c      NRW = 5 + 16*NA + 12*NA*NA + 3*NA*NP + 4*NP +
c    ;       8*NS + NW*NW + NW
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   RW(NRW)
c-----------------------------------------------------------------------
c 80 pointers required by SJEQLB
       COMMON /SJEPTR/
     ;   IoKERR,IoKMON,IoKTRE,IoKUMO,IoNA,IoNB,IoNP,IoNS,IoIB,IoIBO,
     ;   IoJB,IoJBAL,IoJBA,IoJBB,IoJBO,IoJBX,IoJS2,IoKB,IoKB2,IoKBA,
     ;   IoKBB,IoKBO,IoKPC,IoKPCX,IoLB2,IoMPA,IoMPJ,IoN,LoN,IoNSP,
     ;   IoFRND,IoHUGE,IoR1,IoR2,IoR3,IoA,LoA,IoB,LoB,IoBBAL,
     ;   IoCM,LoCM,IoD,LoD,IoDC,LoDC,IoDPML,IoDLAM,IoDLAY,IoE,
     ;   LoE,IoEEQN,IoELAM,IoELMA,IoELMB,IoF,IoG,IoHA,IoHC,IoPA,
     ;   IoPC,IoPMOL,IoQ,LoQ,IoQC,LoQC,IoRC,LoRC,IoRL,IoRP,
     ;   IoSMOA,IoSMOB,IoSMOO,IoSMOL,IoSMUL,IoW,IoX,IoXO,IoY,IoZ
c-----------------------------------------------------------------------
c    round-off factor in computation precision
c    (1.D-12 for DOUBLE PRECISION)
       FRND = 1.D-12
c-----------------------------------------------------------------------
c    length combinations
       IF (NPMAX.GT.NAMAX)  THEN
               NW = NAMAX + NPMAX
           ELSE
               NW = NAMAX + NAMAX
           ENDIF
       NAMAX2 = 2*NAMAX
c ** IW pointers:
       IoKERR = 1
       IoKMON = IoKERR + 1
       IoKTRE = IoKMON + 1
       IoKUMO = IoKTRE + 1
       IoNA = IoKUMO + 1
       IoNB = IoNA + 1
       IoNP = IoNB + 1
       IoNS = IoNP + 1
       IoIB = IoNS
       IoIBO = IoIB + NAMAX
       IoJB = IoIBO + NAMAX
       IoJBAL = IoJB + NAMAX
       IoJBA = IoJBAL + NAMAX
       IoJBB = IoJBA + NAMAX
       IoJBO = IoJBB + NAMAX
       IoJBX = IoJBO + NAMAX
       IoJS2 = IoJBX + NAMAX
       IoKB = IoJS2 + 2*NAMAX
       IoKB2 = IoKB + NSMAX + NAMAX
       IoKBA = IoKB2 + 2*NAMAX
       IoKBB = IoKBA + NSMAX
       IoKBO = IoKBB + NSMAX
       IoKPC = IoKBO + NSMAX
       IoKPCX = IoKPC + NPMAX
       IoLB2 = IoKPCX + NPMAX
       IoMPA = IoLB2 + NAMAX
       IoMPJ = IoMPA + NPMAX
       LoN = NAMAX
       IoN = IoMPJ + NSMAX - LoN
       IoNSP = IoN + NAMAX*NSMAX + LoN
c    check IW dimension
       NIWX =  IoNSP + NPMAX
       IF (NIWX.NE.NIW) THEN
           WRITE (KU,1) NAMAX,NPMAX,NSMAX
1          FORMAT (/' SJEQLB dimensioning error for NAMAX =',I3,
     ;              '  NPMAX =',I3,'  NSMAX =',I3)
           WRITE (KU,2) NIWX
2          FORMAT (/'  NIWORK error; NIWX =',I6)
           STOP
           ENDIF
c
c ** RW pointers:
       IoFRND = 1
       IoHUGE = IoFRND + 1
       IoR1 = IoHUGE + 1
       IoR2 = IoR1 + 1
       IoR3 = IoR2 + 1
       LoA = NW
       IoA = IoR3 - LoA
       LoB = NAMAX2
       IoB = IoA + NW*NW + LoA - LoB
       IoBBAL = IoB + NAMAX2*NAMAX2 + LoB
       LoCM = NAMAX
       IoCM = IoBBAL + NAMAX - LoCM
       LoD = NAMAX
       IoD = IoCM + NAMAX*NAMAX + LoCM - LoD
       LoDC = NAMAX
       IoDC = IoD + NAMAX*NPMAX + LoD - LoDC
       IoDPML = IoDC + NAMAX*NPMAX + LoDC
       IoDLAM = IoDPML + NPMAX
       IoDLAY = IoDLAM + NAMAX
       LoE = NAMAX
       IoE = IoDLAY + NAMAX - LoE
       IoEEQN = IoE + NAMAX*NPMAX + LoE
       IoELAM = IoEEQN + NAMAX
       IoELMA = IoELAM + NAMAX
       IoELMB = IoELMA + NAMAX
       IoF = IoELMB + NAMAX
       IoG = IoF + NSMAX
       IoHA = IoG + NSMAX
       IoHC = IoHA + NAMAX
       IoPA = IoHC + NAMAX
       IoPC = IoPA + NAMAX
       IoPMOL = IoPC + NAMAX
       LoQ = NAMAX2
       IoQ = IoPMOL + NPMAX - LoQ
       LoQC = NAMAX
       IoQC = IoQ + NAMAX2*NAMAX2 + LoQ - LoQC
       LoRC = 2*NAMAX
       IoRC = IoQC + NAMAX*NAMAX + LoQC - LoRC
       IoRL = IoRC + 2*NAMAX*NAMAX + LoRC
       IoRP = IoRL + NAMAX
       IoSMOA = IoRP + NPMAX
       IoSMOB = IoSMOA + NSMAX
       IoSMOO = IoSMOB + NSMAX
       IoSMOL = IoSMOO + NSMAX
       IoSMUL = IoSMOL + NSMAX + NAMAX
       IoW = IoSMUL + NAMAX
       IoX = IoW + NW
       IoXO = IoX + NSMAX
       IoY = IoXO + NSMAX
       IoZ = IoY + NAMAX2
       NRWX = IoZ + NPMAX
c    check RW dimension
       IF (NRWX.NE.NRW) THEN
               WRITE (KU,1) NAMAX,NPMAX,NSMAX
               WRITE (KU,4) NRWX
4              FORMAT (/'  NRWORK error; NRWX =',I6)
               STOP
           ELSE
               RW(IoFRND) = FRND
c            set HUGE in the exponential routine
               X = RW(IoHUGE)
               Y = SJUEXP(X)
               IF (Y.NE.X)  THEN
                   WRITE (KU,6) X,Y
6                  FORMAT (/' Error setting HUGE in SJUEXP')
                   STOP
                   ENDIF
           ENDIF
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJRC(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW)
c
c      Calculates sound speed for the current state, obeying KFRZ.
c
c      Must be called after a completed state calculation.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        NSW         dimension of work array IS
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c          SW(I)       REAL*4 work array
c
c      Variables in the integer work array RW:
c #        KERR        error flag
c @        KFRZ        frozen/constraint control
c @        NP          number of phases
c @        NS          number of species
c
c      Variables in the real work array RW:
c #        C           isentropic sound speed, m/s
c @        P           pressure, Pa
c #        S           entropy, J/kg
c @        SMOL(J)     mols of the Jth species
c -        SMOZ(J)     saved SMOLS
c @        T           temperature, K
c @        V           mixture specific volume, m**3/kg
c
c      Variables used only internally:
c          vX          v at a forward perturbation
c          vY          v at a backward perturbation
c          DRDT        dR/dT at constant P
c          DSDT        dS/dT at constant P
c          DRDP        dS/dP at constant T
c          DSDP        dS/dP at constant T
c          DTDPS       dT/dP at constant S
c          DPDRS       dP/dR at constant entropy
c          DRDPS       dR/dP at constant entropy
c          PZ          saved P
c          RX          density at forward perturbation
c          RY          density at backward perturbation
c          SZ          saved S
c          TZ          saved T
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCC       REAL*4      SW
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW),SW(NSW),
     ;   IEPTR(80),IRPTR(20),ITPTR(20)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/ IEPTR
       COMMON /SJRPTR/ IRPTR
       COMMON /SJTPTR/ ITPTR
       EQUIVALENCE (IoKERR,IEPTR(1)),(IoKFRZ,ITPTR(1)),(IoNP,IEPTR(7)),
     ;     (IoNS,IEPTR(8)),(IoC,IRPTR(10)),(IoP,ITPTR(5)),
     ;     (IoS,ITPTR(8)),(IoSMOL,IEPTR(74)),(IoSMOZ,IRPTR(20)),
     ;     (IoT,ITPTR(6)),(IoV,ITPTR(10))
c-----------------------------------------------------------------------
c    comparison constant in computation precisison
       ZERO = 0.0D0
c
c    get parameters
       NP = IW(IoNP)
       NS = IW(IoNS)

c    save data needed to restore the current state
       PZ = RW(IoP)
       TZ = RW(IoT)
       DO 9 J=1,NS
           RW(IoSMOZ+J) = RW(IoSMOL+J)
9          CONTINUE
c
c -- sound speed computation as sqrt{dP/dRHO at constant S}
c
c    check for single-phase system
       IF (NP.EQ.1)  THEN
c
c          single phase
               KINIT = 3
c          temperature perturbations
               DT = 0.001D0*TZ
c            forward perturbation
               RW(IoT) = TZ + DT
               CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;             ATOM,CHEM,IW,RW,SW,KINIT)
               IF (IW(IoKERR).NE.0)  GOTO 90
               SX = RW(IoS)
               RX = 1.0D0/RW(IoV)
c            backward perturbation
               RW(IoT) = TZ - DT
               CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;             ATOM,CHEM,IW,RW,SW,KINIT)
               IF (IW(IoKERR).NE.0)  GOTO 90
               SY = RW(IoS)
               RY = 1.0D0/RW(IoV)
               DT = 2.0D0*DT
               DSDT = (SX - SY)/DT
               DRDT = (RX - RY)/DT
c
c          pressure perturbations
               RW(IoT) = TZ
               DP = 0.001D0*PZ
c            forward perturbation
               RW(IoP) = PZ + DP
               CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;             ATOM,CHEM,IW,RW,SW,KINIT)
               IF (IW(IoKERR).NE.0)  GOTO 90
               SX = RW(IoS)
               RX = 1.0D0/RW(IoV)
c            backward perturbation
               RW(IoP) = PZ - DP
               CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;             ATOM,CHEM,IW,RW,SW,KINIT)
               IF (IW(IoKERR).NE.0)  GOTO 90
               SY = RW(IoS)
               RY = 1.0D0/RW(IoV)
               DP = 2.0D0*DP
               DSDP = (SX - SY)/DP
               DRDP = (RX - RY)/DP
c
c          calculate sound speed
               IF (DSDT.EQ.ZERO)  GOTO 90
               DTDPS = - DSDP/DSDT
               DRDPS = DRDT*DTDPS + DRDP
               IF (DRDPS.LE.ZERO)  GOTO 90
               RW(IoC) = SQRT(1.0D0/DRDPS)
c
           ELSE
c
c           multi-phase system; requires changes at constant S
               SZ = RW(IoS)
c            forward perturbation
               DP = 0.01D0*PZ
               RW(IoP) = PZ + DP
               MODE = 6
               CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                     ATOM,CHEM,IW,RW,SW,MODE)
               IF (IW(IoKERR).NE.0)  GOTO 90
               RX = 1.0D0/RW(IoV)
c            backward perturbation
               RW(IoP) = 2.0D0*PZ - RW(IoP)
               RW(IoS) = SZ
               RW(IoT) = 2.0D0*TZ - RW(IoT)
               CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                     ATOM,CHEM,IW,RW,SW,MODE)
               IF (IW(IoKERR).NE.0)  GOTO 90
               RY = 1.0D0/RW(IoV)
c            compute sound speed
               DR = RX - RY
               IF (DR.LE.ZERO)  GOTO 90
               DPDRS = 2.0D0*DP/DR
               RW(IoC) = SQRT(DPDRS)
           ENDIF
c
c    restore the state
10     RW(IoT) = TZ
       RW(IoP) = PZ
       DO 19 J=1,NS
           RW(IoSMOL+J) = RW(IoSMOZ+J)
19         CONTINUE
       KINIT = 3
       CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;            ATOM,CHEM,IW,RW,SW,KINIT)
       RETURN
c
c    error in sound speed; set a zero
c90     RW(IoC) = O
90     RW(IoC) = ZERO
       GOTO 10
c
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJRCJ(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW)
c
c      Chapman-Jouguet detonation analysis
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        NSW         dimension of work array IS
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c          SW(I)       REAL*4 work array
c
c      Variables in the integer work array IW:
c #        KERR        error flag
c @        KMON        monitor control
c @        KUMO        unit for monitor output
c
c      Variables in the real work array RW:
c      Note: At call P,T,H,V are unburned properties, TB burned T
c            estimate.
c            On return all properties are of the burned mixture.
c #        C           sound speed in burned gas, m/s
c #        CDET        detonation wave speed, m/s
c @#       H           mixture enthalpy, J/kg
c #        H1          unburned mixture enthalpy, J/kg
c @        HUGE        very large number
c @#       P           pressure, Pa
c #        P1          unburned mixture pressure, Pa
c @#       T           temperature, K
c @        TE          estimated T
c @#       V           mixture specific volume, m**3/kg
c #        V1          unburned mixture volume, m**3/kg
c
c      Variables used internally
c          CS2         sound speed in the burned state gas, m/s
c          ERRFT       error factor for T2
c          ERRFV       error factor for V2
c          FH          stagnation enthalpy jump across wave (made 0)
c          FP          impluse jump across wave (made 0)
c          H2          unburned mixture enthalpy, J/kg
c          KTRCD       iteration counter
c          MODE        SJROPX run control
c          P2          burned mixture pressure, Pa
c          R1          unburned mixture density, kg/m**3
c          R2          burned mixture density, kg/m**3
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC       REAL*4          SW
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW),SW(NSW),
     ;   IEPTR(80),ITPTR(20)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/ IEPTR
       COMMON/ SJRPTR/
     ;   IoKDET,IoKRCT,IoKSND,IoKTRI,IoKTRT,IoI2,IoI3,IoI4,IoTE,IoC,
     ;   IoCDET,IoH1,IoP1,IoT1,IoV1,IoR5,IoR6,IoSMFA,IoSMFB,IoSMOZ
       COMMON /SJTPTR/ ITPTR
         EQUIVALENCE (IoH,ITPTR(7)),(IoP,ITPTR(5)),(IoT,ITPTR(6)),
     ;     (IoV,ITPTR(10)),(IoHUGE,IEPTR(32)),
     ;     (IoKERR,IEPTR(1)),(IoKMON,IEPTR(2)),(IoKUMO,IEPTR(4))
c    maximum iterations
       DATA KTRCDM/20/
c    error parameters in calculation precision
       ERRFT = 1.D-4
       ERRFV = 1.D-4
c-----------------------------------------------------------------------
       ZERO = 0.0D0
c    get parameters
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
c    monitor control
       IF (KMON.EQ.1)  THEN
               MONCJ = 1
               IW(IoKMON) = 0
               WRITE (KUMO,2)
2              FORMAT(' Beginning lengthy detonation calculation.')
           ELSE
               MONCJ = KMON
           ENDIF
c
c    save unburned mixture properties
       RW(IoP1) = RW(IoP)
       RW(IoV1) = RW(IoV)
       R1 = 1.0D0/RW(IoV1)
       RW(IoH1) = RW(IoH)
c
c   estimate burned gas state
       RW(IoT) = RW(IoTE)
       RW(IoV) = RW(IoV)/1.8D0
       RW(IoP) = RW(IoP)*RW(IoTE)/300D0
c
c    other initialization
       DT = RW(IoHUGE)
       DV = RW(IoHUGE)
       KTRCD = 0
c
c -- loop point
c
c    calculate state for trial T, V
c
10     KTRCD = KTRCD + 1
       MODE = 2
       CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,MODE)
       IF (IW(IoKERR).NE.0)  RETURN
c
c    save state
       T2 = RW(IoT)
       H2 = RW(IoH)
       P2 = RW(IoP)
       V2 = RW(IoV)
       R2 = 1.0D0/V2
c
c    get stagnation enthalpy and impulse changes through wave
       CALL SJRCJX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,
     ;             RW(IoP1),P2,RW(IoH1),H2,R1,R2,FH,FP)
c
c    monitor
       IF (MONCJ.GT.0)  WRITE (KUMO,12) KTRCD,
     ;         T2,P2,V2,FH,FP
12     FORMAT (/' C-J detonation trial ',I4/
     ;          ' T2, P2, V2 =',3(1PE15.7)/
     ;          ' stagnation enthalpy mismatch =',E12.4,5X,
     ;          ' impulse mismatch =',E12.4)
c
c    check convergence
       IF ((ABS(DT).LT.ERRFT*RW(IoT)).AND.
     ;     (ABS(DV).LT.ERRFV*RW(IoV)))  THEN
c         exit calculation
           RW(IoCDET) = RW(IoC)*RW(IoV1)/V2
           RETURN
           ENDIF
c
c    check count
       IF (KTRCD.GT.KTRCDM)  THEN
           IW(IoKERR) = 2
           RETURN
           ENDIF
c
c -- perturbations to get derivatives
c
c    temperature perturbation
       DT = 0.02D0*T2
       RW(IoT) = T2 + DT
       RW(IoV) = V2
       CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,MODE)
       CALL SJRCJX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,
     ;             RW(IoP1),RW(IoP),RW(IoH1),RW(IoH),R1,R2,FHX,FPX)
       DFHDT = (FHX - FH)/DT
       DFPDT = (FPX - FP)/DT
       DPDT = (RW(IoP) - P2)/DT
c
c    volume perturbation
       DV = 0.02D0*V2
       RW(IoV) = V2 + DV
       RW(IoT) = T2
       CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,MODE)
       R2X = 1.0D0/RW(IoV)
       CALL SJRCJX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,
     ;             RW(IoP1),RW(IoP),RW(IoH1),RW(IoH),R1,R2X,FHX,FPX)
       DFHDV = (FHX - FH)/DV
       DFPDV = (FPX - FP)/DV
       DPDV = (RW(IoP) - P2)/DV
c
c    solve for changes
       DET = DFHDT*DFPDV - DFPDT*DFHDV
       IF (DET.EQ.ZERO)  THEN
           IW(IoKERR) = 1
           RETURN
           ENDIF
       DT = ( - FH*DFPDV + FP*DFHDV)/DET
       DV = ( - DFHDT*FP + DFPDT*FH)/DET
       DP = DPDT*DT + DPDV*DV
c
c -- limit the magnitudes of changes
c
c    limit temperature change
       DTM = 0.2D0*RW(IoT)
       IF (ABS(DT).GT.DTM) DT = DTM*DT/ABS(DT)
c
c    limit volume change  (volume must be smaller than for unburned gas)
       V2X = V2 + DV
       IF (V2X.GE.RW(IoV1)) THEN
               DVM = 0.5D0*(RW(IoV1) - V2)
           ELSE
               DVM = 0.2D0*V2
           ENDIF
       IF (ABS(DV).GT.DVM) DV = DVM*DV/ABS(DV)
c
c    limit pressure changes
       IF (DP.LT.ZERO)  THEN
              DPM = 0.5D0*RW(IoP)
           ELSE
              DPM = 2.0D0*RW(IoP)
           ENDIF
       IF (ABS(DP).GT.DPM) DP = DPM*DP/ABS(DP)
c
c -- make the changes
           RW(IoT) = T2 + DT
           RW(IoV) = V2 + DV
           RW(IoP) = P2 + DP
           GOTO 10
c
           END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJRCJX(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                   ATOM,CHEM,IW,RW,SW,
     ;                   P1,P2,H1,H2,R1,R2,FH,FP)
c
c      Calculates sound speed and stagnation enthalpy change FH and
c      impulse change FP for Chapman-Jouguet jump from state 1
c      to state 2.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        NSW         dimension of work array IS
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c          SW(I)       REAL*4 work array
c @        P1          unburned mixture pressure, Pa
c @        P2          burned gas pressure, Pa
c @        H1          unburned gas enthalpy, J/kg
c @        H2          burned gas enthalpy, J/kg
c @        R1          unburned gas density, kg/m**3
c @        R2          burned gas density, kg/m**3
c @        FH          enthalpy jump, J/kg
c @        FP          impulse jump,Pa
c
c      Variables in the integer work array IW:
c          KERR        error flag
c
c      Variables in the real work array RW:
c          C           sound speed, m/s
c
c      Variables used only internally:
c          V1S     square of unburned velocity with respect to wave,
c                  (m/s)**2
c          V2S     square of burned velocity with respect to wave,
c                  (m/s)**2
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC       REAL*4      SW
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW),SW(NSW),
     ;   IEPTR(80),IRPTR(20)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/ IEPTR
       COMMON /SJRPTR/ IRPTR
       EQUIVALENCE (IoKERR,IEPTR(1)),(IoC,IRPTR(10))
c-----------------------------------------------------------------------
c    calculate sound speed in burned gas
       CALL SJRC(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW)
       IF (IW(IoKERR).NE.0)  RETURN
       V2S = RW(IoC)*RW(IoC)
c    unburned speed relative to wave set by continuity
       V1S = V2S*(R2/R1)**2
c    stagnation enthalpy change
       FH = H2 + 0.5D0*V2S - (H1 + 0.5D0*V1S)
c    impulse change
       FP = P2 + R2*V2S - (P1 + R1*V1S)
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJRKCI(KCI,L)
c
c      ANDs KCI with L for L = 1 or 2;  used for setting interpolation
c      control.
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       IF (KCI.EQ.3)  RETURN
       IF (KCI.EQ.L)  RETURN
       IF (KCI.EQ.0)  THEN
               KCI = L
               RETURN
           ELSE
               KCI = 3
               RETURN
           ENDIF
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJROP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                  ATOM,CHEM,IW,RW,SW,NOP)
c
c      Calculates equilibrium properties for a specified pair of
c      state variables, as set by NOP:
c
c      Option control NOP:
c          1   Specified T and P
c          2   Specified T and V
c          3   Specified T and S
c          4   Specified P and V
c          5   Specified P and H
c          6   Specified P and S
c          7   Specified V and U
c          8   Specified V and H
c          9   Specified V and S
c         10   Chapman-Jouguet detonation
c              (H,S,V,T contain unburned state, T2 is burned estimate)
c
c          Problems encountered:
c
c      KERR = 1 program problems; refer to user for optional
c                    rerun with diagnostic output.
c             2 convergence failure.
c             3 impossible populations; refer to user for new input.
c-----------------------------------------------------------------------
c      Calls SJTP for T,P specified.  If one-parameter iterations are
c      required, or two-parameter iterations with V not specified,
c      SJROPX does the iterations.  Two parameter adjustments in multi-
c      phase systems with V specified are handled as a series of one-
c      parameter adjustments at specified T and V, and then T is
c      adjusted in SJROP.  This enables the difficult problem of
c      multi-phase equilibrium to be handled properly. See SJROPX for
c      more details.
c
c      A bracketing scheme is used to enable handling of strongly
c      S-shaped curves. The adjustment variable is dedined to be Z,
c      a monotone increasing function of T.  T is adjusted by
c      bracketing between upper and lower bounds.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension  work array IW
c @        NRW         dimension of work array IR
c @        NSW         dimension of work array IS
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c          SW(I)       REAL*4 work array
c @        NOP         run option; see above
c
c      Variables in the integer work array IW:
c @        JFS(J) = JF if the Jth species is the JFth file species
c                      (0 unfiled)
c #        KERR        error flag
c @        KMON        runtime monitor control
c #        KTRE        SJEQLB passes for one call
c #        KTRI        total (T,P) adjustments for an SJRUN call
c #        KTRT        total SJEQLB iterations for an SJRUN call
c @        KUMO        output unit for runtime monitor
c @        NA          number of atom types
c @        NP          number of phases
c @        NS          number of species
c
c      Variables in the real work array RW:
c ?        H           mixture enthalpy, J/kg
c @        HUGE        large machine number
c ?        P           pressure, Pa
c ?        S           mixture entropy, J/kg-K
c ?        T           temperature, K
c ?        TE          T estimate for detonation calculation
c ?        U           mixture internal energy, J/kg
c ?        V           mixture specific volume, m**3/kg
c
c      Variables used only internally:
c          DP          change in P
c          DPDT        dP/dT in adjustments
c          DT          change in T
c          DZDT        dZ/dT at in adjustments
c          DZDTS       dZ/dT as estimated from the bounds.
c          ERRT        DT for converged solution
c          KCI         bracket state; 0 none, 1 below, 2 above,
c                         3 above/below
c          KTRS        iteration counter
c          KTRSM       maximum iterations allowed
c          TA          upper bound on T
c          TB          lower bound on T
c          Z           generalized adjustment variable; monotone incr.
c                      w/T
c          ZA          upper bound on Z, at TA
c          ZB          lower bound on Z, at TB
c          ZTAR        generalized target
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCCC       REAL*4     SW
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW),SW(NSW),
     ;    IEPTR(80),IRPTR(20),ISPTR(10),ITPTR(20)
c-----------------------------------------------------------------------
       COMMON /SJEPTR/ IEPTR
       COMMON /SJRPTR/ IRPTR
       COMMON /SJSPTR/ ISPTR
       COMMON /SJTPTR/ ITPTR
       EQUIVALENCE (IoJFS,ISPTR(6)),(IoKERR,IEPTR(1)),
     ;   (IoKFRZ,ITPTR(1)),(IoKMON,IEPTR(2)),(IoKTRE,IEPTR(3)),
     ;   (IoKUMO,IEPTR(4)),(IoKTRI,IRPTR(4)),(IoKTRT,IRPTR(5)),
     ;   (IoNA,IEPTR(5)),(IoNP,IEPTR(7)),(IoNS,IEPTR(8)),
     ;   (IoHUGE,IEPTR(32)),(IoTE,IRPTR(9)),
     ;   (IoP,ITPTR(5)),(IoT,ITPTR(6)),(IoH,ITPTR(7)),
     ;   (IoS,ITPTR(8)),(IoU,ITPTR(9)),(IoV,ITPTR(10))
c-----------------------------------------------------------------------
c    iteration maximum
       DATA    KTRSM/40/
c    error control parameter in computation precision
       ERRT = 0.1D0
c       ERRT = 0.01
c-----------------------------------------------------------------------
c    comparison constant
       ZERO = 0D0
c     get parameters
       NP = IW(IoNP)
       KMON = IW(IoKMON)
       KUMO = IW(IoKUMO)
c
c -- monitor
c
       IF (KMON.GT.0)  WRITE (KUMO,2)
2      FORMAT (//' Start of run monitor output')
c
c -- initialize
c
       IW(IoKTRI) = 0
       IW(IoKTRT) = 0
c
c -- check for iteration
c
       IF (((NP.EQ.1).OR.(NOP.LT.7)).AND.(NOP.NE.10)) THEN
c        direct solution
           CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                   ATOM,CHEM,IW,RW,SW,NOP)
           GOTO 900
           ENDIF
c
c ** iteration required
c
4      KTRS = 0
       KCI = 0
c
c    save target
c
       L = NOP - 6
       GOTO (7,8,9,10),  L
c
c    KSJRUN = 7; specified V and U
7      ZTAR = RW(IoU)
       GOTO 18
c
c    KSJRUN = 8; specified V and H
8      ZTAR = RW(IoH)
       GOTO 18
c
c    KSJRUN = 9; specified V and S
9      ZTAR = RW(IoS)
       GOTO 18
c
c    KSJRUN = 10: Chapman-Jouguet detonation
10     CALL SJRCJ(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW)
       GOTO 900
c
c    set false limits
18     DT = RW(IoHUGE)
       ZA = RW(IoHUGE)
       ZB = - RW(IoHUGE)
       DZDTS = 1.0D0/RW(IoHUGE)
c
c -- iteration loop point
c
20     KTRS = KTRS + 1
       L = 2
       CALL SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,L)
c
c    check error and convergence
       IF ((IW(IoKERR).NE.0).OR.(ABS(DT).LT.ERRT)) GOTO 900
c
c ** not converged
c
c    check count
       IF (KTRS.GT.KTRSM)  THEN
           IW(IoKERR) = 2
           RETURN
           ENDIF
c
c -- bracketing
c
       L = NOP - 6
       GOTO (70,80,90),  L
c
c    KSJRUN = 7; specified V and U
70     Z = RW(IoU)
       GOTO 100
c
c    KSJRUN = 8; specified V and H
80     Z = RW(IoH)
       GOTO 100
c
c    KSJRUN = 9; specified V and S
90     Z = RW(IoS)
       GOTO 100
c
100    IF (Z.LT.ZTAR)  THEN
c            check trial vs lower bound
               IF (Z.GE.ZB)  THEN
c                derivative estimate on S curve
                   IF ((KCI.EQ.1).OR.(KCI.EQ.3))  THEN
                       TMTB = RW(IoT) - TB
                       IF ((TMTB.GT.ZERO).AND.(Z.GT.ZB))  THEN
                               DZDTS = (Z - ZB)/TMTB
                           ELSE
                               DZDTS = 1.0D0/RW(IoHUGE)
                           ENDIF
                       ENDIF
c                reset lower bound
                   TB = RW(IoT)
                   PB = RW(IoP)
                   ZB = Z
                   L = 1
                   CALL SJRKCI(KCI,L)
                   ENDIF
           ELSE
c            check trial vs upper bound
               IF (Z.LE.ZA)  THEN
c                derivative estimate on S curve
                   IF (KCI.GT.1)  THEN
                       TAMT = TA - RW(IoT)
                       IF ((TAMT.GT.ZERO).AND.(Z.LT.ZA)) THEN
                                DZDTS = (ZA - Z)/TAMT
                           ELSE
                                DZDTS = 1.0D0/RW(IoHUGE)
                           ENDIF
                       ENDIF
c                reset upper bound
                   TA = RW(IoT)
                   PA = RW(IoP)
                   ZA = Z
                   L = 2
                   CALL SJRKCI(KCI,L)
                   ENDIF
           ENDIF
c
c -- select next step
c
       IF (KTRS.EQ.1)  THEN
c            arbitrary step
               DT = 1.0D0
               DPDT = 0.0D0
           ELSE
c            step based on d()/dT
               IF (KCI.NE.3)  THEN
c                    solution not yet bracketed (w/KTRS inflation)
                       DZDT = (Z - ZT)/DT
                       DPDT = (RW(IoP) - PT)/DT
                       DT = KTRS*(ZTAR - Z)/DZDT
                   ELSE
c                    solution bracketed
                       DTAB = TA - TB
                       DZDT = (ZA - ZB)/DTAB
                       DPDT = (PA - PB)/DTAB
                       IF (DZDT.GT.2*DZDTS)  THEN
c                            S-shaped curve; cut bracket in half
                               DT = 0.5D0*(TA + TB) - RW(IoT)
                           ELSE
c                            gradual curve; use derivative
                               DT = (ZTAR - Z)/DZDT
                           ENDIF
                   ENDIF
           ENDIF
c
c    estimate associated dP
       DP = DPDT*DT
c
c -- limit the magnitudes of changes
c
       DTM = 0.1D0*RW(IoT)
       IF (ABS(DT).GT.DTM)  THEN
           FAC = DTM/ABS(DT)
           DT = FAC*DT
           DP = FAC*DP
           ENDIF
       DP = DPDT*DT
       IF (DP.LT.ZERO)  THEN
              DPM = 0.5D0*RW(IoP)
           ELSE
              DPM = 2.0D0*RW(IoP)
           ENDIF
       IF (ABS(DP).GT.DPM)  THEN
           FAC = DPM/ABS(DP)
           DP = FAC*DP
           ENDIF
c
c -- save values
c
       TT = RW(IoT)
       PT = RW(IoP)
       ZT = Z
c
c -- make the changes
c
       RW(IoT) = RW(IoT) + DT
       RW(IoP) = RW(IoP) + DP
c
c    monitor
       IF (KMON.GT.0)  WRITE (KUMO,310) DT,DP
310    FORMAT (' Outer loop adjustments; dT,dP =',2(1PE15.7))
c
       GOTO 20
c
c -- signal end to monitor printing
c
900    IF (KMON.GT.0)  WRITE (KUMO,902)
902    FORMAT (//' End of run monitor output'//)
c
       RETURN
c
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJROPX(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                   ATOM,CHEM,IW,RW,SW,MODE)
c
c      Calculates equilibrium properties for a specified pair of
c      state variables, one of which is T or P, as set by MODE:
c-----------------------------------------------------------------------
c      MODE values:
c          1   Specified T and P
c          2   Specified T and V
c          3   Specified T and S
c          4   Specified P and V
c          5   Specified P and H
c          6   Specified P and S
c          7   Specified V and U  (called only with single phase)
c          8   Specified V and H  (called only with single phase)
c          9   Specified V and S  (called only with single phase)
c-----------------------------------------------------------------------
c      If successful, KERR is returned as zero.
c
c      If problems are encountered:
c          KERR = 1 program problems; refer to user for optional
c                    rerun with diagnostic output.
c                 2 convergence failure.
c                 3 impossible populations; refer to user for new input.
c-----------------------------------------------------------------------
c      Single-parameter adjustments:
c
c          When iterating for target properties, uses a bracketing
c          scheme to avoid problems with sharply S-shaped curves.
c          This permits the final solution to be a mixture of two
c          systems in equilibrium with one another, neither of which
c          could satisfy the input properties (much as in a two-phase
c          system). The variable being adjusted is Z, which is defined
c          to be monotone increasing in the iteration variable F, which
c          is either T or P, the other being specified.
c
c          The calculation is converged when the upper and lower bounds
c          become the same (and hence have the same T and P).   The
c          extensive properties of the two bounding solutions are then
c          averaged to produce a system with the target property. This
c          allows multi-phase systems to be handled properly.
c
c      Two-parameter adjustments:
c
c          A two-dimensional Newton-Raphson adjustment, with limited
c          changes, is used.  In multi-phase systems, if V is one of
c          the variables, the problem is handled as a series of
c          single-parameter adjustments by the calling program SJROP.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array IR
c @        NSW         dimension of work array IS
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c          SW(I)       REAL*4 work array
c @        MODE        run control; see above
c
c      Variables in the integer work array IW:
c @        KFRZ      0 not frozen or constrained
c                    1 constrained equilibrium, only specified
c                      total mols
c                    2 constrained equilibrium, general linear
c                      constraints
c                   -1 composition frozen, phases at same T
c                   -2 composition frozen, phases at different T
c #        KERR        error flag
c @        KMON        monitor control
c #        KTRE        total iterations for an SJEQLB call
c #        KTRI        total (T,P) adjustments for an SJRUN call
c #        KTRT        total SJEQLB iterations for an SJRUN call
c @        KUMO        unit for monitor output
c @        NP          number of phases
c @        NS          number of species
c
c      Variables in the real work array RW:
c @        HUGE        large machine number
c -        SMFA(J)     SMOL at FA
c -        SMFB(J)     SMOL at FB
c #        SMOL(J)     mols of Jth species
c #        H           mixture enthalpy, J/kg
c #        S           mixture entropy, J/kg-K
c #        U           mixture internal energy, J/kg
c #        V           mixture specific volume, m**3/kg
c @        P           pressure, Pa
c @        T           temperature, K
c
c      Variables used only internally:
c          DADP        rate of change of first variable with P
c          DBDP        rate of change of second variable with P
c          DADT        rate of change of first variable with T
c          DADT        rate of change of first variable with T
c          DP          change in P
c          DT          change in T
c          DPM         maximum allowable change in P
c          DTM         maximum allowable change in T
c          DZDF        dZ/dF
c          DZDFS       small dZ/dF of proper sign
c          ERRFP       pressure error factor
c          ERRFT       pressure error factor
c          F           adjusting variable in single-parameter
c                      adjustments
c          FA          F bound above
c          FB          F bound below
c          HSI         specified enthalpy, J/kg
c          HT          trial-state enthalpy, J/kg
c          KCI         bracket state; 0 none, 1 below, 2 above,
c                                     3 above/below
c          KTRS1       iteration counter
c          KTRS1M      maximum allowed iterations
c          KINITD      initialization control for perturbations
c          KINITT      initialization control for trials
c          PT          trial-state pressure, Pa
c          QA          mass fraction of system in bracket state A
c          QB          mass fraction of system in bracket state B
c          SSI         specified entropy, J/kg-K
c          ST          trial-state entropy, J/kg-K
c          USI         specified internal energy, J/kg
c          UT          trial-state internal energy, J/kg
c          VSI         specified specific volume, m**3/kg
c          VT          trial-state specific volume, m**3/kg
c          Z           target variable in single-parameter adjustments
c                       (a monotone increasing function of F)
c          ZA          Z at FA
c          ZB          Z at FB
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCCC       REAL*4     SW
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NSMAX),IW(NIW),RW(NRW),SW(NSW),
     ;    IEPTR(80),IRPTR(20),ISPTR(10),ITPTR(20)
c-----------------------------------------------------------------------
       COMMON /SJEPTR/ IEPTR
       COMMON /SJRPTR/ IRPTR
       COMMON /SJSPTR/ ISPTR
       COMMON /SJTPTR/ ITPTR
       EQUIVALENCE (IoJFS,ISPTR(6)),(IoKERR,IEPTR(1)),
     ;   (IoKFRZ,ITPTR(1)),(IoKMON,IEPTR(2)),(IoKTRE,IEPTR(3)),
     ;   (IoKUMO,IEPTR(4)),(IoKTRI,IRPTR(4)),(IoKTRT,IRPTR(5)),
     ;   (IoNP,IEPTR(7)),(IoNS,IEPTR(8)),(IoHUGE,IEPTR(32)),
     ;   (IoP,ITPTR(5)),(IoT,ITPTR(6)),(IoH,ITPTR(7)),
     ;   (IoS,ITPTR(8)),(IoU,ITPTR(9)),(IoV,ITPTR(10)),
     ;   (IoSMOL,IEPTR(74)),(IoSMFA,IRPTR(18)),(IoSMFB,IRPTR(19))
c-----------------------------------------------------------------------
c    Maximum iterations
       DATA    KTRS1M/50/
C
c    Error factors in computation precision
C
C*****precision > single
C      ZERO = 0.0E0
C      ERRFP = 1.E-5
C      ERRFT = 1.E-8
C*****END precision > single
C*****precision > double
      ZERO = 0.0D0
      ERRFP = 1.D-5
      ERRFT = 1.D-8
C*****END precision > double
C
c-----------------------------------------------------------------------
c     get parameters
        NP = IW(IoNP)
        NS = IW(IoNS)
        KFRZ = IW(IoKFRZ)
        KUMO = IW(IoKUMO)
        KMON = IW(IoKMON)
c
c    set controls
       IF (KFRZ.GE.0)  THEN
c            equilibrium calculation
               KINITT = 1
               IF (NP.EQ.1) THEN
                       KINITD = 3
                   ELSE
                       KINITD = 1
                   ENDIF
           ELSE
c            frozen composition; no SJEQLB calls
               KINITT = 4
               KINITD = 4
           ENDIF
c
c    other initializations
       FA = RW(IoHUGE)
       FB = - RW(IoHUGE)
       ZA = RW(IoHUGE)
       ZB = - RW(IoHUGE)
       DZDFS = 1.0D0/RW(IoHUGE)
       KCI = 0
       KTRS1 = 0
       DT = RW(IoHUGE)
       DP = RW(IoHUGE)
c
c --  save targets and set up for adjustments
c
       GOTO (20,2,3,4,5,6,7,8,9), MODE
       GOTO 900
c
c    MODE = 2: specified T and V
2      ZTAR = - RW(IoV)
       VSI = RW(IoV)
       GOTO 20
c
c    MODE = 3: specified T and S
3      ZTAR =  - RW(IoS)
       SSI = RW(IoS)
       GOTO 20
c
c    MODE = 4: specified P and V
4      ZTAR = RW(IoV)
       VSI = RW(IoV)
       GOTO 20
c
c    MODE = 5: specified P and H
5      ZTAR = RW(IoH)
       HSI = RW(IoH)
       GOTO 20
c
c    MODE = 6: specified P and S
6      ZTAR = RW(IoS)
       SSI = RW(IoS)
       GOTO 20
c
c    MODE = 7; specified V and U
7      ATAR = RW(IoV)
       BTAR = RW(IoU)
       VSI = RW(IoV)
       USI = RW(IoU)
       GOTO 20
c
c    MODE = 8; specified V and H
8      ATAR = RW(IoV)
       BTAR = RW(IoH)
       VSI = RW(IoV)
       HSI = RW(IoH)
       GOTO 20
c
c    MODE = 9; specified V and S
9      ATAR = RW(IoV)
       BTAR = RW(IoS)
       VSI = RW(IoV)
       SSI = RW(IoS)
       GOTO 20
c
c -- calculate the properties for the current trial (T,P)
c
c    skip initialization if T and P are close and one phase
20     IF ((ABS(DT).LT.0.01D0*RW(IoT)).AND.
     ;     (ABS(DP).LT.0.01D0*RW(IoP))) THEN
           IF ((NP.EQ.1).AND.(KFRZ.GE.0))  KINITT = 2
c        set to skip partial derivative re-estimation
           KINITD = 0
           ENDIF
c
c    mixture calculation
       CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,KINITT)
       IF (IW(IoKERR).NE.0)  RETURN
c
c    augment pass tally
       IW(IoKTRT) = IW(IoKTRT) + IW(IoKTRE)
       KTRS1 = KTRS1 + 1
c
c    check for no iterations
       IF (MODE.EQ.1)  GOTO 800
c
c    monitor
       IF (KMON.GT.0)  THEN
           WRITE (KUMO,21) MODE,KTRS1,
     ;        RW(IoT),RW(IoP),RW(IoV),RW(IoU),RW(IoH),RW(IoS)
21         FORMAT (/' State iteration: mode',I2,'  trial',I4,
     ;       '   T, P =',2(1PE15.7)/
     ;       ' V, U, H, S =',4(E15.7))
           ENDIF
c
c    save trial state
       TT = RW(IoT)
       PT = RW(IoP)
       VT = RW(IoV)
       UT = RW(IoU)
       HT = RW(IoH)
       ST = RW(IoS)
c
c    reset controls if frozen
       IF (KFRZ.LT.0)  KINITT = 4
c
c    branch on mode
       GOTO (900,22,23,24,25,26,27,28,29), MODE
       GOTO 900
c
c --  bracketing (required for multi-phase problems)
c
c    MODE = 2: specified T and V
22     Z = - RW(IoV)
       F = RW(IoP)
       GOTO 30
c
c    MODE = 3: specified T and S
23     Z =  - RW(IoS)
       F = RW(IoP)
       GOTO 30
c
c    MODE = 4: specified P and V
24     Z = RW(IoV)
       F = RW(IoT)
       GOTO 30
c
c    MODE = 5: specified P and H
25     Z = RW(IoH)
       F = RW(IoT)
       GOTO 30
c
c    MODE = 6: specified P and S
26     Z = RW(IoS)
       F = RW(IoT)
       GOTO 30
c
c    MODE = 7; specified V and U
27     A = RW(IoV)
       B = RW(IoU)
       GOTO 40
c
c    MODE = 8; specified V and H
28     A = RW(IoV)
       B = RW(IoH)
       GOTO 40
c
c    MODE = 9; specified V and S
29     A = RW(IoV)
       B = RW(IoS)
       GOTO 40
c
c    check bracketing (one-parameter adjustments only)
c
30     IF (Z.LT.ZTAR)  THEN
c            check trial vs lower bound
               IF (Z.GE.ZB)  THEN
c                derivative estimate on S curve
                   IF ((KCI.EQ.1).OR.(KCI.EQ.3))  THEN
                       FMFB = F - FB
                       IF ((FMFB.GT.ZERO).AND.(Z.GT.ZB))  THEN
                               DZDFS = (Z - ZB)/FMFB
                           ELSE
                               DZDFS = 1.0D0/RW(IoHUGE)
                           ENDIF
                       ENDIF
c                reset lower bound
                   FB = F
                   ZB = Z
                   DO 31 J=1,NS
                       RW(IoSMFB+J) = RW(IoSMOL+J)
31                     CONTINUE
                   L = 1
                   CALL SJRKCI(KCI,L)
                   ENDIF
           ELSE
c            check trial vs upper bound
               IF (Z.LE.ZA)  THEN
c                derivative estimate on S curve
                   IF (KCI.GT.1)  THEN
                       FAMF = FA - F
                       IF ((FAMF.GT.ZERO).AND.(Z.LT.ZA)) THEN
                               DZDFS = (ZA - Z)/FAMF
                           ELSE
                               DZDFS = 1.0D0/RW(IoHUGE)
                           ENDIF
                       ENDIF
c                reset upper bound
                   FA = F
                   ZA = Z
                   DO 33 J=1,NS
                       RW(IoSMFA+J) = RW(IoSMOL+J)
33                     CONTINUE
                   L = 2
                   CALL SJRKCI(KCI,L)
                   ENDIF
           ENDIF
c
c -- test convergence
c
40     IF ((ABS(DT).LE.ERRFT*RW(IoT)).
     ;    AND.(ABS(DP).LE.ERRFP*RW(IoP))) GOTO 600
c
c -- not converged
c
c    check iteration count
       IF (KTRS1.GT.KTRS1M)  THEN
           IW(IoKERR) = 2
           RETURN
           ENDIF
c
c    check for two-parameter adjustments
       IF (MODE.GT.6)  GOTO 50

c -- single parameter adjustments
c
c    check for first iteration
       IF (KTRS1.EQ.1)  THEN
           FP = F
           ZP = Z
           DF = 0.01D0*F
           GOTO 48
           ENDIF
c
c    step based on d()/dT determined by consecutive iterations
       IF (KCI.NE.3)  THEN
c            solution not yet bracketed
               FMFP = F - FP
               ZMZP = Z - ZP
               IF ((FMFP.EQ.ZERO).OR.(ZMZP.EQ.ZERO))  THEN
c                    set small value of correct sign
                       DZDF = 1.0D0/RW(IoHUGE)
                   ELSE
c                    estimate derivative
                       DZDF = ZMZP/FMFP
                   ENDIF
               FP = F
               ZP = Z
           ELSE
               ZAB = ZA - ZB
               TERM = ABS(ZA+ZB)
               CALL SJURND(ZAB,TERM)
               IF (ZAB.EQ.ZERO)  GOTO 800
               FAMFB = FA - FB
               TERM = ABS(FA+FB)
               CALL SJURND(FAMFB,TERM)
               IF (FAMFB.EQ.ZERO)  GOTO 600
               DZDF = ZAB/FAMFB
           ENDIF

c    select change
       IF ((KCI.EQ.3).AND.(DZDF.GT.2*DZDFS))  THEN
c            S-shaped curve; cut bracket in half
               DF = 0.5D0*(FA + FB) - F
           ELSE
c            gradual curve; set change using derivative estimate
               DF = (ZTAR - Z)/DZDF
c            increase step if having difficulty bracketing
               IF (KCI.NE.3)  DF = DF*KTRS1
           ENDIF
c
c   mode branch
48     IF (MODE.LT.4)  THEN
c            changing P
               DP = DF
               DT = 0.0D0
           ELSE
c           changing T
               DT = DF
               DP = 0.0D0
           ENDIF
c
c  check change checks
       IF (KCI.EQ.3)  THEN
               GOTO 300
           ELSE
               GOTO 200
           ENDIF
c
c  -- two-parameter adjustments
c
c    check for derivative estimation
50     IF (KINITD.NE.0)  THEN
c
c        temperature perturbation
           DT = 0.01D0*TT
           RW(IoT) = TT + DT
           RW(IoP) = PT
           CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;                ATOM,CHEM,IW,RW,SW,KINITD)
           IF (IW(IoKERR).NE.0)  RETURN
c
c        augment pass tally
           IW(IoKTRT) = IW(IoKTRT) + IW(IoKTRE)
c
c        calculate derivatives
           L = MODE - 6
           GOTO (57,58,59), L
c
c        MODE = 7; specified V and U
57         DADT = (RW(IoV) - VT)/DT
           DBDT = (RW(IoU) - UT)/DT
           GOTO 70
c
c        MODE = 8; specified V and H
58         DADT = (RW(IoV) - VT)/DT
           DBDT = (RW(IoH) - HT)/DT
           GOTO 70
c
c        MODE = 9; specified V and S
59         DADT = (RW(IoV) - VT)/DT
           DBDT = (RW(IoS) - ST)/DT
           GOTO 70
c
c        pressure perturbation
70         DP = 0.01D0*PT
           RW(IoP) = PT + DP
           RW(IoT) = TT
           CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,KINITD)
           IF (IW(IoKERR).NE.0)  RETURN
c
c        augment pass tally
           IW(IoKTRT) = IW(IoKTRT) + IW(IoKTRE)
c
c        calculate derivatives
           L = MODE - 6
           GOTO (77,78,79), L
c
c        MODE = 7; specified V and U
77         DADP = (RW(IoV) - VT)/DP
           DBDP = (RW(IoU) - UT)/DP
           GOTO 90
c
c        MODE = 8; specified V and H
78         DADP = (RW(IoV) - VT)/DP
           DBDP = (RW(IoH) - HT)/DP
           GOTO 90
c
c        MODE = 9; specified V and S
79         DADP = (RW(IoV) - VT)/DP
           DBDP = (RW(IoS) - ST)/DP
           GOTO 90
c
           ENDIF
c
c    change determination
90     DET = DADT*DBDP - DBDT*DADP
       IF (DET.EQ.ZERO)  THEN
           IW(IoKERR) = 1
           RETURN
           ENDIF
       DA = ATAR - A
       DB = BTAR - B
       DT = (DA*DBDP - DB*DADP)/DET
       DP = (DADT*DB - DBDT*DA)/DET
c
c -- limit the magnitudes of changes
c
200    DTM = 0.2D0*TT
       ABSDT = ABS(DT)
       FAC =1.0D0
       IF (ABSDT.GT.DTM) FAC = DTM/ABSDT
       DT = FAC*DT
       DP = FAC*DP

       DPM = 0.2D0*PT
       ABSDP = ABS(DP)
       FAC =1.0D0
       IF (ABSDP.GT.DPM) FAC = DPM/ABSDP
       DT = FAC*DT
       DP = FAC*DP
c
c -- make the changes
c
300    RW(IoT) = TT + DT
       RW(IoP) = PT + DP
c
c    monitor
       IF (KMON.GT.0)  WRITE (KUMO,310) DT,DP
310    FORMAT (' Inner loop adjustments:  dT,dP =',2(1PE15.7))
c
c    increment count
       IW(IoKTRI) = IW(IoKTRI) + 1
       GOTO 20
c
c -- converged; set the properties for output
c
c    check for bracketed averaging
600    IF (KMON.GT.0)  WRITE (KUMO,602) MODE
602    FORMAT (' Mode ',i2,' iteration converged.')
       IF (KCI.EQ.3) THEN
           QA = (ZTAR - ZB)/ZAB
           QB = 1.0D0 - QA
           DO 609 J=1,NS
               RW(IoSMOL+J) = QA*RW(IoSMFA+J) + QB*RW(IoSMFB+J)
609            CONTINUE
c        get properties for this system
           KINIT = 4
           CALL SJTP(NAMAX,NSMAX,NIW,NRW,NSW,ATOM,CHEM,IW,RW,SW,KINIT)
           ENDIF
c
c -- run completed
c
800    RETURN
c
c -- program error exit
c
900    IF (KMON.GT.0)  WRITE (KUMO,902)
902    FORMAT (' Program error in SJRUNX')
       IW(IoKERR) = 1
       RETURN
c
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJRPTS(NAMAX,NPMAX,NSMAX,NIW,NRW,NSW,RW,KU)
c
c      Sets SJROP pointers and checks specified work array dimensions.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in argument list
c
c          NAMAX       maximum number of atom types
c          NPMAX       maximum number of phases
c          NSMAX       maximum number of species
c          NIW         dimension of work array IW
c          NRW         dimension of work array RW
c          RW(I)       REAL work array
c          KU          output unit for error message
c-----------------------------------------------------------------------
c    Targets:
c      NW = max{2*NA,NA+NP}
c      NIW = 17 + 14*NA + 4*NP + 5*NS + NA*NS
c      NRW = 24 + 16*NA + 12*NA*NA + 3*NA*NP + 6*NP +
c    ;      18*NS + NW*NW + NW
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   RW(NRW)
c-----------------------------------------------------------------------
c  20 additional pointers required by SJROP
       COMMON/ SJRPTR/
     ;   IoKDET,IoKRCT,IoKSND,IoKTRI,IoKTRT,IoI2,IoI3,IoNCON,IoTE,IoC,
     ;   IoCDET,IoH1,IoP1,IoT1,IoV1,IoR5,IoR6,IoSMFA,IoSMFB,IoSMOZ
c-----------------------------------------------------------------------
c   set pointers required by SJTP and SJEQLB
       NIWZ = NIW - 8
       NRWZ = NRW - 9 - 3*NSMAX
       CALL SJTPTS(NAMAX,NPMAX,NSMAX,NIWZ,NRWZ,NSW,RW,KU)
c  ** IW pointers
       IoKDET = NIWZ + 1
       IoKRCT = IoKDET + 1
       IoKSND = IoKRCT + 1
       IoKTRI = IoKSND + 1
       IoKTRT = IoKTRI + 1
       IoI2 = IoKTRT + 1
       IoI3 = IoI2 + 1
       IoNCON = IoI3 + 1
       NIWX = IoNCON
c    check
       IF (NIWX.NE.NIW) THEN
           WRITE (KU,1) NAMAX,NPMAX,NSMAX
1          FORMAT (/' SJROP dimensioning error for NAMAX =',I3,
     ;              '  NPMAX =',I3,'  NSMAX =',I3)
           WRITE (KU,2) NIWX
2          FORMAT (/'  NIWORK error: NIWX = ',I6)
           STOP
           ENDIF
c  ** RW pointers
       IoTE = NRWZ + 1
       IoC = IoTE + 1
       IoCDET = IoC + 1
       IoH1 = IoCDET + 1
       IoP1 = IoH1 + 1
       IoT1 = IoP1 + 1
       IoV1 = IoT1 + 1
       IoR5 = IoV1 + 1
       IoR6 = IoR5 + 1
       IoSMOZ = IoR6
       IoSMFA = IoSMOZ + NSMAX
       IoSMFB = IoSMFA + NSMAX
       NRWX = IoSMFB + NSMAX
c    check
       IF (NRWX.NE.NRW) THEN
           WRITE (KU,1) NAMAX,NPMAX,NSMAX
           WRITE (KU,4) NRWX
4          FORMAT (/'  NRWORK should be = ',I6)
           STOP
           ENDIF
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJSPTS(NAMAX,NPMAX,NSMAX,NIW,NRW,NSW,RW,KU)
c
c      Sets SJSET pointers and checks specified work array dimensions.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in argument list
c
c          NAMAX       maximum number of atom types
c          NPMAX       maximum number of phases
c          NSMAX       maximum number of species
c          NIW         dimension of work array IW
c          NRW         dimension of work array RW
c          NSW         dimension of work array SW
c          RW(I)       REAL work array
c          KU          output unit for error message
c-----------------------------------------------------------------------
c    Targets:
c      NW = max{2*NA,NA+NP}
c      NIW = 22 + 14*NA + 4*NP + 8*NS + 2*NA*NS
c      NRW = 24 + 16*NA + 12*NA*NA + 3*NP*NA + 6*NP +
c    ;       18*NS + NW*NW + NW
c      NSW = 123*NS
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   RW(NRW)
c-----------------------------------------------------------------------
c  10 additional pointers required by SJSET
       COMMON /SJSPTR/
     ;   IoKNFS,IoKSME,IoKUFL,IoNAF,IoNSF,IoJFS,IoNF,LoNF,IoITHL,IoITHM
c-----------------------------------------------------------------------
c   set pointers required by SJROP, SJTP and SJEQLB
       NIWZ = NIW - 5 - 3*NSMAX - NAMAX*NSMAX
       NRWZ = NRW
       CALL SJRPTS(NAMAX,NPMAX,NSMAX,NIWZ,NRWZ,NSW,RW,KU)
       IoKNFS = NIWZ + 1
       IoKSME = IoKNFS + 1
       IoKUFL = IoKSME + 1
       IoNAF = IoKUFL + 1
       IoNSF = IoNAF + 1
       IoJFS = IoNSF
       LoNF = NAMAX
       IoNF = IoJFS + NSMAX - LoNF
       IoITHL = IoNF + NAMAX*NSMAX + LoNF
       IoITHM = IoITHL + NSMAX
       NIWX = IoITHM + NSMAX
c    check
       IF (NIWX.NE.NIW) THEN
           WRITE (KU,1) NAMAX,NPMAX,NSMAX
1          FORMAT (/' SJSET dimensioning error for NAMAX =',I3,
     ;              '  NPMAX =',I3,'  NSMAX =',I3)
           WRITE (KU,2) NIWX
2          FORMAT (/'  NIWORK should be',I6)
           STOP
           ENDIF
c
c    SW pointers are not explicit
CCC       NSWX = 123*NSMAX
CCC       IF (NSWX.NE.NSW)  THEN
CCC        WRITE (KU,1) NAMAX,NPMAX,NSMAX
CCC        WRITE (KU,4) NSWX
CCC4       FORMAT (/'  NSWORK should be ',I6)
CCC        STOP
CCC        ENDIF
c
       RETURN
       END
c-----------------------------------------------------------------------
c
       SUBROUTINE SJTGHS(CHEMJ,T,HMH0J,S0J)
c
c      Gets user input of temperature-dependent properties for non-file
c      species CHEM
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        CHEMJ   CHARACTER*16 chemical name of the species
c @        T       temperature, K
c #        HMH0J   enthalpy above 298.15K, kcal/mol
c #        S0J     entropy at 1 atm, cal/mol
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
       CHARACTER*16 CHEMJ
c-----------------------------------------------------------------------
       COMMON /SJTERM/ KTERM,KUTERM
c-----------------------------------------------------------------------
c    constant in machine precision
       ZERO = 0.0D0
c
c    header
       WRITE (KUTERM,4)  T,CHEMJ
4      FORMAT (/' Enter JANNAF table data at ',F7.1,'K, 1 atm. for ',A)
c
c    get the enthalpy
10     WRITE (KUTERM,11)
11     FORMAT (/'   enthalpy above 298.15K (kcal/mol)?  ')
       READ (KUTERM,*,ERR=18) HMH0J
       GOTO 20
18     CALL SJDERR
       GOTO 10
c
c    get the entropy
20     WRITE (KUTERM,21)
21     FORMAT ('   entropy (cal/mol-K)?  ')
       READ (KUTERM,*,ERR=28) S0J
       IF (S0J.LE.ZERO)  THEN
           WRITE (KUTERM,14)
14         FORMAT (' The entropy must be positive.'/)
           GOTO 20
           ENDIF
       RETURN
28     CALL SJDERR
       GOTO 20
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJTP(NAMAX,NSMAX,NIW,NRW,NSW,
     ;         ATOM,CHEM,IW,RW,SW,KINIT)
c
c      Calculates  properties at given T,P.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NAMAX       maximum number of atoms
c @        NSMAX       maximum number of species
c @        NIW         dimension  work array IW
c @        NRW         dimension of work array IR
c @        NSW         dimension of work array IS
c @        ATOM(I)     CHARACTER*16 name of Ith atom
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c          SW(I)       REAL*4 work array
c @        KINIT       initialization control
c                     1 full initialization
c                     2 start with the last values if T,P are
c                       sufficiently close to their previous values
c                     3 start with the last values
c                     4 use mols and element potentials but recalculate
c                         mol fractions (final mixed phase calculation)
c
c      Variables in the integer work array IW:
c #        IB(K)       Kth independent atom
c #        KERR        error flag
c @        KFRZ      0 not frozen or constrained
c                    1 constrained equilibrium, only specified
c                      total mols
c                    2 constrained equilibrium, general linear
c                      constraints
c                   -1 composition frozen, phases at same T
c                   -2 composition frozen, phases at different T
c #        KTRE        SJEQLB pass counter
c @        NA          number of atom types
c #        NB          number of bases
c @        NP          number of phases
c @        NS          number of species
c @        NSP(M)      number of species in phase M
c
c      Variables in the real work array RW:
c @        CVCJ        Joules/cal
c @        DCS(J)      density of the Jth species, g/cm*3  (0 for gas)
c @        DHF0(J)     enthalpy of form. of Jth species at 298.15K,
c                      kcal/m
c #        ELAM(K)     element potential for kth independent atom
c #        H           mixture enthalpy, J/kg
c @        P           pressure, PA
c #        PATM        Pa/atm
c #        PMOL(M)     mols of the Mth phase
c #        S           mixture entropy, J/kg-K
c #        SMOL(J)     mols of Jth species
c @        T           temperature, K
c @        TP(M)       temperature of Mth phase, K
c #        U           mixture internal energy, J/kg
c #        V           mixture volume, m**3/kg
c #        WM          mixture molal mass, g/mol
c @        WMS(J)      molal mass of Jth species, g/mol
c #        X(J)        phase mol fraction of jth species
c #        XM(J)       mixture mol fraction of jth species
c #        YM(J)       mixture mass fraction of the Jth species
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCC       REAL*4          SW
       CHARACTER*16 ATOM, CHEM
c-----------------------------------------------------------------------
       DIMENSION   ATOM(NAMAX),CHEM(NAMAX),IW(NIW),RW(NRW),SW(NSW),
     ;   IEPTR(80)
c-----------------------------------------------------------------------
       COMMON /SJEPTR/ IEPTR
       COMMON /SJTPTR/
     ;   IoKFRZ,IoCVCJ,IoPATM,IoRGAS,IoP,IoT,IoH,IoS,IoU,IoV,
     ;   IoWM,IoTP,IoDCS,IoDHF0,IoWMS,IoHMH0,IoS0,IoWMP,IoXM,IoYM
       EQUIVALENCE (IoIB,IEPTR(9)),(IoKERR,IEPTR(1)),
     ;   (IoKTRE,IEPTR(3)),(IoKUMO,IEPTR(4)),(IoNA,IEPTR(5)),
     ;   (IoNB,IEPTR(6)),(IoNP,IEPTR(7)),(IoNS,IEPTR(8)),
     ;   (IoNSP,IEPTR(30)),(IoELAM,IEPTR(39)),(IoHUGE,IEPTR(32)),
     ;   (IoPMOL,IEPTR(62)),(IoSMOL,IEPTR(74)),(IoX,IEPTR(77))
       common /kcallblock/ kcall
c-----------------------------------------------------------------------
       IF (KCALL.EQ.0)  THEN
           TOLD = 0D0
           POLD = 0D0
           DTMAX = 50D0
           PRMAX = .5D0
           KCALL = 1
           ENDIF
       ZERO = 0.0D0
       TINY = 1.0D0/RW(IoHUGE)
c
c    get parameters
       NA = IW(IoNA)
       NP = IW(IoNP)
       NS = IW(IoNS)
c
c   set counter
      IW(IoKTRE) = 0
c
c -- compute constants
c
       RLP = RW(IoRGAS)*LOG(RW(IoP)/RW(IoPATM))
c
c -- get species properties
c
c    check for different phase temperatures
       L = - 2
       IF (IW(IoKFRZ).EQ.L)  THEN
c            different temperatures; set a non-displayable temperature
               RW(IoT) = RW(IoHUGE)
           ELSE
c            set phase temperatures = T
               DO 3 M=1,NP
                   RW(IoTP+M) = RW(IoT)
3                  CONTINUE
           ENDIF
c
       CALL SJTPRP(NSMAX,NIW,NRW,NSW,CHEM,IW,RW,SW)
c
c -- check for frozen composition
c
       IF (IW(IoKFRZ).LT.0) THEN
c        frozen; get phase mols and mol fractions
           CALL SJIPMX(NSMAX,NIW,NRW,CHEM,IW,RW)
           NB = NA
           IW(IoNB) = NB
           DO 9 K=1,NB
c            set basis atoms
               IW(IoIB+K) = K
c            set to show stars at printout
               RW(IoELAM+K) = RW(IoHUGE)
9              CONTINUE
           GOTO 60
           ENDIF
c
c  -- check for final mixed-phase calculation
c
       IF (KINIT.EQ.4)  THEN
c        recalculate phase mols and mol fractions from mols
           CALL SJIPMX(NSMAX,NIW,NRW,CHEM,IW,RW)
           GOTO 60
           ENDIF
c
c -- equilibrium computation
c
       GOTO (10,20,30)  KINIT
c
c    set to initialize equilibrium solution
10     KIN = 1
       GOTO 50
c
c   decide if close enough to skip initialization
20     IF (ABS(RW(IoT)-TOLD).GT.DTMAX)  GOTO 10
       IF (ABS((RW(IoP)-POLD)/RW(IoP)).GT.PRMAX)  GOTO 10
c
c    initialize with the last phase mols and mol fractions
30     K = 1
c    estimate element potentials
       CALL SJIESL(NSMAX,NIW,NRW,CHEM,IW,RW,K,DUMMY,IDUMMY)
c    rare species problem probable if error
       IF (IW(IoKERR).NE.0)  THEN
           IW(IoKERR) = 4
           RETURN
           ENDIF
       KIN = 0
c
c    make the equilibrium calculation
c
50    continue
       CALL SJEQLB(NAMAX,NSMAX,NIW,NRW,ATOM,CHEM,IW,RW,KIN)
      IF (IW(IoKERR).NE.0)  RETURN
c
c -- compute the mols, mixture mass fractions and mol fractions
c      and the molal masses of each phase
c
60     TMASS = 0
       TMOL = 0
       J2 = 0
       DO 63 M=1,NP
           J1 = J2 + 1
           J2 = J2 + IW(IoNSP+M)
           PMASS = 0
           DO 61 J=J1,J2
               RW(IoSMOL+J) = RW(IoX+J)*RW(IoPMOL+M)
               RW(IoYM+J) = RW(IoSMOL+J)*RW(IoWMS+J)
               PMASS = PMASS + RW(IoYM+J)
61             CONTINUE
           TMOL = TMOL + RW(IoPMOL+M)
           TMASS = TMASS + PMASS
           IF (RW(IoPMOL+M).GT.ZERO)  THEN
                   RW(IoWMP+M) = PMASS/RW(IoPMOL+M)
               ELSE
                   RW(IoWMP+M) = 0
               ENDIF
63         CONTINUE
       IF ((TMASS.LE.ZERO).OR.(TMOL.LE.ZERO))  THEN
           IW(IoKERR) = 1
           RETURN
           ENDIF
       DO 65 J=1,NS
           RW(IoYM+J) = RW(IoYM+J)/TMASS
           RW(IoXM+J) = RW(IoSMOL+J)/TMOL
65         CONTINUE
       RW(IoWM) = TMASS/TMOL
c
c -- mixture property calculation
c
       RW(IoH) = 0.0D0
       RW(IoS) = 0.0D0
       RW(IoV) = 0.0D0
       J2 = 0
       DO 79 M=1,NP
           J1 = J2 + 1
           J2 = J2 + IW(IoNSP+M)
           DO 77 J=J1,J2
c            get H and S at T and 1 atm
               HX = RW(IoHMH0+J) + RW(IoDHF0+J)
               SX = RW(IoS0+J)
c            correct entropy for mol fraction
               IF (RW(IoX+J).GT.TINY)
     ;           SX = SX - RW(IoRGAS)*LOG(RW(IoX+J))
c            check species type
               IF (RW(IoDCS+J).EQ.ZERO)  THEN
c                    gas; correct entropy for pressure
                       SX = SX - RLP
c                    calculate partial molal volume,  m**3/mol
                       VX = RW(IoRGAS)*RW(IoCVCJ)*RW(IoTP+M)/RW(IoP)
                   ELSE
c                    condensed species; correct enthalpy for pressure
                      HX = HX +
     ;                 (RW(IoP) - RW(IoPATM))*1.0D-9*RW(IoWMS+J)/
     ;                                    (RW(IoDCS+J)*RW(IoCVCJ))
c                     calculate partial molal volume,  m**3/mol
                      VX = RW(IoWMS+J)*1.0D-6/RW(IoDCS+J)
                   ENDIF
               RW(IoH) = RW(IoH) + RW(IoSMOL+J)*HX
               RW(IoS) = RW(IoS) + RW(IoSMOL+J)*SX
               RW(IoV) = RW(IoV) + RW(IoSMOL+J)*VX
77             CONTINUE
79         CONTINUE
c
c    here H is in kcal, S is in cal, and V is in m**3
c    convert to J/kg, J/kg-K, and m**3/kg, the basic SI units
c
       RW(IoH) = RW(IoH)*RW(IoCVCJ)*1.D6/TMASS
       RW(IoS) = RW(IoS)*RW(IoCVCJ)*1.D3/TMASS
       RW(IoV) = RW(IoV)*1000.D0/TMASS
c
c    compute U, J/kg
       RW(IoU) = RW(IoH) - RW(IoP)*RW(IoV)
c
c    save last  T and P
       TOLD = RW(IoT)
       POLD = RW(IoP)
c
c    exit
       RETURN
c
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJTPRP(NSMAX,NIW,NRW,NSW,CHEM,IW,RW,SW)
c
c      Determines the temperature-dependent properties of the species.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in the argument list:
c @        NSMAX       maximum number of species (dimension of CHEM)
c @        NIW         dimension of work array IW
c @        NRW         dimension of work array RW
c @        NSW         dimension of work array SW
c @        CHEM(J)     CHARACTER*16 name of Jth species
c          IW(I)       integer work array
c          RW(I)       REAL work array
c @        SW(I)       REAL*4 work array holding species data file
c
c      Variables in the integer work array IW:
c @        JFS(J)   J  if the Jth species is the JFth file species
c                   0  if the Jth species is not in the data file
c @        NP          number of phases
c @        NSP(M)      number of species in Mth phase
c
c      Variables in the real work array RW:
c @        CVCJ        J/cal
c @        DCS(J)      density of condensed species, g/cm**3;  0 for gas
c @        DHF0(J)     enth. of form. at 298.15 K of Jth species
c                      (kcal/mol)
c #        G(J)        g(T,P)/RT for the Jth species
c #        HMH0(J)     enthalpy H-H(298.15) at 1 atm. for Jth species,
c                      kcal/mol
c @        P           system pressure, Pa (N/m**2)
c @        PATM        Pa/atm
c @        RGAS        gas constant, cal/mol-K
c #        S0(J)       entropy at (T,1 atm) for the Jth species,
c                      cal/mol-K
c @        TP(M)       temperature of the Mth phase, K
c @        WMS(J)      molal mass of the Jth species, g/mol.
c
c      Variables used only internally:
c          HX          enthalpy at (T,P), kcal/mol
c          RT          Rgas*T
c          RLP         Rgas*ln(P/Patm)
c          SX          entropy at (T,P), cal/mol-K
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCC       REAL*4         SW
       CHARACTER*16 CHEM
c-----------------------------------------------------------------------
       DIMENSION   CHEM(NSMAX),IW(NIW),RW(NRW),SW(NSW),
     ;   IEPTR(80),ISPTR(10)
c-----------------------------------------------------------------------
c    pointers
       COMMON /SJEPTR/ IEPTR
       COMMON /SJSPTR/ ISPTR
       COMMON /SJTPTR/
     ;   IoKFRZ,IoCVCJ,IoPATM,IoRGAS,IoP,IoT,IoH,IoS,IoU,IoV,
     ;   IoWM,IoTP,IoDCS,IoDHF0,IoWMS,IoHMH0,IoS0,IoWMP,IoXM,IoYM
       EQUIVALENCE (IoNP,IEPTR(7)),(IoNSP,IEPTR(30)),
     ;   (IoG,IEPTR(57)),(IoJFS,ISPTR(6))
c-----------------------------------------------------------------------
       ZERO = 0.0D0
c
c    get parameters
       NP = IW(IoNP)
c
c     compute term
       RLP = RW(IoRGAS)*LOG(RW(IoP)/RW(IoPATM))
c
       J2 = 0
       DO 29 M=1,NP
           J1 = J2 + 1
           J2 = J2 + IW(IoNSP+M)
           RT = RW(IoRGAS)*RW(IoTP+M)
           DO 19 J=J1,J2
c            get thermodynamic properties of the species
               IF (IW(IoJFS+J).LT.0)  THEN
c                  nonfile species
                       CALL SJTGHS(CHEM(J),RW(IoTP+M),
     ;                             RW(IoHMH0+J),RW(IoS0+J))
                   ELSE
c                file species
                       CALL SJTIHS(NSMAX,NSW,SW,IW(IoJFS+J),
     ;                             RW(IoTP+M),RW(IoHMH0+J),RW(IoS0+J))
                   ENDIF
c
c            total H and correct H and S to mixture P for g/RT
c            calculation
               HX = RW(IoHMH0+J) + RW(IoDHF0+J)
               SX = RW(IoS0+J)
               IF (RW(IoDCS+J).EQ.ZERO)  THEN
c                    gas species; correct entropy to mixture P
                       SX = SX - RLP
                   ELSE
c                    condensed species; correct enthalpy to mixture P
                       HX = HX +
     ;                      (RW(IoP) - RW(IoPATM))*1.0D-9*RW(IoWMS+J)/
     ;                                    (RW(IoDCS+J)*RW(IoCVCJ))
                   ENDIF
c            calculate g(T,P)/RT
               RW(IoG+J) = (HX*1000.D0 - RW(IoTP+M)*SX)/RT
19             CONTINUE
29         CONTINUE
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c      Microsoft metacomand:
c
c$STORAGE:2
c-----------------------------------------------------------------------
c
       SUBROUTINE SJTPTS(NAMAX,NPMAX,NSMAX,NIW,NRW,NSW,RW,KU)
c
c      Sets SJTP pointers and checks specified work array dimensions.
c      Loads physical cosntants and conversion factors.
c-----------------------------------------------------------------------
c      Nomenclature:
c
c      Variables in argument list
c
c          NAMAX       maximum number of atom types
c          NPMAX       maximum number of phases
c          NSMAX       maximum number of species
c          NIW         dimension of work array IW
c          NRW         dimension of work array RW
c          RW(I)       REAL work array
c          KU          output unit for error message
c
c      Variables in work array RW:
c          CVCJ        J/cal
c          PATM        Pa/atm
c          RGAS        gas constant, cal/mol-K
c-----------------------------------------------------------------------
c    Targets:
c      NW = max{2*NA,NA+NP}
c      NIW = 9 + 14*NA + 4*NP + 5*NS + NA*NS
c      NRW = 15 + 16*NA + 12*NA*NA + 3*NA*NP + 6*NP +
c    ;      15*NS + NW*NW + NW
c-----------------------------------------------------------------------
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c-----------------------------------------------------------------------
       DIMENSION   RW(NRW)
c-----------------------------------------------------------------------
c  20 additional pointers required by  SJTP
       COMMON /SJTPTR/
     ;   IoKFRZ,IoCVCJ,IoPATM,IoRGAS,IoP,IoT,IoH,IoS,IoU,IoV,
     ;   IoWM,IoTP,IoDCS,IoDHF0,IoWMS,IoHMH0,IoS0,IoWMP,IoXM,IoYM
c-----------------------------------------------------------------------
c    Physical constants
       CVCJ = 4.184D0
       PATM = 101325.0D0
       RGAS = 1.987165D0
c-----------------------------------------------------------------------
c   set pointers required by SJEQLB
       NIWZ = NIW - 1
       NRWZ = NRW - 10 - 2*NPMAX - 7*NSMAX
       CALL SJEPTS(NAMAX,NPMAX,NSMAX,NIWZ,NRWZ,NSW,RW,KU)
c  ** IW pointers
       IoKFRZ = NIWZ + 1
c    check IW dimension
       NIWX =  IoKFRZ
       IF (NIWX.NE.NIW) THEN
           WRITE (KU,1) NAMAX,NPMAX,NSMAX
1          FORMAT (/' SJTP dimensioning error for NAMAX =',I3,
     ;              '  NPMAX =',I3,'  NSMAX =',I3)
           WRITE (KU,2) NIWX
2          FORMAT (/'  NIWORK error; NIWX =',I6)
           STOP
           ENDIF
C ** RW pointers
       IoCVCJ = NRWZ + 1
       IoPATM = IoCVCJ + 1
       IoRGAS = IoPATM + 1
       IoP = IoRGAS + 1
       IoT = IoP + 1
       IoH = IoT + 1
       IoS = IoH + 1
       IoU = IoS + 1
       IoV = IoU + 1
       IoWM = IoV + 1
       IoTP = IoWM
       IoDCS = IoTP + NPMAX
       IoDHF0 = IoDCS + NSMAX
       IoWMS = IoDHF0 + NSMAX
       IoHMH0 = IoWMS + NSMAX
       IoS0 = IoHMH0 + NSMAX
       IoWMP = IoS0 + NSMAX
       IoXM = IoWMP + NPMAX
       IoYM = IoXM + NSMAX
c    check RW dimension
       NRWX = IoYM + NSMAX
       IF (NRWX.NE.NRW) THEN
c            notify user
               WRITE (KU,1) NAMAX,NPMAX,NSMAX
               WRITE (KU,4) NRWX
4              FORMAT (/'  NRWORK error; NRWX =',I6)
               STOP
           ELSE
c            load constants
               RW(IoCVCJ) = CVCJ
               RW(IoPATM) = PATM
               RW(IoRGAS) = RGAS
           ENDIF
       RETURN
       END
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
       SUBROUTINE SJDERR
c
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION  (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
c      Data error message display
c-----------------------------------------------------------------------
       COMMON /SJTERM/ KTERM,KUTERM
c-----------------------------------------------------------------------
       WRITE (KUTERM,1)
1      FORMAT (/' Data input error; try again.')
       RETURN
       END
