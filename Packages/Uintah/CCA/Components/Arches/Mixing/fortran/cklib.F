      SUBROUTINE CKABS
C///////////////////////////////////////////////////////////////////////
C
C     CHEMKIN-II  VERSION 4.9
C
C    CHANGES FROM LAST VERSION
C     1. ADDED THIS COMMENT BLOCK
C    CHANGES FROM VERSION 1.0
C     1. REPLACE "REAL*8" WITH "DOUBLE PRECISION"
C    CHANGES FROM VERSION 1.1
C     1. add change block to CKCHRG
C     2. correct change block in CKHORT
C    CHANGES FROM VERSION 1.2
C     1. change SCOPY for integer arrays to DO loops
C    CHANGES FROM VERSION 1.3
C     1. added PROLOGUES
C     2. binary file now includes minimum length of arrays
C    CHANGES FROM VERSION 1.4
C     1. New versions of CKABML/CKABMS, CKGBML/CKGBMS, CKSBML/CKSBMS
C        for mixture-averaging require additional argument P
C     2. Replace SDOT's and DDOT's by loops
C     3. Reaction strings now have "=>" in irreversible reactions,
C                                 "<=>" in reversible reactions.
C    CHANGES FROM VERSION 1.5
C     1. Implement Landau-Teller rate expression
C     2. Add utility routine CKR2CH
C    CHANGES FROM VERSION 1.6
C     1. Added error checking and additional arguments to character
C        manipulation subroutines.
C     2. Fixed an error with IFIRCH(LINE) in IPPLEN
C    CHANGES FROM VERSION 1.7
C     1. Get rid of non-standard comment statements.
C    CHANGES FROM VERSION 1.8
C     1. vax/cray change blocks for machine constants changed to
C        smallexp, bigexp change blocks
C     2. add ERR= to first read statement in CKINIT
C    CHANGES TO VERSION 2.0
C     1. Subroutine CKLEN to provide necessary lengths of work arrays.
C     2. Subroutine CKKFKR provides arrays of forward and reverse
C        reaction rates.
C    CHANGES TO VERSION 2.1
C     1. New binary file has an additional record to indicate its
C        version, machine precision, and error status
C     2. SUBROUTINE CKPNT reads a binary file to get COMMON /CKSTRT/
C        pointers.
C     3. SUBROUTINE CKSAVE writes pointers and work arrays to a
C        binary file.
C     4. Add COMMON /MACH/ and initialization of BIG,SMALL,EXPARG to
C        SUBROUTINE CKPNT
C     5. Change LOG(*) to LOG(MAX(*,SMALL)) in several subroutines.
C    CHANGES TO VERSION 2.2
C     1. Bugfix in CKABML
C     2. In CKXNUM (and CKSNUM), if NEXP is negative, it is not an
C        error to find fewer values.
C    CHANGES TO VERSION 2.3
C     1. Accept binary file V.2.0
C    CHANGES TO VERSION 2.4
C     1. Accept binary  file V.2.1
C    CHANGES TO VERSION 2.5 (11/15/90, F. Rupley)
C     1. Accept binary file V.2.2
C    CHANGES TO VERSION 2.6 (12/15/90, F. Rupley)
C     1. Accept binary file V.2.3
C    CHANGES TO VERSION 2.7 (12/20/90, F. Rupley)
C     1. Accept binary file V.2.4
C    CHANGES TO VERSION 2.8 (1/18/91, F. Rupley)
C     1. Accept binary file V.2.5
C    CHANGES TO VERSION 2.9 (2/15/91, F. Rupley per R. Kee)
C     1. Add a fourth parameter to the array of Arhennius coefficients
C        for the II reactions;
C        increase the value of NPAR in COMMON /CKSTRT/ by one (this
C        also increases the length of the array of reverse Arhennius
C        parameters);
C        initialize the value of the fourth parameter to 1.0 in
C        CKINIT;
C        use this value as a "perturbation factor" for the forward
C        rates in CKRAT;
C        add SUBROUTINE CKRDEX to allow applications codes to change
C        the perturbation factor RD(I) in sensitivity calculations.
C     2. Accept binary file V.2.6 (LENRCK was increased by II+NREV to
C        reflect above changes in RCKWRK array.
C     CHANGES FOR VERSION 3.0 (4/1/91 F. Rupley)
C     1. Accept binary file V.2.7 (modification of CKDUP)
C     2. Subroutine CKRHEX allows perturbation of thermodynamic
C        coefficient a6.
C     CHANGES FOR VERSION 3.1 (5/9/91 F. Rupley)
C     1. Add Subroutine CKMXTP to return number of temperatures used
C        in thermodynamic fits.
C     CHANGES FOR VERSION 3.2 (6/10/91 H. Moffat)
C     1. Added Subroutine CKFAL, which returns the fall-off parameters
C        for the mechanism.
C     2. Added Subroutine CKNUF, which returns the reactant
C        stoichiometric coefficients.
C     3. Fixed an error in CKSNUM, which caused an error condition when
C        the input string did not have any blanks inbetween words.
C     4. Fixed two errors in CKTHB. The default third body efficiency
C        should be equal to 1.
C     CHANGES FOR VERSION 3.3 (6/27/91 F. Rupley)
C     1. Accept binary file V.2.8 (modified interpreter output to
C        print all 16 characters of species names)
C     CHANGES FOR VERSION 3.4 (2/19/92 F. Rupley)
C     1. Correct error in CKITR (IcNR should be IcNS)
C     CHANGES FOR VERSION 3.6 (2/24/92 F. Rupley per E. Meeks)
C     1. Accept binary file V.2.9 (additional error checking for
C        reverse T-L reactions, 2*II additional real work space)
C     2. Correct calculation for reverse T-L reaction rates
C     3. New subroutines CKRATT, CKRATX (subsets of CKRAT)
C     4. New pointers NcKF,NcKR to store intermediate temperature-
C        dependent rates.
C     CHANGES FOR VERSION 3.7 (3/10/92 F. Rupley per Kee/Grcar)
C     1. Calls to CKRAT replaced by calls to CKRATT and CKRATX.
C     2. New subroutine CKKFRT returns the forward and reverse
C        rates (RKFT, RKRT) calculated by CKRATT (does not consider
C        pressure dependencies).
C     3. New subroutine CKWYPK returns the rates of production
C        given the RKFT and RKRT from (2).
C     CHANGES FOR V.3.8 (4/15/92 F. Rupley)
C     1. Accept binary file V.3.0 (correction to CKDUP)
C     CHANGES FOR V.3.9 (4/17/92 F. Rupley)
C     1. Bugfix in CKSAVE (did not write new pointers NcKF,NcKR)
C     CHANGES FOR V.4.0 (10/1/92 F. Rupley per M. Coltrin)
C     1. COMMON /CKCONS/ VERS, PREC, KERR, LENI, LENR, LENC
C        eliminates need for LINKCK in argument list of CKSAVE
C     CHANGES FOR V.4.1 (2/24/93 F. Rupley)
C     1. Accept binary file V.3.1 (correction to CKREAC)
C     CHANGES FOR V.4.2 (9/14/93 F. Rupley)
C     1. Move perturbation factoring from CKRATT to CKRATX
C     CHANGES FOR V.4.3 (11/9/93 F. Rupley per E. Meeks)
C     1. Min/max single-precision exponent in CKR2CH should be
C        be 30 instead of 38.
C     CHANGES FOR V.4.4 (11/10/93 F. Rupley)
C     1. Accept binary file V.3.2 (correction to CKUNIT)
C     CHANGES FOR V.4.5 (1/26/94 F. Rupley per R. Kee)
C     1. Implement real stoichometric coefficients; binary file V.3.3.
C     CHANGES FOR VERSION 4.6 (3/15/94 F. Rupley)
C     1.  DOS/PC compatibility effort includes adding file names to
C         OPEN statements (updating interpreter), removing 
C         unusued but possibly initialized variables.
C     CHANGES FOR VERSION 4.7 (4/14/94 F. Rupley, suggested by E. Meeks)
C     1.  use INCLUDE 'ckstrt.f' instead of having the CKSTRT common
C         block in every subroutine.
C     CHANGES FOR VERSION 4.8 (4/19/94 F. Rupley)
C     1.  Accept binary file V.3.5 (correction to CKBAL, CKRBAL)
C     CHANGES FOR VERSION 4.9 (4/20/94 F. Rupley)
C     1.  Accept binary file V.3.6 (correction to CKUNIT)
C///////////////////////////////////////////////////////////////////////
C
C  START PROLOGUE
C
C  SUBROUTINE CKABS
C
C  The work arrays contain all the pertinent information about the
C  species and the reaction mechanism.  They also contain some work
C  space needed by various routines for internal manipulations.  If a
C  user wishes to modify a CKLIB subroutine or to write new routines,
C  he will probably want to use the work arrays directly.  The starting
C  adddresses for information stored in the work arrays are found in
C  the labeled common block, COMMON /CKSTRT/, and are explained below.
C
C     COMMON /CKSTRT/ NMM , NKK , NII , MXSP, MXTB, MXTP, NCP , NCP1,
C    1                NCP2, NCP2T,NPAR, NLAR, NFAR, NLAN, NFAL, NREV,
C    2                NTHB, NRLT, NWL,  NRNU, NORD, MXORD,IcMM, IcKK, 
C    3                IcNC, IcPH, IcCH, IcNT, IcNU, IcNK, IcNS, IcNR, 
C    4                IcLT, IcRL, IcRV, IcWL, IcFL, IcFO, IcKF, IcTB, 
C    5                IcKN, IcKT, IcRNU,IcORD,IcKOR,NcAW, NcWT, NcTT, 
C    6                NcAA, NcCO, NcRV, NcLT, NcRL, NcFL, NcKT, NcWL, 
C    7                NcRU, NcRC, NcPA, NcKF, NcKR, NcRNU,NcKOR,NcK1, 
C    8                NcK2, NcK3, NcK4, NcI1, NcI2, NcI3, NcI4
C
C  INDEX CONSTANTS.
C
C     NMM    - Total number of elements in problem.
C     NKK    - Total number of species in problem.
C     NII    - Total number of reactions in problem.
C     MXSP   - Maximum number of species (reactants plus products)
C              allowed for any reaction;  unless changed in the
C              interpreter, MXSP=6.
C     MXTB   - Maximum number of enhanced third-bodies allowed fo any
C              reaction;  unless changed in the interpreter, MXTB=10.
C     MXTP   - Maximum number of temperatures allowed in fits of
C              thermodynamic properties for any species;  unless
C              changed in the interpreter and the thermodynamic
C              database, MXTP=3.
C     NCP    - Number of polynomial coefficients to fits of CP/R for
C              a species;  unless changed in the interpreter and the
C              thermodynamic database, NCP=5.
C     NCP1   - NCP + 1
C     NCP2   - NCP + 2
C     NCP2T  - Total number of thermodynamic fit coefficients for the
C              species;  unless changed, NCP2T = (MXTP-1)*NCP2 = 14.
C     NPAR   - Number of parameters required in the rate expression
C              for the reactions;  in the current formulation NPAR=3.
C     NLAR   - Number of parameters required for Landau-Teller
C              reactions; NLAR=4.
C     NFAR   - Number of parameters allowed for fall-off reactions;
C              NFAR=8.
C     NLAN   - Total number of Landau-Teller reactions.
C     NFAL   - Total number of fall-off reactions.
C     NREV   - Total number of reactions with reverse parameters.
C     NTHB   - Total number of reactions with third-bodies.
C     NRLT   - Total number of Landau-Teller reactions with reverse
C              parameters.
C     NWL    - Total number of reactions with radiation wavelength
C              enhancement factors.
C
C  STARTING ADDRESSES FOR THE CHARACTER WORK SPACE, CCKWRK.
C
C     IcMM   - Starting address of an array of the NMM element names.
C              CCKWRK(IcMM+M-1) is the name of the Mth element.
C     IcKK   - Starting address of an array of the NKK species names.
C              CCKWRK(icKK+M-1) is the name of the Kth species.
C
C  STARTING ADDRESSES FOR THE INTEGER WORK SPACE, ICKWRK.
C
C     IcNC  - Starting address of an array of the elemental content
C             of the NMM elements in the NKK species.
C             ICKWRK(IcNC+(K-1)*NMM+M-1) is the number of atoms of the
C             Mth element in the Kth species.
C     IcPH  - Starting address of an array of phases of the NKK species.
C             ICKWRK(IcPH+K-1) = -1, the Kth species is a solid
C                              =  0, the Kth species is a gas
C                              = +1, the Kth species is a liquid
C     IcCH  - Starting address of an array of the electronic charges of
C             the NKK species.
C             ICKWRK(IcCH+K-1) = -2, the Kth species has two excess
C                                    electrons.
C     IcNT  - Starting address of an array of the number of temperatures
C             used to fit thermodynamic coefficients for the
C             NKK species.
C             ICKWRK(IcNT+K-1) = N, N temperatures were used in the fit
C                                   for the Kth species.
C     IcNU  - Starting address of a matrix of stoichiometric
C             coefficients of the MXSP species in the NII reactions.
C             ICKWRK(IcNU+(I-1)*MXSP+N-1) is the coefficient of the Nth
C             participant species in the Ith reaction
C     IcNK  - Starting address of a matrix of species index numbers for
C             the MXSP species in the NII reactions.
C             ICKWRK(IcNK+(I-1)*MXSP+N-1) = K, the species number of
C             the Nth participant species in the Ith reaction.
C     IcNS  - Starting address of an array of the total number of
C             participant species for the NII reactions, and the
C             reversibility of the reactions.
C             ICKWRK(IcNS+I-1) = +N, the Ith reaction is reversible
C                                    and has N participant species
C                                    (reactants + products)
C                              = -N, the Ith reaction is irreversible
C                                    and has N participant species
C                                    (reactants + products)
C     IcNR  - Starting address of an array of the number of reactants
C             only for the NII reactions.
C             ICKWRK(IcNR+I-1) is the total number of reactants in the
C             Ith reaction.
C     IcLT  - Starting address of an array of the NLAN reaction numbers
C             for which Landau-Teller parameters have been given.
C             ICKWRK(IcLT+N-1) is the reaction number of the Nth
C             Landau-Teller reaction.
C     IcRL  - Starting address of an array of the NRLT reaction numbers
C             for which reverse Landau-Teller parameters have been
C             given.
C             ICKWRK(IcRL+N-1) is the reaction number of the Nth
C             reaction with reverse Landau-Teller parameters.
C     IcRV  - Starting address of an array of the NREV reaction numbers
C             for which reverse Arhennius coefficients have been given.
C             ICKWRK(IcRV+N-1) is the reaction number of the Nth
C             reaction with reverse coefficients.
C     IcWL  - Starting address of an array of the NWL reactions numbers
C             for which radiation wavelength has been given.
C             ICKWRK(IcWL+N-1) is the reaction number of the Nth
C             reaction with wavelength enhancement.
C     IcFL  - Starting address of an array of the NFAL reaction numbers
C             with fall-off parameters.
C             ICKWRK(IcFL+N-1) is the reaction number of the Nth
C             fall-off reaction.
C     IcFO  - Starting address of an array describing the type of
C             the NFAL fall-off reactions.
C             ICKWRK(IcFO+N-1) is the type of the Nth fall-off
C             reaction: 1 for 3-parameter Lindemann Form
C                       2 for 6- or 8-parameter SRI Form
C                       3 for 6-parameter Troe Form
C                       4 for 7-parameter Troe form
C     IcKF  - Starting address of an array of the third-body species
C             numbers for the NFAL fall-off reactions.
C             ICKWRK(IcKF+N-1) = 0: the concentration of the third-body
C                                   is the total of the concentrations
C                                   of all species in the problem
C                              = K: the concentration of the third-body
C                                   is the concentration of species K.
C     IcTB  - Starting address of an array of reaction numbers for the
C             NTHB third-body reactions.
C             ICKWRK(IcTB+N-1) is the reaction number of the Nth
C             third-body reaction.
C     IcKN  - Starting address of an array of the number of enhanced
C             third bodies for the NTHB third-body reactions.
C             ICKWRK(IcKN+N-1) is the number of enhanced species for
C             the Nth third-body reaction.
C     IcKT  - Starting address of an array of species numbers for the
C             MXTB enhanced 3rd bodies in the NTHB third-body reactions.
C             ICKWRK(IcTB+(N-1)*MXTB+L-1) is the species number of the
C             Lth enhanced species in the Nth third-body reaction.
C
C  STARTING ADDRESSES FOR THE REAL WORK SPACE, RCKWRK.
C
C     NcAW  - Starting address of an array of atomic weights of the
C             NMM elements (gm/mole).
C             RCKWRK(NcAW+M-1) is the atomic weight of element M.
C     NcWT  - Starting address of an array of molecular weights for
C             the NKK species (gm/mole).
C             RCKWRK(NcWT+K-1) is the molecular weight of species K.
C     NcTT  - Starting address of an array of MXTP temperatures used in
C             the fits of thermodynamic properties of the NKK species
C             (Kelvins).
C             RCKWRK(NcTT+(K-1)*MXTP+N-1) is the Nth temperature for the
C             Kth species.
C     NcAA  - Starting address of a three-dimensional array of
C             coefficients for the NCP2 fits to the thermodynamic
C             properties for the NKK species, for (MXTP-1) temperature
C             ranges.
C             RCKWRK(NcAA+(L-1)*NCP2+(K-1)*NCP2T+N-1) = A(N,L,K);
C             A(N,L,K),N=1,NCP2T = polynomial coefficients in the fits
C             for the Kth species and the Lth temperature range, where
C             the total number of temperature ranges for the Kth species
C             is ICKWRK(IcNT+K-1) - 1.
C     NcCO  - Starting address of an array of NPAR Arrhenius parameters
C             for the NII reactions.
C             RCKWRK(NcCO+(I-1)*NPAR+(L-1)) is the Lth parameter of the
C             Ith reaction, where
C                L=1 is the pre-exponential factor (mole-cm-sec-K),
C                L=2 is the temperature exponent, and
C                L=3 is the activation energy (Kelvins).
C     NcRV  - Starting address of an array of NPAR reverse Arrhenius
C             parameters for the NREV reactions.
C             RCKWRK(NcRV+(N-1)*NPAR+(L-1)) is the Lth reverse
C             parameter for the Nth reaction with reverse parameters
C             defined, where
C                L=1 is the pre-exponential factor (mole-cm-sec-K),
C                L=2 is the temperature exponent, and
C                L=3 is the activation energy (Kelvins).
C             The reaction number is ICKWRK(IcRV+N-1).
C     NcLT  - Starting location of an array of the NLAR parameters for
C             the NLAN Landau-Teller reactions.
C             RCKWRK(NcLT+(N-1)*NLAR+(L-1)) is the Lth Landau-Teller
C             parameter for the Nth Landau-Teller reaction, where
C                L=1 is B(I) (Eq. 72) (Kelvins**1/3), and
C                L=2 is C(I) (Eq. 72) (Kelvins**2/3).
C             The reaction number is ICKWRK(IcLT+N-1).
C     NcRL  - Starting location of an array of the NLAR reverse
C             parameters for the NRLT Landau-Teller reactions for which
C             reverse parameters were given.
C             RCKWRK(NcRL+(N-1)*NLAR+(L-1)) is the Lth reverse
C             parameter for the Nth reaction with reverse Landau-Teller
C             parameters, where
C                L=1 is B(I) (Eq. 72) (Kelvins**1/3), and
C                L=2 is C(I) (Eq. 72) (Kelvins**2/3).
C             The reaction number is ICKWRK(IcRL+N-1).
C     NcFL  - Starting location of an array of the NFAR fall-off
C             parameters for the NFL fall-off reactions.
C             RCKWRK(NcFL+(N-1)*NFAR+(L-1)) is the Lth fall-off
C             parameter for the Nth fall-off reaction, where the low
C             pressure limits are defined by
C                L=1 is the pre-exponential factor (mole-cm-sec-K),
C                L=2 is the temperature exponent, and
C                L=3 is the activation energy (Kelvins).
C             Additional parameters define the centering, depending on
C             the type of formulation -
C                Troe: L=4 is the Eq. 68 parameter a,
C                      L=5 is the Eq. 68 parameter T*** (Kelvins),
C                      L=6 is the Eq. 68 parameter T*   (Kelvins), and
C                      L=7 is the Eq. 68 parameter T**  (Kelvins).
C                SRI:  L=4 is the Eq. 69 parameter a,
C                      L=5 is the Eq. 69 parameter b (Kelvins),
C                      L=6 is the Eq. 69 parameter c (kelvins),
C                      L=7 is the Eq. 69 parameter d, and
C                      L=8 is the Eq. 69 parameter e.
C             The reaction number is ICKWRK(IcFL+N-1), and the type
C             of formulation is ICKWRK(IcFO+N-1).
C     NcWL  - Starting location of an array of wavelengths for the NWL
C             wavelength-enhanced reactions.
C             RCKWRK(NcWL+N-1) is the wavelength enhancement (angstrom)
C             for the Nth wavelength-enhanced reaction;
C             the reaction number is ICKWRK(IcWL+N-1).
C     NcKT  - Starting location of an array of MXTB enhancement factors
C             for the NTHB third-body reactions.
C             RCKWRK(NcKT+(N-1)*MXTB+(L-1)) is the enhancement factor
C             for the Lth enhanced species in the Nth third-body
C             reaction;
C             the reaction number is ICKWRK(IcTB+N-1), and the Lth
C             enhanced species index number is
C             ICKWRK(IcKT+(N-1)*MXTB+L-1).
C     NcRU  - RCKWRK(NcRU) is the universal gas constant (ergs/mole-K).
C     NcRC  - RCKWRK(NcRC) is the universal gas constant (cal/mole-K).
C     NcPA  - RCKWRK(NcPA) is the pressure of one standard atmosphere
C             (dynes/cm**2).
C     NcKF  - Starting address of an array of intermediate forward
C             temperature-dependent rates for the II reactions.
C     NcKR  - Starting address of an array of intermediate reverse
C             temperature-dependent rates for the II reactions.
C     NcK1  - Starting addresses of arrays of internal work space
C     NcK2
C     NcK3                  space of length NKK
C     NcK4
C     NcI1  - Starting addresses of arrays of internal work space
C     NcI2
C     NcI3                  space of length NII
C     NcI4
C
C  The binary file consists of the following binary records:
C
C
C   1) Information about the binary file:  VERS, PREC, KERR
C      Where VERS   = character*16 string representing the version
C                     number of the interpreter which created the
C                     the binary file.
C            PREC   = character*16 string representing the machine
C                     precision of the binary file (SINGLE, DOUBLE).
C            KERR   = logical which indicates whether or not
C                    an error occurred in the interpreter input.
C   2) Index constants:
C      LENI, LENR, LENC, NMM,  NKK,  NII,  MXSP, MXTB,
C      MXTP, NCP,  NPAR, NLAR, NFAR, NREV, NFAL, NTHB,
C      NLAN, NRLT, NWL, NCHRG
C      Where LENI = required length of ICKWRK.
C            LENR = required length of RCKWRK.
C            LENC = required length of CCKWRK.
C            NCHRG= total number of species with an electronic
C                   charge not equal to zero.
C
C  3) Element information:
C     ((CCKWRK(IcMM + M-1),                       !element names
C      RCKWRK(NcAW + M-1)),                       !atomic weights
C      M=1,NMM)
C
C  4) Species information:
C     ((CCKWRK(IcKK+K-1),                         !species names
C      (ICKWRK(IcNC+(K-1)*NMM+M-1),M=1,MMM),      !composition
C      ICKWRK(IcPH+K-1),                          !phase
C      ICKWRK(IcCH+K-1),                          !charge
C      RCKWRK(NcWT+K-1),                          !molec weight
C      ICKWRK(IcNT+K-1),                          !# of fit temps
C      (RCKWRK(NcTT+(K-1)*MXTP + L-1),L=1,MXTP),  !array of temps
C      ((RCKWRK(NcAA+(L-1)*NCP2+(K-1)*NCP2T+N-1), !fit coeff'nts
C               N=1,NCP2), L=1,(MXTP-1))),
C      K = 1,NKK)
C
C  5) Reaction information (if NII>0):
C     (ICKWRK(IcNS+I-1),                          !# of species
C      ICKWRK(IcNR+I-1),                          !# of reactants
C      (RCKWRK(NcCO+(I-1)*NPAR+N-1), N=1,NPAR),   !Arr. coefficients
C      (ICKWRK(IcNU+(I-1)*MXSP+N-1),              !stoic coef
C      ICKWRK(IcNK+(I-1)*MXSP+N-1), N=1,MXSP),    !species numbers
C      I = 1,NII)
C
C  6) Reverse parameter information (if NREV>0):
C     (ICKWRK(IcRV+N-1),                          !reaction numbers
C      (RCKWRK(NcRV+(N-1)*NPAR+L-1),L=1,NPAR),    !reverse coefficients
C      N = 1,NREV)
C
C  7) Fall-off reaction information (if NFAL>0):
C     (ICKWRK(IcFL+N-1),                          !reaction numbers
C      ICKWRK(IcFO+N-1),                          !fall-off option
C      ICKWRK(IcKF+N-1),                          !3rd-body species
C      (RCKWRK(NcFL+(N-1)*NFAR+L-1),L=1,NFAR),    !fall-off parameters
C      N=1,NFAL)
C
C  8) Third-body reaction information (if NTHB>0):
C     (ICKWRK(IcTB+N-1),                          !reaction numbers
C      ICKWRK(IcKN+N-1),                          !# of 3rd bodies
C      (ICKWRK(IcKT+(N-1)*MXTB+L-1),              !3rd-body species
C      RCKWRK(NcKT+(N-1)*MXTB+L-1),L=1,MXTB),     !enhancement factors
C      N=1,NTHB)
C
C  9) Landau-Teller reaction information (if NLAN>0):
C     (ICKWRK(IcLT+N-1),                          !reaction numbers
C      (RCKWRK(NcLT+(N-1)*NLAR+L-1),L=1,NLAR),    !L-T parameters
C      N=1,NLAN)
C
C 10) Reverse Landau-Teller reaction information (if NRLT>0):
C     (ICKWRK(IcRL+N-1),                          !reaction numbers
C      (RCKWRK(NcRL+(N-1)*NLAR+L-1),L=1,NLAR),    !rev. L-T parameters
C      N=1,NRLT)
C
C 11) Photon radiation reaction information (if NWL>0):
C     (ICKWRK(IcWL+N-1),                          !reaction numbers
C      RCKWRK(NcWL+N-1),                          !wavelength factor
C      N=1,NWL)
C
C  END PROLOGUE
C
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKABE  (ICKWRK, RCKWRK, RA, RB, RE)
C
C  START PROLOGUE
C
C  SUBROUTINE CKABE  (ICKWRK, RCKWRK, RA, RB, RE)
C     Returns the Arrhenius coefficients of the reactions;
C     see Eq. (52).
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     RA     - Pre-exponential constants for the reactions.
C                   cgs units - mole-cm-sec-K
C                   Data type - real array
C                   Dimension RA(*) at least II, the total number of
C                   reactions.
C     RB     - Temperature dependence exponents for the reactions.
C                   cgs units - none
C                   Data type - real array
C                   Dimension RB(*) at least II, the total number of
C                   reactions.
C     RE     - Activation energies for the reactions.
C                   cgs units - Kelvins
C                   Data type - real array
C                   Dimension RE(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
#include "ckstrt.h"
      DIMENSION RA(*), RB(*), RE(*), ICKWRK(*), RCKWRK(*)
C

C
      DO 100 I = 1, NII
         IND = NcCO + (I-1)*(NPAR+1)
         RA(I) = RCKWRK(IND)
         RB(I) = RCKWRK(IND+1)
         RE(I) = RCKWRK(IND+2)
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKABML (P, T, X, ICKWRK, RCKWRK, ABML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKABML (P, T, X, ICKWRK, RCKWRK, ABML)*
C     Returns the Helmholtz free energy of the mixture in molar units,
C     given the pressure, temperature, and mole fractions;
C     see Eq. (46).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     ABML   - Mean Helmholtz free energy in molar units.
C                   cgs units - ergs/mole
C                   Data type - real scalar
C
C   END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKUML (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
      RLNP = RCKWRK(NcRU) * LOG(P / RCKWRK(NcPA))
C
      ABML = 0.0D0
      DO 100 K = 1, NKK
         ABML = ABML + X(K) * ( RCKWRK(NcK2 + K - 1) - T *
     1          (RCKWRK(NcK1 + K - 1) - RCKWRK(NcRU)
     2          * LOG(MAX(X(K),SMALL)) - RLNP) )
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKABMS (P, T, Y, ICKWRK, RCKWRK, ABMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKABMS (P, T, Y, ICKWRK, RCKWRK, ABMS)*
C     Returns the mean Helmholtz free energy of the mixture in
C     mass units, given the pressure, temperature and mass fractions;
C     see Eq. (47).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     ABMS   - Mean Helmholtz free energy in mass units.
C                   cgs units - ergs/gm
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKUML (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
      CALL CKYTX (Y, ICKWRK, RCKWRK, RCKWRK(NcK3))
C
      RLNP = RCKWRK(NcRU) * LOG (P / RCKWRK(NcPA))
C
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + RCKWRK(NcK3 + K - 1) *
     1             ( RCKWRK(NcK2 + K - 1) - T *
     2             ( RCKWRK(NcK1 + K - 1) -
     3               RCKWRK(NcRU)*
     4               LOG(MAX(RCKWRK(NcK3 + K - 1),SMALL)) - RLNP))
  100 CONTINUE
C
      CALL CKMMWY (Y, ICKWRK, RCKWRK, WTM)
      ABMS = SUM / WTM
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKAML  (T, ICKWRK, RCKWRK, AML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKAML  (T, ICKWRK, RCKWRK, AML)
C     Returns the standard state Helmholtz free energies in molar
C     units;  see Eq. (25).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     AML    - Standard state Helmholtz free energies in molar units
C              for the species.
C                   cgs units - ergs/mole
C                   Data type - real array
C                   Dimension AML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), AML(*)
C
#include "ckstrt.h"
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKHML (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
C
      RUT = T*RCKWRK(NcRU)
      DO 150 K = 1, NKK
         AML(K) = RCKWRK(NcK2 + K - 1) - RUT - T*RCKWRK(NcK1 + K - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKAMS  (T, ICKWRK, RCKWRK, AMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKAMS  (T, ICKWRK, RCKWRK, AMS)
C     Returns the standard state Helmholtz free energies in mass
C     units;  see Eq. (32).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     AMS    - Standard state Helmholtz free energies in mass units
C              for the species.
C                   cgs units - ergs/gm
C                   Data type - real array
C                   Dimension AMS(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), AMS(*)
C
#include "ckstrt.h"
C
      CALL CKSMS (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKHMS (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
C
      RUT = T*RCKWRK(NcRU)
      DO 150 K = 1, NKK
         AMS(K) = RCKWRK(NcK2 + K - 1) - RUT/RCKWRK(NcWT + K - 1)
     1                                 - T*RCKWRK(NcK1 + K - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKATHM (NDIM1, NDIM2, ICKWRK, RCKWRK, MAXTP, NT, TMP,
     1                   A)
C
C  START PROLOGUE
C
C  SUBROUTINE CKATHM (NDIM1, NDIM2, ICKWRK, RCKWRK, MAXTP, NT, TMP,
C                     A)
C     Returns the coefficients of the fits for thermodynamic properties
C     of the species; see Eqns. (19)-(21).
C
C  INPUT
C     NDIM1  - First dimension of the three-dimensional array of
C              thermodynamic fit coefficients, A; NDIM1 must be at
C              least NCP2, the total number of coefficients for one
C              temperature range.
C     NDIM2  - Second dimension of the three-dimensionalarray of
C              thermodynamic fit coefficients, A; NDIM2 must be at
C              least MXPT-1, the total number of temperature ranges.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     NT     - Number of temperatures used for fitting coefficients of
C              thermodynamic properties for the species.
C                   Data type - integer array
C                   Dimension NT(*) at least KK, the total number of
C                   species.
C     TMP    - Common temperatures dividing the thermodynamic fits for
C              the species.
C                   cgs units - K
C                   Data type - real array
C                   Dimension TMP(MAXT,*) exactly MAXT for the first
C                   dimension (the maximum number of temperatures
C                   allowed for a species) , and at least KK for the
C                   second dimension (the total number of species)
C     A      - Three dimensional array of fit coefficients to the
C              thermodynamic data for the species.
C              The indicies in  A(N,L,K) mean-
C              N = 1,NN are polynomial coefficients in CP/R
C                CP/R(K)=A(1,L,K) + A(2,L,K)*T + A(3,L,K)*T**2 + ...
C              N = NN+1 is a6 in Eq. (20)
C              N = NN+2 is a7 in Eq. (21)
C              L = 1..MXTP-1 is for each temperature range.
C              K  is  the  species index
C                   Data type - real array
C                   Dimension A(NPCP2,NDIM2,*) exactly NPCP2 and MXTP-1
C                   for the first and second dimensions and at least
C                   KK for the third.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION NT(*), TMP(MAXTP,*), A(NDIM1,NDIM2,*),
     1          ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      DO 100 K = 1, NKK
         NT(K) = ICKWRK(IcNT + K - 1)
  100 CONTINUE
C
      DO 140 L = 1, MXTP
         DO 140 K = 1, NKK
            TMP(L,K) = RCKWRK(NcTT + (K-1)*MXTP + L - 1)
  140 CONTINUE
C
      DO 150 K = 1, NKK
         DO 150 L = 1, MXTP-1
            NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
            DO 150 M = 1, NCP2
               A(M, L, K) = RCKWRK(NA1 + M - 1)
150   CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKAWT  (ICKWRK, RCKWRK, AWT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKAWT  (ICKWRK, RCKWRK, AWT)
C     Returns the atomic weights of the elements
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     AWT    - Atomic weights of the elements.
C                   cgs units - gm/mole
C                   Data type - real array
C                   Dimension AWT(*) at least MM, the total number of
C                   elements in the problem.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), AWT(*)
C
#include "ckstrt.h"
C
      DO 100 M = 1, NMM
         AWT(M) = RCKWRK(NcAW + M - 1)
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCDC  (T, C, ICKWRK, RCKWRK, CDOT, DDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCDC  (T, C, ICKWRK, RCKWRK, CDOT, DDOT)
C     Returns the molar creation and destruction rates of the species
C     given the temperature and molar concentrations;  see Eq. (73).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     DDOT   - Chemical molar destruction rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension DDOT(*) at least KK, the total number of
C                   species.
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), C(*), CDOT(*), DDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 K = 1, NKK
         RCKWRK(NcK1 + K - 1) = C(K)
   50 CONTINUE
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))

C
      DO 100 K = 1, NKK
         CDOT(K) = 0.0D0
         DDOT(K) = 0.0D0
  100 CONTINUE
      DO 200 I = 1, NII
         RKF = RCKWRK(NcI1 + I -1)
         RKR = RCKWRK(NcI2 + I -1)
         DO 200 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               NUR = IABS(ICKWRK(IcNU + (I-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*NUR
               DDOT(NKR) = DDOT(NKR) + RKF*NUR
            ENDIF
            IF (NKP .NE. 0) THEN
               NUP = ICKWRK(IcNU + (I-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*NUP
               DDOT(NKP) = DDOT(NKP) + RKR*NUP
            ENDIF
  200 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 300 L = 1, NRNU
         I   = ICKWRK(IcRNU + L - 1)
         RKF = RCKWRK(NcI1 + I -1)
         RKR = RCKWRK(NcI2 + I -1)
         DO 300 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               RNUR = ABS(RCKWRK(NcRNU + (L-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*RNUR
               DDOT(NKR) = DDOT(NKR) + RKF*RNUR
            ENDIF
            IF (NKP .NE. 0) THEN
               RNUP = RCKWRK(NcRNU + (L-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*RNUP
               DDOT(NKP) = DDOT(NKP) + RKR*RNUP
            ENDIF
  300 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCDXP (P, T, X, ICKWRK, RCKWRK, CDOT, DDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCDXP (P, T, X, ICKWRK, RCKWRK, CDOT, DDOT)
C     Returns the molar creation and destruction rates of the species
C     given pressure, temperature and mole fractions;  see Eq. (73).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     DDOT   - Chemical molar destruction rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension DDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), CDOT(*), DDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCP (P, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))

C
      DO 100 K = 1, NKK
         CDOT(K) = 0.0D0
         DDOT(K) = 0.0D0
  100 CONTINUE
      DO 200 I = 1, NII
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 200 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               NUR = IABS(ICKWRK(IcNU + (I-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*NUR
               DDOT(NKR) = DDOT(NKR) + RKF*NUR
            ENDIF
            IF (NKP .NE. 0) THEN
               NUP = ICKWRK(IcNU + (I-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*NUP
               DDOT(NKP) = DDOT(NKP) + RKR*NUP
            ENDIF
  200 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 300 L = 1, NRNU
         I   = ICKWRK(IcRNU + L - 1)
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 300 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               RNUR = ABS(RCKWRK(NcRNU + (L-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*RNUR
               DDOT(NKR) = DDOT(NKR) + RKF*RNUR
            ENDIF
            IF (NKP .NE. 0) THEN
               RNUP = RCKWRK(NcRNU + (L-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*RNUP
               DDOT(NKP) = DDOT(NKP) + RKR*RNUP
            ENDIF
  300 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCDXR (RHO, T, X, ICKWRK, RCKWRK, CDOT, DDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCDXR (RHO, T, X, ICKWRK, RCKWRK, CDOT, DDOT)
C     Returns the molar creation and destruction rates of the species
C     given the mass density, temperature and mole fractions;
C     see Eq. (73).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     DDOT   - Chemical molar destruction rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension DDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), CDOT(*), DDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCR (RHO, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 K = 1, NKK
         CDOT(K) = 0.0D0
         DDOT(K) = 0.0D0
  100 CONTINUE
      DO 200 I = 1, NII
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 200 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               NUR = IABS(ICKWRK(IcNU + (I-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*NUR
               DDOT(NKR) = DDOT(NKR) + RKF*NUR
            ENDIF
            IF (NKP .NE. 0) THEN
               NUP = ICKWRK(IcNU + (I-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*NUP
               DDOT(NKP) = DDOT(NKP) + RKR*NUP
            ENDIF
  200 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 300 L = 1, NRNU
         I   = ICKWRK(IcRNU + L - 1)
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 300 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               RNUR = ABS(RCKWRK(NcRNU + (L-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*RNUR
               DDOT(NKR) = DDOT(NKR) + RKF*RNUR
            ENDIF
            IF (NKP .NE. 0) THEN
               RNUP = RCKWRK(NcRNU + (L-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*RNUP
               DDOT(NKP) = DDOT(NKP) + RKR*RNUP
            ENDIF
  300 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCDYP (P, T, Y, ICKWRK, RCKWRK, CDOT, DDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCDYP (P, T, Y, ICKWRK, RCKWRK, CDOT, DDOT)
C     Returns the molar creation and destruction rates of the species
C     given mass density, temperature and mass fractions;
C     see Eq. (73).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     DDOT   - Chemical molar destruction rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension DDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), CDOT(*), DDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKYTCP (P, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 K = 1, NKK
         CDOT(K) = 0.0D0
         DDOT(K) = 0.0D0
  100 CONTINUE
      DO 200 I = 1, NII
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 200 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               NUR = IABS(ICKWRK(IcNU + (I-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*NUR
               DDOT(NKR) = DDOT(NKR) + RKF*NUR
            ENDIF
            IF (NKP .NE. 0) THEN
               NUP = ICKWRK(IcNU + (I-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*NUP
               DDOT(NKP) = DDOT(NKP) + RKR*NUP
            ENDIF
  200 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
C
      DO 300 L = 1, NRNU
         I   = ICKWRK(IcRNU + L - 1)
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 300 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               RNUR = ABS(RCKWRK(NcRNU + (L-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*RNUR
               DDOT(NKR) = DDOT(NKR) + RKF*RNUR
            ENDIF
            IF (NKP .NE. 0) THEN
               RNUP = RCKWRK(NcRNU + (L-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*RNUP
               DDOT(NKP) = DDOT(NKP) + RKR*RNUP
            ENDIF
  300 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCDYR (RHO, T, Y, ICKWRK, RCKWRK, CDOT, DDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCDYR (RHO, T, Y, ICKWRK, RCKWRK, CDOT, DDOT)
C     Returns the molar creation and destruction rates of the species
C     given the mass density, temperature and mass fractions;
C     see Eq. (73).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     DDOT   - Chemical molar destruction rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension DDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), CDOT(*), DDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKYTCR (RHO, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 K = 1, NKK
         CDOT(K) = 0.0D0
         DDOT(K) = 0.0D0
  100 CONTINUE
      DO 200 I = 1, NII
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 200 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               NUR = IABS(ICKWRK(IcNU + (I-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*NUR
               DDOT(NKR) = DDOT(NKR) + RKF*NUR
            ENDIF
            IF (NKP .NE. 0) THEN
               NUP = ICKWRK(IcNU + (I-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*NUP
               DDOT(NKP) = DDOT(NKP) + RKR*NUP
            ENDIF
  200 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 300 L = 1, NRNU
         I   = ICKWRK(IcRNU + L - 1)
         RKF = RCKWRK(NcI1 + I - 1)
         RKR = RCKWRK(NcI2 + I - 1)
         DO 300 N = 1, 3
            NKR = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            NKP = ICKWRK(IcNK + (I-1)*MXSP + N + 2)
            IF (NKR .NE. 0) THEN
               RNUR = ABS(RCKWRK(NcRNU + (L-1)*MXSP + N - 1))
               CDOT(NKR) = CDOT(NKR) + RKR*RNUR
               DDOT(NKR) = DDOT(NKR) + RKF*RNUR
            ENDIF
            IF (NKP .NE. 0) THEN
               RNUP = RCKWRK(NcRNU + (L-1)*MXSP + N + 2)
               CDOT(NKP) = CDOT(NKP) + RKF*RNUP
               DDOT(NKP) = DDOT(NKP) + RKR*RNUP
            ENDIF
  300 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCHRG (ICKWRK, RCKWRK, KCHARG)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCHRG (ICKWRK, RCKWRK, KCHARG)
C     Returns the electronic charges of the species
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     KCHARG - Electronic charges of the species; KCHARG(K)=-2
C              indicates that the Kth species has two excess electrons.
C                   Data type - integer array
C                   Dimension KCHARG(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), KCHARG(*)
C
#include "ckstrt.h"
C
      DO 100 K = 1, NKK
         KCHARG(K) = ICKWRK(IcCH + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCOMP (IST, IRAY, II, I)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCOMP (IST, IRAY, II, I)*
C     Returns the index of an element of a reference character
C     string array which corresponds to a character string;
C     leading and trailing blanks are ignored.
C
C
C  INPUT
C     IST   - A character string.
C                  Data type - CHARACTER*(*)
C     IRAY  - An array of character strings;
C                  Data type - CHARACTER*(*)
C                  Dimension IRAY(*) at least II
C     II    - The length of IRAY.
C                  Data type - integer scalar.
C
C  OUTPUT
C     I     - The first integer location in IRAY in which IST
C             corresponds to IRAY(I); if IST is not also an
C             entry in IRAY, I=0.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER*(*) IST, IRAY(*)
C
      I = 0
      DO 10 N = II, 1, -1
         IS1 = IFIRCH(IST)
         IS2 = ILASCH(IST)
         IR1 = IFIRCH(IRAY(N))
         IR2 = ILASCH(IRAY(N))
         IF ( IS2.GE.IS1 .AND. IS2.GT.0 .AND.
     1        IR2.GE.IR1 .AND. IR2.GT.0 .AND.
     2        IST(IS1:IS2).EQ.IRAY(N)(IR1:IR2) ) I = N
   10 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCONT (K, Q, ICKWRK, RCKWRK, CIK)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCONT (K, Q, ICKWRK, RCKWRK, CIK)
C     Returns the contributions of the reactions to the molar
C     production rate of a species;  see Eqs. (49) and (51).
C
C  INPUT
C     K      - Integer species number.
C                   Data type - integer scalar
C     Q      - Rates of progress for the reactions.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension Q(*) at least II, the total number of
C                   reactions.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CIK    - Contributions of the reactions to the molar production
C              rate of the Kth species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CIK(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Q(*), CIK(*)
C
#include "ckstrt.h"
C
      DO 100 I = 1, NII
         CIK(I) = 0.0D0
  100 CONTINUE
      DO 200 N = 1, MXSP
         DO 200 I = 1, NII
            NK = ICKWRK(IcNK + MXSP*(I-1) + N - 1)
            IF (NK .EQ. K) THEN
               NC = ICKWRK(IcNU + MXSP*(I-1) + N - 1)
               CIK(I) = CIK(I) + NC*Q(I)
            ENDIF
200   CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 300 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         DO 300 N = 1, MXSP
            NK = ICKWRK(IcNK + MXSP*(I-1) + N - 1)
            IF (NK .EQ. K) THEN
               RNC = RCKWRK(NcRNU + MXSP*(L-1) + N - 1)
               CIK(I) = CIK(I) + RNC*Q(I)
            ENDIF
  300 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCPBL (T, X, ICKWRK, RCKWRK, CPBML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCPBL (T, X, ICKWRK, RCKWRK, CPBML)
C     Returns the mean specific heat at constant pressure;
C     see Eq. (33).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C   OUTPUT
C     CPBML  - Mean specific heat at constant pressure in molar units.
C                   cgs units - ergs/(mole*K)
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*)
C
#include "ckstrt.h"
C
      CALL CKCPML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CPBML = 0.0D0
      DO 100 K = 1, NKK
         CPBML = CPBML + X(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCPBS (T, Y, ICKWRK, RCKWRK, CPBMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCPBS (T, Y, ICKWRK, RCKWRK, CPBMS)
C     Returns the mean specific heat at constant pressure;
C     see Eq. (34).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CPBMS  - Mean specific heat at constant pressure in mass units.
C                   cgs units - ergs/(gm*K)
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*)
C
#include "ckstrt.h"
C
      CALL CKCPMS (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CPBMS = 0.0D0
      DO 100 K = 1, NKK
         CPBMS = CPBMS + Y(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCPML (T, ICKWRK, RCKWRK, CPML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCPML (T, ICKWRK, RCKWRK, CPML)
C     Returns the specific heats at constant pressure in molar units
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CPML   - Specific heats at constant pressure in molar units
C              for the species.
C                   cgs units - ergs/(mole*K)
C                   Data type - real array
C                   Dimension CPML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), CPML(*), TN(10)
C
#include "ckstrt.h"
C
      TN(1) = 1.0D0
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         CPML(K) = 0.0
         DO 250 N = 1, NCP
            CPML(K) = CPML(K) + RCKWRK(NcRU)*TN(N)*RCKWRK(NA1 + N - 1)
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCPMS (T, ICKWRK, RCKWRK, CPMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCPMS (T, ICKWRK, RCKWRK, CPMS)
C     Returns the specific heats at constant pressure in mass units;
C     see Eq. (26).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CPMS   - Specific heats at constant pressure in mass units
C              for the species.
C                   cgs units - ergs/(gm*K)
C                   Data type - real array
C                   Dimension CPMS(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), CPMS(*), TN(10)
C
#include "ckstrt.h"
C
      TN(1) = 1.0D0
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 240 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  240    CONTINUE
         CPMS(K) = RCKWRK(NcRU) * SUM / RCKWRK(NcWT + K - 1)
C
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCPOR (T, ICKWRK, RCKWRK, CPOR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCPOR (T, ICKWRK, RCKWRK, CPOR)
C     Returns the nondimensional specific heats at constant pressure;
C     see Eq. (19).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CPOR   - Nondimensional specific heats at constant pressure
C              for the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension CPOR(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), TN(10), CPOR(*)
C
#include "ckstrt.h"
C
      TN(1) = 1.0D0
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         CPOR(K) = 0.0D0
         DO 250 N = 1, NCP
            CPOR(K) = CPOR(K) + TN(N)*RCKWRK(NA1 + N - 1)
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCRAY (LINE, NN, KRAY, LOUT, NDIM, NRAY, NF, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCRAY (LINE, NN, KRAY, LOUT, NDIM, NRAY, NF, KERR)
C     This subroutine is called to parse a character string, LINE,
C     that is composed of several blank-delimited substrings.  Each
C     substring in LINE is compared with an ordered reference array
C     of character strings, KRAY(*).  For each substring in LINE that
C     is also an entry in KRAY(*), the index position in KRAY(*) is
C     returned in the integer array, NRAY(*).  It is expected that
C     each substring in LINE will be found in KRAY(*). If a substring
C     cannot be found in KRAY(*) an error flag will be returned. For
C     example, after reading a line of species names, the subroutine
C     might be called to assign Chemkin species index numbers to the
C     list of species names.  This application is illustrated in the
C     following example:
C
C     input:  LINE    = "OH  N2  NO"
C             KRAY(*) = "H2" "O2" "N2" "H" "O" "N" "OH" "H2O" "NO"
C             NN      = 9, the number of entries in KRAY(*)
C             LOUT    = 6, a logical unit number on which to write
C                       diagnostic messages.
C             NDIM    = 10, the dimension of array NRAY(*)
C     output: NRAY(*) = 7, 3, 9, the index numbers of the entries
C                       in KRAY(*) corresponding to the substrings
C                       in LINE
C             NF      = 3, the number of correspondences found.
C             KERR    = .FALSE.
C
C  INPUT
C     LINE - A character string.
C                 Data type - CHARACTER*(*)
C     KRAY - An array of character strings; dimension KRAY(*) at
C                 least NN.
C                 Data type - CHARACTER*(*)
C     NN   - Total number of character strings in KRAY
C                 Data type - integer scalar
C     LOUT - Output unit for printed diagnostics
C                 Data type - integer scalar
C     NDIM - Dimension of the integer array NRAY.
C                 Data type - integer scalar
C
C  OUTPUT
C     NRAY - Index numbers of the elements of KRAY which
C            correspond to the substrings in LINE.
C                 Data type - integer array
C                 Dimension NRAY(*) at least NDIM
C     NF   - Number of correspondences found.
C                 Data type - integer scalar
C     KERR - Error flag; syntax or dimensioning errors will
C            result in KERR=.TRUE.
C                 Data type - logical
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*), KRAY(*)*(*), SUB(80)*80
      DIMENSION NRAY(*)
      LOGICAL KERR, IERR
C
      KERR = .FALSE.
      NF = 0
C
      IDIM = 80
      CALL CKSUBS (LINE, LOUT, IDIM, SUB, NFOUND, IERR)
      IF (IERR) THEN
         KERR = .TRUE.
         WRITE (LOUT,*) ' Error in CKCRAY...'
         RETURN
      ENDIF
C
      DO 50 N = 1, NFOUND
         CALL CKCOMP (SUB(N), KRAY, NN, K)
         IF (K .LE. 0) THEN
            LT = MAX (ILASCH(SUB(N)), 1)
            WRITE (LOUT,'(A)')
     1      ' Error in CKCRAY...'//SUB(N)(:LT)//' not found...'
            KERR = .TRUE.
         ELSE
            IF (NF+1 .GT. NDIM) THEN
               WRITE (LOUT,'(A)')
     1       ' Error in CKCRAY...dimension of NRAY too small...'
               KERR = .TRUE.
            ELSE
               NF = NF + 1
               NRAY(NF) = K
            ENDIF
         ENDIF
   50 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTC  (T, C, ICKWRK, RCKWRK, CDOT, TAU)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTC  (T, C, ICKWRK, RCKWRK, CDOT, TAU)
C     Returns the molar creation rates and characteristic destruction
C     times of the species given temperature and molar concentrations;
C     see Eqs. (76) and (78).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     TAU    - Characteristic destruction times of the species.
C                   cgs units - sec
C                   Data type - real array
C                   Dimension TAU(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), C(*), TAU(*), CDOT(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKCDC (T, C, ICKWRK, RCKWRK, CDOT, RCKWRK(NcK1))
      DO 150 K = 1, NKK
         TAU(K) = C(K) / (RCKWRK(NcK1 + K - 1)+SMALL)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTX  (C, ICKWRK, RCKWRK, X)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTX  (C, ICKWRK, RCKWRK, X)
C     Returns the mole fractions given the molar concentrations;
C     see Eq. (13).
C
C  INPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), C(*), X(*)
C
#include "ckstrt.h"
C
      CTOT = 0.0D0
      DO 100 K = 1, NKK
         CTOT = CTOT + C(K)
  100 CONTINUE
      DO 200 K = 1, NKK
         X(K) = C(K)/CTOT
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTXP (P, T, X, ICKWRK, RCKWRK, CDOT, TAU)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTXP (P, T, X, ICKWRK, RCKWRK, CDOT, TAU)
C     Returns the molar creation rates and characteristic destruction
C     times of the species given the pressure, temperature and mole
C     fractions;  see Eqs. (76) and (78).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     TAU    - Characteristic destruction times of the species.
C                   cgs units - sec
C                   Data type - real array
C                   Dimension TAU(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), TAU(*), CDOT(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKCDXP (P, T, X, ICKWRK, RCKWRK, CDOT, RCKWRK(NcK1))
      CALL CKXTCP (P, T, X, ICKWRK, RCKWRK, RCKWRK(NcK2))
      DO 150 K = 1, NKK
         TAU(K) = RCKWRK(NcK2 + K - 1) / (RCKWRK(NcK1 + K - 1)+SMALL)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTXR (RHO, T, X, ICKWRK, RCKWRK, CDOT, TAU)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTXR (RHO, T, X, ICKWRK, RCKWRK, CDOT, TAU)
C     Returns the molar creation rates and characteristic destruction
C     times of the species given the mass density, temperature and
C     mole fractions;  see Eqs. (76) and (78).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     TAU    - Characteristic destruction times of the species.
C                   cgs units - sec
C                   Data type - real array
C                   Dimension TAU(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), TAU(*), CDOT(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKCDXR (RHO, T, X, ICKWRK, RCKWRK, CDOT, RCKWRK(NcK1))
      CALL CKXTCR (RHO, T, X, ICKWRK, RCKWRK, RCKWRK(NcK2))
      DO 150 K = 1, NKK
         TAU(K) = RCKWRK(NcK2 + K - 1) / (RCKWRK(NcK1 + K - 1)+SMALL)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTY  (C, ICKWRK, RCKWRK, Y)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTY  (C, ICKWRK, RCKWRK, Y)
C     Returns the mass fractions given the molar concentrations;
C     see Eq. (12).
C
C  INPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), C(*), Y(*)
C
#include "ckstrt.h"
C
      RHO = 0.0D0
      DO 100 K = 1, NKK
         RHO = RHO + C(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
C
      DO 200 K = 1, NKK
         Y(K) = C(K)*RCKWRK(NcWT + K - 1)/RHO
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTYP (P, T, Y, ICKWRK, RCKWRK, CDOT, TAU)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTYP (P, T, Y, ICKWRK, RCKWRK, CDOT, TAU)
C     Returns the molar creation rates and characteristic destruction
C     times of the species given the mass density, temperature and
C     mass fractions;  see Eqs. (76) and (78).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     TAU    - Characteristic destruction times of the species.
C                   cgs units - sec
C                   Data type - real array
C                   Dimension TAU(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), TAU(*), CDOT(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKCDYP (P, T, Y, ICKWRK, RCKWRK, CDOT, RCKWRK(NcK1))
      CALL CKYTCP (P, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK2))
      DO 150 K = 1, NKK
         TAU(K) = RCKWRK(NcK2 + K - 1) / (RCKWRK(NcK1 + K - 1)+SMALL)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCTYR (RHO, T, Y, ICKWRK, RCKWRK, CDOT, TAU)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCTYR (RHO, T, Y, ICKWRK, RCKWRK, CDOT, TAU)
C     Returns the molar creation rates and characteristic destruction
C     times of the species given the mass density, temperature and
C     mass fractions;  see Eqs. (76) and (78).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CDOT   - Chemical molar creation rates of the species.
C                   cgs units - mole/(cm**3*sec)
C                   Data type - real array
C                   Dimension CDOT(*) at least KK, the total number of
C                   species.
C     TAU    - Characteristic destruction times of the species.
C                   cgs units - sec
C                   Data type - real array
C                   Dimension TAU(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), TAU(*), CDOT(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKCDYR (RHO, T, Y, ICKWRK, RCKWRK, CDOT, RCKWRK(NcK1))
      CALL CKYTCR (RHO, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK2))
      DO 150 K = 1, NKK
         TAU(K) = RCKWRK(NcK2 + K - 1) / (RCKWRK(NcK1 + K - 1)+SMALL)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCVBL (T, X, ICKWRK, RCKWRK, CVBML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCVBL (T, X, ICKWRK, RCKWRK, CVBML)
C     Returns the mean specific heat at constant volume in molar units;
C     see Eq. (35).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CVBML  - Mean specific heat at constant volume in molar units.
C                   cgs units - ergs/(mole*K)
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*)
C
#include "ckstrt.h"
C
      CALL CKCVML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CVBML = 0.0D0
      DO 100 K = 1, NKK
         CVBML = CVBML + X(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCVBS (T, Y, ICKWRK, RCKWRK, CVBMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCVBS (T, Y, ICKWRK, RCKWRK, CVBMS)
C     Returns the mean specific heat at constant volume in mass units;
C     see Eq. (36).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CVBMS  - Mean specific heat at constant volume in mass units.
C                   cgs units - ergs/(gm*K)
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*)
C
#include "ckstrt.h"
C
      CALL CKCVMS (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CVBMS = 0.0D0
      DO 100 K = 1, NKK
         CVBMS = CVBMS + Y(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCVML (T, ICKWRK, RCKWRK, CVML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCVML (T, ICKWRK, RCKWRK, CVML)
C     Returns the specific heats in constant volume in molar units;
C     see Eq. (22).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CVML   - Specific heats at constant volume in molar units
C              for the species.
C                   cgs units - ergs/(mole*K)
C                   Data type - real array
C                   Dimension CVML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), CVML(*)
C
#include "ckstrt.h"
C
      CALL CKCPML (T, ICKWRK, RCKWRK, CVML)
C
      DO 150 K = 1, NKK
         CVML(K) = CVML(K) - RCKWRK(NcRU)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKCVMS (T, ICKWRK, RCKWRK, CVMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCVMS (T, ICKWRK, RCKWRK, CVMS)
C     Returns the specific heats at constant volume in mass units;
C     see Eq. (29).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     CVMS   - Specific heats at constant volume in mass units
C              for the species.
C                   cgs units - ergs/(gm*K)
C                   Data type - real array
C                   Dimension CVMS(*) at least KK, the total number of
C                   species.
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), CVMS(*)
C
#include "ckstrt.h"
C
      CALL CKCPMS (T, ICKWRK, RCKWRK, CVMS)
C
      DO 150 K = 1, NKK
         CVMS(K) = CVMS(K) - RCKWRK(NcRU) / RCKWRK(NcWT + K - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKEQC  (T, C, ICKWRK, RCKWRK, EQKC)
C
C  START PROLOGUE
C
C  SUBROUTINE CKEQC  (T, C, ICKWRK, RCKWRK, EQKC)
C     Returns the equilibrium constants of the reactions given
C     temperature and molar concentrations;  see Eq. (54).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     EQKC   - Equilibrium constants in concentration units
C              for the reactions.
C                   cgs units - (mole/cm**3)**some power, depending on
C                               the reaction
C                   Data type - real array
C                   Dimension EQKC(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), C(*), EQKC(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 I = 1, NII
         EQKC(I) = RCKWRK(NcI1 + I - 1)
   50 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKEQXP (P, T, X, ICKWRK, RCKWRK, EQKC)
C
C  START PROLOGUE
C
C  SUBROUTINE CKEQXP (P, T, X, ICKWRK, RCKWRK, EQKC)
C     Returns the equilibrium constants for the reactions given
C     pressure, temperature and mole fractions;  see Eq. (54).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     EQKC   - Equilibrium constants in concentration units
C              for the reactions.
C                   cgs units - (mole/cm**3)**some power, depending on
C                               the reaction
C                   Data type - real array
C                   Dimension EQKC(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), EQKC(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 I = 1, NII
         EQKC(I) = RCKWRK(NcI1 + I - 1)
   50 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKEQXR (RHO, T, X, ICKWRK, RCKWRK, EQKC)
C
C  START PROLOGUE
C
C  SUBROUTINE CKEQXR (RHO, T, X, ICKWRK, RCKWRK, EQKC)
C     Returns the equilibrium constants of the reactions given mass
C     density, temperature and mole fractions;  see Eq. (54).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     EQKC   - Equilibrium constants in concentration units
C              for the reactions.
C                   cgs units - (mole/cm**3)**some power, depending on
C                               the reaction
C                   Data type - real array
C                   Dimension EQKC(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), EQKC(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 I = 1, NII
         EQKC(I) = RCKWRK(NcI1 + I - 1)
   50 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKEQYP (P, T, Y, ICKWRK, RCKWRK, EQKC)
C
C  START PROLOGUE
C
C  SUBROUTINE CKEQYP (P, T, Y, ICKWRK, RCKWRK, EQKC)
C     Returns the equilibrium constants for the reactions given
C     pressure, temperature and mass fractions;  see Eq. (54).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     EQKC   - Equilibrium constants in concentration units
C              for the reactions.
C                   cgs units - (mole/cm**3)**some power, depending on
C                               the reaction
C                   Data type - real array
C                   Dimension EQKC(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), EQKC(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 I = 1, NII
         EQKC(I) = RCKWRK(NcI1 + I - 1)
   50 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKEQYR (RHO, T, Y, ICKWRK, RCKWRK, EQKC)
C
C  START PROLOGUE
C
C  SUBROUTINE CKEQYR (RHO, T, Y, ICKWRK, RCKWRK, EQKC)
C     Returns the equilibrium constants of the reactions given mass
C     density, temperature and mass fractions;  see Eq. (54).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     EQKC   - Equilibrium constants in concentration units
C              for the reactions.
C                   cgs units - (mole/cm**3)**some power, depending on
C                               the reaction
C                   Data type - real array
C                   Dimension EQKC(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), EQKC(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 I = 1, NII
         EQKC(I) = RCKWRK(NcI1 + I - 1)
   50 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKFAL  (NDIM, ICKWRK, RCKWRK, IFOP, KFAL, FPAR)
C
C  START PROLOGUE
C
C   SUBROUTINE CKFAL  (NDIM, ICKWRK, RCKWRK, IFOP, KFAL, FPAR)
C     Returns a set of flags indicating whether a reaction has
C     fall-off behavior and an array of the fall-off
C     parameters.
C
C  INPUT
C     NDIM   - First dimension of the two dimensional array FPAR;
C              NDIM must be greater than or equal to the maximum
C              number of fall-off parameters, NFAR, which is
C              currently equal to 8.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     IFOP   - Array of flags indicating fall-off behavior:
C               0 - No fall-off behavior
C               1 - fall-off behavior - Lindeman form (3 parameters)
C               2 - fall-off behavior - SRI form      (8 parameters)
C               3 - fall-off behavior - Troe form     (6 parameters)
C               4 - fall-off behavior - Troe form     (7 parameters)
C                   Data type - integer array
C                   Dimension IFOP(*) at least II, the total number
C                   of reactions.
C     KFAL   - Array of flags indicating type of bath-gas
C              concentration to be used in fall-off expressions
C              (see footnote on page 23).
C               0 - Use total concentration of gas mixture
C                    (with the added capability of using enhanced
C                     third body coefficients) (default)
C               K - Use the concentration of species K
C                   Data type - integer array
C                   Dimension KFAL(*) at least II, the total number
C                   of reactions.
C     FPAR   - Matrix of fall-off parameters. The number of fall-off
C                   parameters will depend on the particular
C                   functional form indicated by the IFOP array:
C                   FPAR(1,I), FPAR(2,I), FPAR(3,I) are always the
C                   parameters entered on the LOW auxiliary keyword
C                   line in the CHEMKIN interpretor input file.
C                     FPAR(1,I) = Pre-exponential for low pressure
C                                 limiting rate constant
C                                 cgs units - mole-cm-sec-K
C                     FPAR(2,I) = Temperature dependence exponents
C                                 for the low pressure limiting rate
C                                 constants.
C                     FPAR(3,I) = Activation energy for the low
C                                 pressure limiting rate constant.
C                                 cgs units - Kelvins
C                   Additional FPAR values depend on IFOP:
C                   IFOP(I) = 2:
C                     FPAR(4,I) = a           (See Eqn. (69))
C                     FPAR(5,I) = b (Kelvin)  (See Eqn. (69))
C                     FPAR(6,I) = c (Kelvin)  (See Eqn. (69))
C                     FPAR(7,I) = d           (See Eqn. (69))
C                     FPAR(8,I) = e           (See Eqn. (69))
C                   IFOP(I) = 3:
C                     FPAR(4,I) = a             (See Eqn. (68))
C                     FPAR(5,I) = T*** (Kelvin) (See Eqn. (68))
C                     FPAR(6,I) = T*   (Kelvin) (See Eqn. (68))
C                   IFOP(I) = 4:
C                     FPAR(4,I) = a             (See Eqn. (68))
C                     FPAR(5,I) = T*** (Kelvin) (See Eqn. (68))
C                     FPAR(6,I) = T*   (Kelvin) (See Eqn. (68))
C                     FPAR(7,I) = T**  (Kelvin) (See Eqn. (68))
C                   Data type - real array
C                   Dimension FPAR(NDIM,*) exactly NDIM (at least NFAR,
C                   the maximum number of fall-off parameters
C                   - currently 8) for the first
C                   dimension and at least II for the second, the total
C                   number of reactions).
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), IFOP(*), KFAL(*), FPAR(NDIM,*)
C
#include "ckstrt.h"
C
      DO 100 I = 1, NII
        IFOP(I) = 0
        KFAL(I) = 0
  100 CONTINUE
C
      DO 150 I = 1, NII
         DO 140 N = 1, NFAR
            FPAR(N,I) = 0.0D0
  140    CONTINUE
  150 CONTINUE
C
      DO 250 N = 1, NFAL
        I       = ICKWRK(IcFL + N - 1)
        IFOP(I) = ICKWRK(IcFO + N - 1)
        KFAL(I) = ICKWRK(IcKF + N - 1)
        IF (IFOP(I) .EQ. 1) THEN
          DO 210 L = 1, 3
            FPAR(L,I) = RCKWRK(NcFL + (N-1)*NFAR + L - 1)
  210     CONTINUE
        ELSE IF (IFOP(I) .EQ. 2) THEN
          DO 220 L = 1, 8
            FPAR(L,I) = RCKWRK(NcFL + (N-1)*NFAR + L - 1)
  220     CONTINUE
        ELSE IF (IFOP(I) .EQ. 3) THEN
          DO 230 L = 1, 6
            FPAR(L,I) = RCKWRK(NcFL + (N-1)*NFAR + L - 1)
  230     CONTINUE
        ELSE IF (IFOP(I) .EQ. 4) THEN
          DO 240 L = 1, 7
            FPAR(L,I) = RCKWRK(NcFL + (N-1)*NFAR + L - 1)
  240     CONTINUE
        ENDIF
  250 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKGBML (P, T, X, ICKWRK, RCKWRK, GBML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKGBML (P, T, X, ICKWRK, RCKWRK, GBML)*
C     Returns the mean Gibbs free energy of the mixture in molar units,
C     given the pressure, temperature and mole fractions;
C     see Eq. (44).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     GBML   - Mean Gibbs free energy in molar units.
C                   cgs units - ergs/mole
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKHML (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
C
      RLNP = RCKWRK(NcRU) * LOG(P / RCKWRK(NcPA))
      GBML = 0.0D0
      DO 100 K = 1, NKK
         GBML = GBML + X(K) * ( RCKWRK(NcK2 + K - 1) - T *
     1          (RCKWRK(NcK1 + K - 1) - RCKWRK(NcRU) *
     2           LOG(MAX(X(K),SMALL)) - RLNP))
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKGBMS (P, T, Y, ICKWRK, RCKWRK, GBMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKGBMS (P, T, Y, ICKWRK, RCKWRK, GBMS)*
C     Returns the mean Gibbs free energy of the mixture in mass units,
C     given the pressure, temperature, and mass fractions;
C     see Eq. (45).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     GBMS   - Mean Gibbs free energy in mass units.
C                   cgs units - ergs/gm
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKHML (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
      CALL CKYTX (Y, ICKWRK, RCKWRK, RCKWRK(NcK3))
C
      RLNP = RCKWRK(NcRU) * LOG(P / RCKWRK(NcPA))
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + RCKWRK(NcK3 + K - 1) *
     1             ( RCKWRK(NcK2 + K - 1) - T *
     2             ( RCKWRK(NcK1 + K - 1) -
     3               RCKWRK(NcRU) *
     4               LOG(MAX(RCKWRK(NcK3 + K - 1),SMALL)) - RLNP))
  100 CONTINUE
C
      CALL CKMMWY (Y, ICKWRK, RCKWRK, WTM)
      GBMS = SUM / WTM
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKGML  (T, ICKWRK, RCKWRK, GML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKGML  (T, ICKWRK, RCKWRK, GML)
C     Returns the standard state Gibbs free energies in molar units;
C     see Eq. (24).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     GML    - Standard state gibbs free energies in molar units
C              for the species.
C                   cgs units - ergs/mole
C                   Data type - real array
C                   Dimension GML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), GML(*)
C
#include "ckstrt.h"
C
      CALL CKHML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
      DO 150 K = 1, NKK
         GML(K) = RCKWRK(NcK1 + K - 1) - T*RCKWRK(NcK2 + K - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKGMS  (T, ICKWRK, RCKWRK, GMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKGMS  (T, ICKWRK, RCKWRK, GMS)
C     Returns the standard state Gibbs free energies in mass units;
C     see Eq. (31).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     GMS    - Standard state Gibbs free energies in mass units
C              for the species.
C                   cgs units - ergs/gm
C                   Data type - real array
C                   Dimension GMS(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), GMS(*)
C
#include "ckstrt.h"
C
      CALL CKHMS (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKSMS (T, ICKWRK, RCKWRK, RCKWRK(NcK2))
      DO 150 K = 1, NKK
         GMS(K) = RCKWRK(NcK1 + K - 1) - T*RCKWRK(NcK2 + K - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKHBML (T, X, ICKWRK, RCKWRK, HBML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKHBML (T, X, ICKWRK, RCKWRK, HBML)
C     Returns the mean enthalpy of the mixture in molar units;
C     see Eq. (37).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     HBML   - Mean enthalpy in molar units.
C                   cgs units - ergs/mole
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*)
C
#include "ckstrt.h"
C
      CALL CKHML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      HBML = 0.0D0
      DO 100 K = 1, NKK
         HBML = HBML + X(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKHBMS (T, Y, ICKWRK, RCKWRK, HBMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKHBMS (T, Y, ICKWRK, RCKWRK, HBMS)
C     Returns the mean enthalpy of the mixture in mass units;
C     see Eq. (38).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     HBMS   - Mean enthalpy in mass units.
C                   cgs units - ergs/gm
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*)
C
#include "ckstrt.h"
C
      CALL CKHMS (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      HBMS = 0.0D0
      DO 100 K = 1, NKK
         HBMS = HBMS + Y(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKHML  (T, ICKWRK, RCKWRK, HML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKHML  (T, ICKWRK, RCKWRK, HML)
C     Returns the enthalpies in molar units
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     HML    - Enthalpies in molar units for the species.
C                   cgs units - ergs/mole
C                   Data type - real array
C                   Dimension HML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), HML(*), TN(10)
C
#include "ckstrt.h"
C
      RUT = T*RCKWRK(NcRU)
      TN(1) = 1.0D0
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)/N
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         HML(K) = RUT * (SUM + RCKWRK(NA1 + NCP1 - 1)/T)
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKHMS  (T, ICKWRK, RCKWRK, HMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKHMS  (T, ICKWRK, RCKWRK, HMS)
C     Returns the enthalpies in mass units;  see Eq. (27).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C  OUTPUT
C     HMS    - Enthalpies in mass units for the species.
C                   cgs units - ergs/gm
C                   Data type - real array
C                   Dimension HMS(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), HMS(*), TN(10)
C
#include "ckstrt.h"
C
      RUT = T*RCKWRK(NcRU)
      TN(1)=1.0D0
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)/N
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         HMS(K) = RUT * (SUM + RCKWRK(NA1 + NCP1 - 1)/T)
     1               / RCKWRK(NcWT + K - 1)
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKHORT (T, ICKWRK, RCKWRK, HORT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKHORT (T, ICKWRK, RCKWRK, HORT)
C     Returns the nondimensional enthalpies;  see Eq. (20).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     HORT   - Nondimensional enthalpies for the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension HORT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION TN(10), HORT(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      DO 150 N = 1, NCP
         TN(N) = T**(N-1)/N
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         HORT(K) = SUM + RCKWRK(NA1 + NCP1 - 1)/T
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKI2CH (NUM, STR, I, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKI2CH (NUM, STR, I, KERR)
C     Returns a character string representation of an integer
C     and the effective length of the string.
C
C  INPUT
C     NUM   - A number to be converted to a character string;
C             the maximum magnitude of NUM is machine-dependent.
C                  Data type - integer scalar.
C
C  OUTPUT
C     STR   - A left-justified character string representing NUM
C                  Data type - CHARACTER*(*)
C     I     - The effective length of the character string
C                  Data type - integer scalar
C     KERR  - Error flag;  character length errors will result in
C             KERR=.TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER STR*(*), IST(10)*(1)
      LOGICAL KERR
      DATA IST/'0','1','2','3','4','5','6','7','8','9'/
      BIGI = 2147483647.D0
C
      I = 0
      STR = ' '
      ILEN = LEN(STR)
      KERR = .FALSE.
      IF (ILEN.LT.1 .OR. IABS(NUM).GT.BIGI) THEN
         KERR = .TRUE.
         RETURN
      ENDIF
C
      IF (NUM .EQ. 0) THEN
         STR = '0'
         I = 1
         RETURN
      ELSEIF (NUM .LT. 0) THEN
         STR(1:) = '-'
      ENDIF
C
      INUM = IABS(NUM)
      NCOL = NINT(LOG10(REAL(INUM))) + 1
C
      DO 10 J = NCOL, 1, -1
         IDIV = INUM / 10.0D0**(J-1)
         IF (J.EQ.NCOL .AND. IDIV.EQ.0) GO TO 10
         LT = ILASCH(STR)
         IF (LT .EQ. ILEN) THEN
            STR = ' '
            KERR = .TRUE.
            RETURN
         ENDIF
         STR(LT+1:) = IST(IDIV+1)
         INUM = INUM - IDIV*10.0D0**(J-1)
   10 CONTINUE
      I = ILASCH(STR)
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKINDX (ICKWRK, RCKWRK, MM, KK, II, NFIT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKINDX (ICKWRK, RCKWRK, MM, KK, II, NFIT)*
C     Returns a group of indices defining the size of the particular
C     reaction mechanism
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     MM     - Total number of elements in mechanism.
C                   Data type - integer scalar
C     KK     - Total number of species in mechanism.
C                   Data type - integer scalar
C     II     - Total number of reactions in mechanism.
C                   Data type - integer scalar
C     NFIT   - number of coefficients in fits to thermodynamic data
C              for one temperature range; NFIT = number of
C              coefficients in polynomial fits to CP/R  +  2.
C                   Data type - integer scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      MM = NMM
      KK = NKK
      II = NII
      NFIT = NCP2
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKINIT (LENIWK, LENRWK, LENCWK, LINC, LOUT, ICKWRK,
     1                   RCKWRK, CCKWRK)
C
C  START PROLOGUE
C
C  SUBROUTINE CKINIT (LENIWK, LENRWK, LENCWK, LINC, LOUT, ICKWRK,
C                     RCKWRK, CCKWRK)*
C     Reads the binary file and creates the internal work arrays
C     ICKWRK, CCKWRK, and RCKWORK.  CKINIT must be called before any
C     other CHEMKIN subroutine is called.  The work arrays must then
C     be made available as input to the other CHEMKIN subroutines.
C
C  INPUT
C     LENIWK - Length of the integer work array, ICKWRK.
C                   Data type - integer scalar
C     LENCWK - Length of the character work array, CCKWRK.
C              The minimum length of CCKWRK(*) is MM + KK.
C                   Data type - integer scalar
C     LENRWK - Length of the real work array, WORK.
C                   Data type - integer scalar
C     LINC  -  Logical file number for the binary file.
C                   Data type - integer scalar
C     LOUT  -  Output file for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C     CCKWRK - Array of character work space.
C                   Data type - CHARACTER*16 array
C                   Dimension CCKWRK(*) at least LENCWK.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*)
      CHARACTER CCKWRK(*)*(*), VERS*16, PREC*16
      LOGICAL IOK, ROK, COK, KERR
      COMMON /CKCONS/ PREC, VERS, KERR, LENI, LENR, LENC
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
      COMMON /CMIN/ CKMIN
C
C     Data about the machine dependent constants is carried in
C
C     COMMON/MACH/SMALL,BIG,EXPARG
C
      DATA RU,RUC,PA /8.314D7, 1.987D0, 1.01325D6/
C
C      THIS STATEMENT WILL NOT COMPILE, MACHINE-DEPENDENT CONSTANTS
C*****exponent range > +/-30
C      SMALL = 1.0E-30
C      BIG   = 1.0E+30
C*****END exponent range > +/-30
C*****exponent range > +/-300
      SMALL = 10.0D0**(-300)
      BIG   = 10.0D0**(+300)
C*****END exponent range > +/-300
      EXPARG = LOG(BIG)
C
C      WRITE (LOUT,15)
   15 FORMAT (/1X,' CKLIB:  Chemical Kinetics Library',
     1        /1X,'         CHEMKIN-II Version 4.9, April 1994',
C*****precision > double
     2        /1X,'         DOUBLE PRECISION')
C*****END precision > double
C*****precision > single
C     2       /1X,'         SINGLE PRECISION')
C*****END precision > single
C
      CALL CKLEN (LINC, LOUT, LI, LR, LC)
C
      IOK = (LENIWK .GE. LI)
      ROK = (LENRWK .GE. LR)
      COK = (LENCWK .GE. LC)
      IF (.NOT.IOK .OR. .NOT.ROK .OR. .NOT.COK) THEN
         IF (.NOT. IOK) WRITE (LOUT, 300) LI
         IF (.NOT. ROK) WRITE (LOUT, 350) LR
         IF (.NOT. COK) WRITE (LOUT, 375) LC
         STOP
      ENDIF
C
      REWIND LINC
      READ (LINC, ERR=110) VERS, PREC, KERR
      READ (LINC, ERR=110) LENI, LENR, LENC, MM, KK, II,
     1                     MAXSP, MAXTB, MAXTP, NTHCF, NIPAR, NITAR,
     2                     NIFAR, NRV, NFL, NTB, NLT, NRL, NW, NCHRG,
     3                     NSTO, NOR, MAXORD, CKMN
C
      IF (LEN(CCKWRK(1)) .LT. 16) THEN
         WRITE (LOUT,475)
         STOP
      ENDIF
C
      NMM = MM
      NKK = KK
      NII = II
      MXSP = MAXSP
      MXTB = MAXTB
      MXTP = MAXTP
      MXOR = MAXORD
      NCP  = NTHCF
      NCP1 = NTHCF+1
      NCP2 = NTHCF+2
      NCP2T = NCP2*(MAXTP-1)
      NPAR = NIPAR
      NLAR = NITAR
      NFAR = NIFAR
      NTHB = NTB
      NLAN = NLT
      NFAL = NFL
      NREV = NRV
      NRLT = NRL
      NWL  = NW
      NRNU= NSTO
      NORD = NOR
      MXORD= MAXORD
      CKMIN= CKMN
C
C             APPORTION work arrays
C
C             SET  ICKWRK(*)=1  TO FLAG THAT CKINIT HAS BEEN CALLED
C
      ICKWRK(1) = 1
C
C             STARTING LOCATIONS OF INTEGER SPACE
C
C! elemental composition of species
      IcNC = 2
C! species phase array
      IcPH = IcNC + KK*MM
C! species charge array
      IcCH = IcPH + KK
C! # of temperatures for fit
      IcNT = IcCH + KK
C! stoichiometric coefficients
      IcNU = IcNT + KK
C! species numbers for the coefficients
      IcNK = IcNU + MAXSP*II
C! # of non-zero coefficients  (<0=reversible, >0=irreversible)
      IcNS = IcNK + MAXSP*II
C! # of reactants
      IcNR = IcNS + II
C! Landau-Teller reaction numbers
      IcLT = IcNR + II
C! Reverse Landau-Teller reactions
      IcRL = IcLT + NLAN
C! Fall-off reaction numbers
      IcFL = IcRL + NRLT
C! Fall-off option numbers
      IcFO = IcFL + NFAL
C! Fall-off enhanced species
      IcKF = IcFO + NFAL
C! Third-body reaction numbers
      IcTB = IcKF + NFAL
C! number of 3rd bodies for above
      IcKN = IcTB + NTHB
C! array of species #'s for above
      IcKT = IcKN + NTHB
C! Reverse parameter reaction numbers
      IcRV = IcKT + MAXTB*NTHB
C! Radiation wavelength reactions
      IcWL = IcRV + NREV
C! Real stoichometry reactions
      IcRNU= IcWL + NWL
C! Change of order reactions
      IcORD= IcRNU + NRNU
C! Species for which there is a change of order
      IcKOR= IcORD + NORD
C
      ITOT = IcKOR + NORD*MXORD - 1
C
C             STARTING LOCATIONS OF CHARACTER SPACE
C
C! start of element names
      IcMM = 1
C! start of species names
      IcKK = IcMM + MM
      ITOC = IcKK + KK - 1
C
C             STARTING LOCATIONS OF REAL SPACE
C
C! atomic weights
      NcAW = 1
C! molecular weights
      NcWT = NcAW + MM
C! temperature fit array for species
      NcTT = NcWT + KK
C! thermodynamic coefficients
      NcAA = NcTT + MAXTP*KK
C! Arrhenius coefficients (3)
      NcCO = NcAA + (MAXTP-1)*NCP2*KK
C! Reverse coefficients
      NcRV = NcCO + (NPAR+1)*II
C! Landau-Teller #'s for NLT reactions
      NcLT = NcRV + (NPAR+1)*NREV
C! Reverse Landau-Teller #'s
      NcRL = NcLT + NLAR*NLAN
C! Fall-off parameters for NFL reactions
      NcFL = NcRL + NLAR*NRLT
C! 3rd body coef'nts for NTHB reactions
      NcKT = NcFL + NFAR*NFAL
C! wavelength
      NcWL = NcKT + MAXTB*NTHB
C! real stoichometric coefficients
      NcRNU= NcWL + NWL
C! change of order for species/reactions
      NcKOR= NcRNU + NRNU*MXSP
C! universal gas constant
      NcRU = NcKOR + NORD*MXORD
C! universal gas constant in units
      NcRC = NcRU + 1
C! pressure of one atmosphere
      NcPA = NcRC + 1
C! intermediate temperature-dependent forward rates
      NcKF = NcPA + 1
C! intermediate temperature-dependent reverse rates
      NcKR = NcKF + II
C! internal work space of length kk
      NcK1 = NcKR + II
C!          'ditto'
      NcK2 = NcK1 + KK
C!          'ditto'
      NcK3 = NcK2 + KK
C!          'ditto'
      NcK4 = NcK3 + KK
      NcI1 = NcK4 + KK
      NcI2 = NcI1 + II
      NcI3 = NcI2 + II
      NcI4 = NcI3 + II
      NTOT = NcI4 + II - 1
C
C        SET UNIVERSAL CONSTANTS IN CGS UNITS
C
      RCKWRK(NcRU) = RU
      RCKWRK(NcRC) = RUC
      RCKWRK(NcPA) = PA
C
C!element names, !atomic weights
      READ (LINC,err=111) (CCKWRK(IcMM+M-1), RCKWRK(NcAW+M-1), M=1,MM)
C
C!species names, !composition, !phase, !charge, !molec weight,
C!# of fit temps, !array of temps, !fit coeff'nts
      READ (LINC,err=222) (CCKWRK(IcKK+K-1),
     1     (ICKWRK(IcNC+(K-1)*MM + M-1),M=1,MM),
     2     ICKWRK(IcPH+K-1),
     3     ICKWRK(IcCH+K-1),
     4     RCKWRK(NcWT+K-1),
     5     ICKWRK(IcNT+K-1),
     6     (RCKWRK(NcTT+(K-1)*MAXTP + L-1),L=1,MAXTP),
     7     ((RCKWRK(NcAA+(L-1)*NCP2+(K-1)*NCP2T+N-1),
     8     N=1,NCP2), L=1,(MAXTP-1)),    K = 1,KK)
C
      IF (II .EQ. 0) RETURN
C
C!# spec,reactants, !Arr. coefficients, !stoic coef, !species numbers
      READ (LINC,end=100,err=333)
     1     (ICKWRK(IcNS+I-1), ICKWRK(IcNR+I-1),
     2      (RCKWRK(NcCO+(I-1)*(NPAR+1)+N-1), N=1,NPAR),
     3      (ICKWRK(IcNU+(I-1)*MAXSP+N-1),
     4       ICKWRK(IcNK+(I-1)*MAXSP+N-1), N=1,MAXSP),
     5      I = 1,II)
C
C     PERTURBATION FACTOR
C
      DO 10 I = 1, II
         RCKWRK(NcCO + (I-1)*(NPAR+1) + NPAR) = 1.0D0
   10 CONTINUE
C
      IF (NREV .GT. 0) READ (LINC,err=444)
     1   (ICKWRK(IcRV+N-1), (RCKWRK(NcRV+(N-1)*(NPAR+1)+L-1),
     1   L=1,NPAR), N = 1,NREV)
C
      IF (NFAL .GT. 0) READ (LINC,err=555)
     1   (ICKWRK(IcFL+N-1), ICKWRK(IcFO+N-1), ICKWRK(IcKF+N-1),
     2   (RCKWRK(NcFL+(N-1)*NFAR+L-1),L=1,NFAR),N=1,NFAL)
C
      IF (NTHB .GT. 0) READ (LINC,err=666)
     1   (ICKWRK(IcTB+N-1), ICKWRK(IcKN+N-1),
     2   (ICKWRK(IcKT+(N-1)*MAXTB+L-1),
     3     RCKWRK(NcKT+(N-1)*MAXTB+L-1),L=1,MAXTB),N=1,NTHB)
C
      IF (NLAN .GT. 0) READ (LINC,err=777)
     1   (ICKWRK(IcLT+N-1), (RCKWRK(NcLT+(N-1)*NLAR+L-1),L=1,NLAR),
     2    N=1,NLAN)
C
      IF (NRLT .GT. 0) READ (LINC,err=888)
     1   (ICKWRK(IcRL+N-1), (RCKWRK(NcRL+(N-1)*NLAR+L-1),L=1,NLAR),
     2    N=1,NRLT)
C
      IF (NWL .GT. 0) READ (LINC,err=999)
     1   (ICKWRK(IcWL+N-1), RCKWRK(NcWL+N-1), N=1,NWL)
C
      IF (NRNU .GT. 0) READ (LINC,err=1111)
     1   (ICKWRK(IcRNU+N-1), (RCKWRK(NcRNU+(N-1)*MAXSP+L-1),L=1,MAXSP),
     2    N=1,NRNU)
C
      IF (NORD .GT. 0) READ (LINC,err=2222)
     1   (ICKWRK(IcORD+N-1), (ICKWRK(IcKOR+(N-1)*MXORD+L-1),
     2                        RCKWRK(NcKOR+(N-1)*MXORD+L-1),
     3   L = 1, MXORD), N=1, NORD)
C
  100 CONTINUE
      RETURN
C
  110 WRITE (LOUT,*) ' Error reading binary file...'
      STOP
  111 WRITE (LOUT,*) ' Error reading element data...'
      STOP
  222 WRITE (LOUT,*) ' Error reading species data...'
      STOP
  333 WRITE (LOUT,*) ' Error reading reaction data...'
      STOP
  444 WRITE (LOUT,*) ' Error reading reverse Arrhenius parameters...'
      STOP
  555 WRITE (LOUT,*) ' Error reading Fall-off data...'
      STOP
  666 WRITE (LOUT,*) ' Error reading third-body data...'
      STOP
  777 WRITE (LOUT,*) ' Error reading Landau-Teller data...'
      STOP
  888 WRITE (LOUT,*) ' Error reading reverse Landau-Teller data...'
      STOP
  999 WRITE (LOUT,*) ' Error reading Wavelength data...'
      STOP
 1111 WRITE (LOUT,*) ' Error reading real stoichometric data...'
      STOP
 2222 WRITE (LOUT,*) ' Error reading order data...'
      STOP
C
  300 FORMAT (10X,'ICKWRK MUST BE DIMENSIONED AT LEAST ',I5)
  350 FORMAT (10X,'RCKWRK MUST BE DIMENSIONED AT LEAST ',I5)
  375 FORMAT (10X,'CCKWRK MUST BE DIMENSIONED AT LEAST ',I5)
  475 FORMAT (10X,'CHARACTER LENGTH OF CCKWRK MUST BE AT LEAST 16 ')
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKITR  (ICKWRK, RCKWRK, ITHB, IREV)
C
C  START PROLOGUE
C
C  SUBROUTINE CKITR  (ICKWRK, RCKWRK, ITHB, IREV)
C     Returns a set of flags indicating whether the reactions are
C     reversible or whether they contain arbitrary third bodies
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     ITHB   - Third-body flags for the reactions;
C              ITHB(I)= -1  reaction I is not a third-body reactions
C              ITHB(I)=  0  reaction I is is a third-body reaction with
C                           no enhanced third body efficiencies
C              ITHB(I)=  N  reaction I is a third-body reaction with
C                           N species enhanced third-body efficiencies.
C                   Data type - integer array
C                   Dimension ITHB(*) at least II, the total number of
C                   reactions.
C
C     IREV   - Reversibility flags and number of species
C              (reactants plus products) for reactions.
C              IREV(I)=+N, reversible reaction I has N species
C              IREV(I)=-N, irreversible reaction I has N species
C                   Data type - integer array
C                   Dimension IREV(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ITHB(*), IREV(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      DO 100 I = 1, NII
         IREV(I) = ICKWRK(IcNS + I - 1)
         ITHB(I) = -1
  100 CONTINUE
      DO 150 N = 1, NTHB
         ITHB(ICKWRK(IcTB + N - 1)) = ICKWRK(IcKN + N - 1)
  150 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKKFKR (P, T, X, ICKWRK, RCKWRK, FWDK, REVK)
C
C  START PROLOGUE
C
C  SUBROUTINE CKKFKR (P, T, X, ICKWRK, RCKWRK, FWDK, REVK)
C     Returns the forward and reverse reaction rates for the
C     reactions given pressure, temperature and mole fractions.
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     FWDK   - Forward reaction rates for the reactions.
C                   cgs units - depends on the reaction
C                   Data type - real array
C                   Dimension FWDK(*) at least II, the total number of
C                   reactios.
C     REVK   - Reverse reaction rates for the reactions.
C                   cgs units - depends on the reaction
C                   Data type - real array
C                   Dimension REVK(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), FWDK(*), REVK(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCP (P, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 200 I = 1, NII
         FWDK(I) = RCKWRK(NcI1 + I - 1)
         REVK(I) = RCKWRK(NcI2 + I - 1)
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKKFRT (P, T, ICKWRK, RCKWRK, RKFT, RKRT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKKFRT (P, T, ICKWRK, RCKWRK, RKFT, RKRT)
C     Returns the forward and reverse reaction rates for the
C     reactions given pressure and temperature.
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     RKFT   - Forward reaction rates for the reactions.
C                   cgs units - depends on the reaction
C                   Data type - real array
C                   Dimension FWDK(*) at least II, the total number of
C                   reactios.
C     RKRT   - Reverse reaction rates for the reactions.
C                   cgs units - depends on the reaction
C                   Data type - real array
C                   Dimension REVK(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), RKFT(*), RKRT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 200 I = 1, NII
         RKFT(I) = RCKWRK(NcKF + I - 1)
         RKRT(I) = RCKWRK(NcKR + I - 1)
  200 CONTINUE
C	
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKLEN (LINC, LOUT, LI, LR, LC)
C
C  START PROLOGUE
C
C  SUBROUTINE CKLEN (LINC, LOUT, LENI, LENR, LENC)
C     Returns the lengths required for the work arrays.
C
C  INPUT
C
C     LINC  -  Logical file number for the binary file.
C                   Data type - integer scalar
C     LOUT  -  Output file for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     LENI  -  Minimum length required for the integer work array.
C                   Data type - integer scalar
C     LENR  -  Minimum length required for the real work array.
C                   Data type - integer scalar
C     LENC  -  Minimum length required for the character work array.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      PARAMETER (NLIST = 3)
      LOGICAL KERR, VOK, POK
      CHARACTER LIST(NLIST)*16, PREC*16, VERS*16
      COMMON /CKCONS/ PREC, VERS, KERR, LENI, LENR, LENC
C      DATA LIST/'1.9','2.0','2.1','2.2','2.3','2.4','2.5','2.6',
C     1          '2.7','2.8','2.9','3.0','3.1','3.2','3.3'/
      DATA LIST /'3.4','3.5','3.6'/
C
      VERS = ' '
      PREC = ' '
      LENI = 0
      LENR = 0
      LENC = 0
C
      KERR = .FALSE.
      REWIND LINC
      READ (LINC, ERR=999) VERS, PREC, KERR
C
      VOK = .FALSE.
      DO 5 N = 1, NLIST
         IF (VERS .EQ. LIST(N)) VOK = .TRUE.
    5 CONTINUE
C
      POK = .FALSE.
C*****precision > double
      IF (INDEX(PREC, 'DOUB') .GT. 0) POK = .TRUE.
C*****END precision > double
C*****precision > single
C      IF (INDEX(PREC, 'SING') .GT. 0) POK = .TRUE.
C*****END precision > single
C
      IF (KERR .OR. (.NOT.POK) .OR. (.NOT.VOK)) THEN
         IF (KERR) THEN
            WRITE (LOUT,'(/A,/A)')
     1      ' There is an error in the Chemkin binary file...',
     2      ' Check CHEMKIN INTERPRETER output for error conditions.'
         ENDIF
         IF (.NOT. VOK) THEN
            WRITE (LOUT,'(/A,A)')
     1      ' Chemkin binary file is incompatible with Chemkin',
     2      ' Library Version 4.9'
         ENDIF
         IF (.NOT. POK) THEN
            WRITE (LOUT,'(/A,A)')
     1      ' Precision of Chemkin binary file does not agree with',
     2      ' precision of Chemkin library'
         ENDIF
         STOP
      ENDIF
C
      READ (LINC, ERR=999) LENICK, LENRCK, LENCCK, MM, KK, II,
     1                     MAXSP, MAXTB, MAXTP, NTHCF, NIPAR, NITAR,
     2                     NIFAR, NRV, NFL, NTB, NLT, NRL, NW, NCHRG,
     3                     NSTO, NOR, MAXORD, CKMN
      REWIND LINC
C
      LENI = LENICK
      LENR = LENRCK
      LENC = LENCCK
      LI   = LENI
      LR   = LENR
      LC   = LENC
      RETURN
C
  999 CONTINUE
      WRITE (LOUT, 50)
   50 FORMAT (' Error reading Chemkin binary file.')
      STOP
      END
C----------------------------------------------------------------------C
C
      SUBROUTINE CKMMWC (C, ICKWRK, RCKWRK, WTM)
C
C  START PROLOGUE
C
C  SUBROUTINE CKMMWC (C, ICKWRK, RCKWRK, WTM)
C     Returns the mean molecular weight of the gas mixture given the
C     molar concentrations;  see Eq. (5).
C
C  INPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WTM    - Mean molecular weight of the species mixture.
C                   cgs units - gm/mole
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION C(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      CTOT = 0.0D0
      DO 100 K = 1, NKK
         CTOT = CTOT + C(K)
  100 CONTINUE
C
      WTM = 0.0D0
      DO 200 K = 1, NKK
         WTM = WTM + C(K)*RCKWRK(NcWT + K - 1)
  200 CONTINUE
      WTM = WTM / CTOT
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKMMWX (X, ICKWRK, RCKWRK, WTM)
C
C  START PROLOGUE
C
C  SUBROUTINE CKMMWX (X, ICKWRK, RCKWRK, WTM)
C     Returns the mean molecular weight of the gas mixture given the
C     mole fractions;  see Eq. (4).
C
C  INPUT
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WTM    - Mean molecular weight of the species mixture.
C                   cgs units - gm/mole
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      WTM = 0.0D0
      DO 100 K = 1, NKK
         WTM = WTM + X(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKMMWY (Y, ICKWRK, RCKWRK, WTM)
C
C  START PROLOGUE
C
C  SUBROUTINE CKMMWY (Y, ICKWRK, RCKWRK, WTM)
C     Returns the mean molecular weight of the gas mixture given the
C     mass fractions;  see Eq. (3).
C
C  INPUT
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WTM    - Mean molecular weight of the species mixture.
C                   cgs units - gm/mole
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      SUMYOW=0.0D0
      DO 150 K = 1, NKK
         SUMYOW = SUMYOW + Y(K)/RCKWRK(NcWT + K - 1)
150   CONTINUE
      WTM = 1.0D0/SUMYOW
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKMXTP (ICKWRK, MAXTP)
C
C  START PROLOGUE
C
C  SUBROUTINE CKMXTP (ICKWRK, MAXTP)
C     Returns the maximum number of temperatures used in
C     fitting the thermodynamic properties of the species.
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C
C  OUTPUT
C     MXTP   - Maximum number of temperatures used in
C              fitting the thermodynamic properties of
C              the species.
C                   Date type - integer scalar
C                   cgs units:  none
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*)
C
#include "ckstrt.h"
C
      MAXTP = MXTP
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKNCF  (MDIM, ICKWRK, RCKWRK, NCF)
C
C  START PROLOGUE
C
C  SUBROUTINE CKNCF  (MDIM, ICKWRK, RCKWRK, NCF)
C     Returns the elemental composition of the species
C
C  INPUT
C     MDIM   - First dimension of the two-dimensional array NCF;
C              MDIM must be equal to or greater than the number of
C              elements, MM.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     NCF    - Matrix of the elemental composition of the species;
C              NCF(M,K) is the number of atoms of the Mth element
C              in the Kth species.
C                   Data type - integer array
C                   Dimension NCF(MDIM,*) exactly MDIM (at least MM,
C                   the total number of elements in the problem) for
C                   the first dimension and at least KK, the total
C                   number of species, for the second.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), NCF(MDIM,*)
C
#include "ckstrt.h"
C
      DO 150 K = 1, NKK
         J = IcNC + (K-1)*NMM
         DO 150 M = 1, NMM
            NCF(M,K) = ICKWRK(J + M - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKNPAR (LINE, NPAR, LOUT, IPAR, ISTART, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKNPAR (LINE, NPAR, LOUT, IPAR, ISTART, KERR)
C     This subroutine is called to parse a character string, LINE,
C     that is composed of several blank-delimited substrings.
C     That final segment of LINE containing NPAR substrings is
C     found, beginning in the ISTART column; this segment is
C     then copied into the character string IPAR.  This allows
C     format-free input of combined alpha-numeric data.
C     For example, after reading a line containing alpha-numeric
C     information ending with several numbers, the subroutine
C     might be called to find the segment of the line containing
C     the numbers:
C
C     input:  LINE*80   = "t1 t2 dt  300.0  3.0E3  50"
C             NPAR      = 3, the number of substrings requested
C             LOUT      = 6, a logical unit number on which to write
C                         diagnostic messages.
C     output: IPAR*80   = "300.0  3.0E3  50"
C             ISTART    = 13, the starting column in LINE of the
C                         NPAR substrings
C             KERR      = .FALSE.
C
C  INPUT
C     LINE   - A character string.
C                   Data type - CHARACTER*(*)
C     NPAR   - Number of substrings expected.
C                   Data type - integer scalar
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     IPAR   - A character string containing only the NPAR substrings.
C                   Data type - CHARACTER*(*)
C     ISTART - The starting location in LINE of the NPAR substrings.
C                   Data type - integer scalar
C     KERR   - Error flag; character length or syntax error will
C              result in KERR = .TRUE.
C                   Date type: logical
C
C  END PROLOGUE
C
C     A '!' will comment out a line, or remainder of the line.
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*), IPAR*(*)
      LOGICAL FOUND, KERR
C
C----------Find Comment String (! signifies comment)
C
      ILEN = IPPLEN(LINE)
      KERR = .FALSE.
C
      IF (ILEN.GT.0) THEN
         FOUND = .FALSE.
         N = 0
         DO 40 I = ILEN, 1, -1
            IF (FOUND) THEN
               IF (LINE(I:I).EQ.' ') THEN
                  N = N+1
                  FOUND = .FALSE.
                  IF (N.EQ.NPAR) THEN
                     ISTART = I+1
                     L1 = ILEN - ISTART + 1
                     L2 = LEN(IPAR)
                     IF (L2 .GE. L1) THEN
                        IPAR = LINE(ISTART:ILEN)
                     ELSE
                        WRITE (LOUT,*)
     1               ' Error in CKNPAR...character length too small...'
                        KERR = .TRUE.
                     ENDIF
                     RETURN
                  ENDIF
               ENDIF
            ELSE
               IF (LINE(I:I).NE.' ') FOUND = .TRUE.
            ENDIF
   40    CONTINUE
      ENDIF
C
      WRITE (LOUT,*) ' Error in CKNPAR...',NPAR,' values not found...'
      KERR = .TRUE.
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKNU   (KDIM, ICKWRK, RCKWRK, NUKI)
C
C  START PROLOGUE
C
C  SUBROUTINE CKNU   (KDIM, ICKWRK, RCKWRK, NUKI)
C     Returns the stoichiometric coefficients of the reaction
C     mechanism;  see Eq. (50).
C
C  INPUT
C     KDIM   - First dimension of the two-dimensional array NUKI;
C              KDIM must be greater than or equal to the total
C              number of species, KK.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     NUKI   - Matrix of stoichiometric coefficients for the species
C              in the reactions;  NUKI(K,I) is the stoichiometric
C              coefficient of species K in reaction I.
C                   Data type - integer array
C                   Dimension NUKI(KDIM,*) exactly KDIM (at least KK,
C                   the total number of species) for the first
C                   dimension and at least II for the second, the total
C                   number of reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), NUKI(KDIM,*)
C
#include "ckstrt.h"
C
      DO 100 I = 1, NII
         DO 100 K = 1, NKK
            NUKI(K,I) = 0
  100 CONTINUE
      DO 200 N = 1, MXSP
         DO 200 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            IF (K .NE. 0) THEN
               NU= ICKWRK(IcNU + (I-1)*MXSP + N -1)
               NUKI(K,I) = NUKI(K,I) + NU
            ENDIF
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKNUF   (KDIM, ICKWRK, RCKWRK, NUFKI)
C
C  START PROLOGUE
C
C  SUBROUTINE CKNUF   (KDIM, ICKWRK, RCKWRK, NUKI)
C     Returns the stoichiometric coefficients for the forward
C     reactions in the reaction mechanism.  All stoichiometric
C     coefficients for reactants are defined to be negative, by
C     definition; see Eq. (50).  Note this subroutine is to be
C     contrasted with the subroutine, CKNU, which returns the net
C     stoichiometric coefficients for a reaction.
C
C  INPUT
C     KDIM   - First dimension of the two-dimensional array NUKI;
C              KDIM must be greater than or equal to the total
C              number of species, KK.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     NUFKI  - Matrix of stoichiometric coefficients for the species
C              in the forward directions of the reactions;
C              NUKI(K,I) is the stoichiometric
C              coefficient of species K in forward direction of
C              reaction I.
C                   Data type - integer array
C                   Dimension NUKI(KDIM,*) exactly KDIM (at least KK,
C                   the total number of species) for the first
C                   dimension and at least II for the second, the total
C                   number of reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), NUFKI(KDIM,*)
C
#include "ckstrt.h"
C
      DO 100 I = 1, NII
         DO 100 K = 1, NKK
            NUFKI(K,I) = 0
  100 CONTINUE
      IF (MXSP .EQ. 6) THEN
        DO 200 I = 1, NII
           KIND = IcNK + (I-1)*MXSP
           NIND = IcNU + (I-1)*MXSP
	   K1 = ICKWRK(KIND)
           K2 = ICKWRK(KIND + 1)
           K3 = ICKWRK(KIND + 2)
	   IF (K1 .NE. 0) THEN
              NU = ICKWRK(NIND)
              NUFKI(K1,I) = NUFKI(K1,I) + NU
           ENDIF
           IF (K2 .NE. 0) THEN
              NU = ICKWRK(NIND + 1)
              NUFKI(K2,I) = NUFKI(K2,I) + NU
           ENDIF
           IF (K3 .NE. 0) THEN
              NU = ICKWRK(NIND + 2)
              NUFKI(K3,I) = NUFKI(K3,I) + NU
           ENDIF
 200    CONTINUE
      ELSE
        DO 300 N = 1, (MXSP/2)
	 DO 300 I = 1, NII
	   K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
           IF (K .NE. 0) THEN
              NU= ICKWRK(IcNU + (I-1)*MXSP + N - 1)
	      NUFKI(K,I) = NUFKI(K,I) + NU
           ENDIF
 300    CONTINUE
      ENDIF
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKPC   (RHO, T, C, ICKWRK, RCKWRK, P)
C
C  START PROLOGUE
C
C  SUBROUTINE CKPC   (RHO, T, C, ICKWRK, RCKWRK, P)
C     Returns the pressure of the gas mixture given the mass density,
C     temperature and molar concentrations;  see Eq. (2).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION C(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      CTOT = 0.0D0
      SUM = 0.0D0
      DO 100 K = 1, NKK
         CTOT = CTOT + C(K)
         SUM  = SUM + C(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
      P    = RHO*RCKWRK(NcRU) * T * CTOT / SUM
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKPHAZ (ICKWRK, RCKWRK, KPHASE)
C
C  START PROLOGUE
C
C  SUBROUTINE CKPHAZ (ICKWRK, RCKWRK, KPHASE)
C     Returns a set of flags indicating phases of the species
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     KPHASE - Phases of the species;
C              KPHASE(K)=-1  the Kth species is solid
C              KPHASE(K)= 0  the Kth species is gaseous
C              KPHASE(K)=+1  the Kth species is liquid
C                   Data type - integer array
C                   Dimension KPHASE(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), KPHASE(*)
C
#include "ckstrt.h"
C
      DO 100 K = 1, NKK
         KPHASE(K) = ICKWRK(IcPH + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKPNT (LSAVE, LOUT, NPOINT, V, P, LI, LR, LC, IERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKPNT (LSAVE, LOUT, NPOINT, VERS, PREC, LENI, LENR,
C                    LENC, KERR)
C     Reads from a binary file information about a Chemkin
C     binary file, pointers for the Chemkin Library, and
C     returns lengths of work arrays.
C
C  INPUT
C     LSAVE  - Integer input unit for binary data file.
C                   Data type - integer scalar
C     LOUT   - Integer output unit for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     NPOINT - Total number of pointers.
C                   Data type - integer scalar
C     VERS   - Version number of the Chemkin binary file.
C                   Data type - real scalar
C     PREC   - Machine precision of the Chemkin binary file.
C                   Data type - character string
C     LENI   - Minimum length required for the integer work array.
C                   Data type - integer scalar
C     LENR   - Minimum length required for the real work array.
C                   Data type - integer scalar
C     LENC   - Minimum length required for the character work array.
C                   Data type - integer scalar
C     KERR   - Logical error flag.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
#include "ckstrt.h"
C
      LOGICAL KERR, IERR
      CHARACTER PREC*16, VERS*16, P*16, V*16
      COMMON /CMIN/ CKMIN
      COMMON /CKCONS/ PREC, VERS, KERR, LENI, LENR, LENC
C
C     Data about the machine dependent constants is carried in
C
      COMMON/MACH/SMALL,BIG,EXPARG
C
C      THIS STATEMENT WILL NOT COMPILE, MACHINE-DEPENDENT CONSTANTS
C*****exponent range > +/-30
C      SMALL = 1.0E-30
C      BIG   = 1.0E+30
C*****END exponent range > +/-30
C*****exponent range > +/-300
      SMALL = 10.0D0**(-300)
      BIG   = 10.0D0**(+300)
C*****END exponent range > +/-300
      EXPARG = LOG(BIG)
C
      KERR = .FALSE.
      READ (LSAVE, ERR=100) NPOINT, VERS, PREC, LENI, LENR, LENC,
     *                NMM , NKK , NII , MXSP, MXTB, MXTP, NCP , NCP1,
     1                NCP2, NCP2T,NPAR, NLAR, NFAR, NLAN, NFAL, NREV,
     2                NTHB, NRLT, NWL,  NRNU, NORD, MXORD,IcMM, IcKK, 
     3                IcNC, IcPH, IcCH, IcNT, IcNU, IcNK, IcNS, IcNR, 
     4                IcLT, IcRL, IcRV, IcWL, IcFL, IcFO, IcKF, IcTB, 
     5                IcKN, IcKT, IcRNU,IcORD,IcKOR,NcAW, NcWT, NcTT, 
     6                NcAA, NcCO, NcRV, NcLT, NcRL, NcFL, NcKT, NcWL, 
     7                NcRU, NcRC, NcPA, NcKF, NcKR, NcRNU,NcKOR,NcK1, 
     8                NcK2, NcK3, NcK4, NcI1, NcI2, NcI3, NcI4, CKMIN
C
      V = VERS
      P = PREC
      LI = LENI
      LR = LENR
      LC = LENC
      IERR = KERR
      RETURN
C
  100 CONTINUE
      WRITE (LOUT, *) ' Error reading Chemkin binary file data...'
      KERR   = .TRUE.
      IERR   = KERR
      NPOINT = 0
      VERS   = ' '
      V      = VERS
      PREC   = ' '
      P      = PREC
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKPX   (RHO, T, X, ICKWRK, RCKWRK, P)
C
C  START PROLOGUE
C
C  SUBROUTINE CKPX   (RHO, T, X, ICKWRK, RCKWRK, P)
C     Returns the pressure of the gas mixture given the mass density,
C     temperature and mole fractions;  see Eq. (*).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + X(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
      P = RHO * RCKWRK(NcRU) * T / SUM
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKPY   (RHO, T, Y, ICKWRK, RCKWRK, P)
C
C  START PROLOGUE
C
C  SUBROUTINE CKPY   (RHO, T, Y, ICKWRK, RCKWRK, P)
C     Returns the pressure of the gas mixture given the mass density,
C     temperature and mass fractions;  see Eq. (*).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      SUMYOW = 0.0D0
      DO 150 K = 1, NKK
         SUMYOW = SUMYOW + Y(K)/RCKWRK(NcWT + K - 1)
150   CONTINUE
      P = RHO * RCKWRK(NcRU) * T * SUMYOW
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKQC   (T, C, ICKWRK, RCKWRK, Q)
C
C  START PROLOGUE
C
C  SUBROUTINE CKQC   (T, C, ICKWRK, RCKWRK, Q)
C     Returns the rates of progress for the reactions given
C     temperature and molar concentrations;  see Eqs. (51) and (58).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Q      - Rates of progress for the reactions.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension Q(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION C(*), ICKWRK(*), RCKWRK(*), Q(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 50 K = 1, NKK
         RCKWRK(NcK1 + K - 1) = C(K)
   50 CONTINUE
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 I = 1, NII
         Q(I) = RCKWRK(NcI1 + I - 1) - RCKWRK(NcI2 + I - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKQXP  (P, T, X, ICKWRK, RCKWRK, Q)
C
C  START PROLOGUE
C
C  SUBROUTINE CKQXP  (P, T, X, ICKWRK, RCKWRK, Q)
C     Returns the rates of progress for the reactions given pressure,
C     temperature and mole fractions;  see Eqs. (51) and (58).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Q      - Rates of progress for the reactions.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension Q(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*), Q(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCP (P, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 I = 1, NII
         Q(I) = RCKWRK(NcI1 + I - 1) - RCKWRK(NcI2 + I - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKQXR  (RHO, T, X, ICKWRK, RCKWRK, Q)
C
C  START PROLOGUE
C
C  SUBROUTINE CKQXR  (RHO, T, X, ICKWRK, RCKWRK, Q)
C     Returns the rates of progress for the reactions given mass
C     density, temperature and mole fractions;  see Eqs. (51) and (58).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Q      - Rates of progress for the reactions.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension Q(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*), Q(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCR (RHO, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 I = 1, NII
         Q(I) = RCKWRK(NcI1 + I - 1) - RCKWRK(NcI2 + I - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKQYP  (P, T, Y, ICKWRK, RCKWRK, Q)
C
C  START PROLOGUE
C
C  SUBROUTINE CKQYP  (P, T, Y, ICKWRK, RCKWRK, Q)
C     Returns the rates of progress for the reactions given pressure,
C     temperature and mass fractions;  see Eqs. (51) and (58).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Q      - Rates of progress for the reactions.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension Q(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*), Q(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKYTCP (P, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 I = 1, NII
         Q(I) = RCKWRK(NcI1 + I - 1) - RCKWRK(NcI2 + I - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKQYR  (RHO, T, Y, ICKWRK, RCKWRK, Q)
C
C  START PROLOGUE
C
C  SUBROUTINE CKQYR  (RHO, T, Y, ICKWRK, RCKWRK, Q)
C     Returns the rates of progress for the reactions given mass
C     density, temperature and mass fractions;  see Eqs. (51) and (58).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Q      - Rates of progress for the reactions.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension Q(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*), Q(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKYTCR (RHO, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 I = 1, NII
         Q(I) = RCKWRK(NcI1 + I - 1) - RCKWRK(NcI2 + I - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKR2CH (RNUM, STR, I, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKR2CH (RNUM, STR, I, KERR)
C     Returns a character string representation of a real number
C     and the effective length of the string.
C
C  INPUT
C     RNUM   - A number to be converted to a string.
C              the maximum magnitude of RNUM is machine-dependent.
C                   Data type - real scalar
C
C  OUTPUT
C     STR   - A left-justified character string representing RNUM,
C             with 5 to 10 characters, depending on the input value.
C             i.e., RNUM=  0.0      returns STR=" 0.00"
C                   RNUM= -10.5     returns STR="-1.05E+01"
C                   RNUM= 1.86E-100 returns in STR=" 1.86E-100"
C                   Data type - CHARACTER*(*);
C                   the minimum length of STR required is 5
C     I     - The effective length of STR
C                   Data type - integer scalar
C     KERR  - Error flag;  character length error will result in
C             KERR=.TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER STR*(*), INUM*3, IEXP*4
      LOGICAL KERR, IERR
C
C*****exponent range > +/-30
C      SMALL = 1.0E-30
C      BIG   = 1.0E+30
C*****END exponent range > +/-30
C*****exponent range > +/-300
      SMALL = 10.0D0**(-300)
      BIG   = 10.0D0**(+300)
C*****END exponent range > +/-300
C
      ILEN = LEN(STR)
      STR = ' '
      KERR = .FALSE.
      I = 0
      IF (ILEN .LT. 5) THEN
         KERR = .TRUE.
         RETURN
      ENDIF
C
      IF (RNUM .EQ. 0.0D0) THEN
         STR = ' 0.00'
         I = 5
         RETURN
      ENDIF
C
C     convert RNUM to a value between 100.0 and 999.0
C
      IF (RNUM.LT.-BIG.OR.RNUM.GT.BIG .OR.
     1   (RNUM.GT.0.0D0 .AND. RNUM.LT.SMALL) .OR.
     2   (RNUM.LT.0.0D0 .AND. RNUM.GT.SMALL)) THEN
         KERR = .TRUE.
         RETURN
      ENDIF
C
      IF (RNUM .LT. 0) THEN
          VAL = -RNUM
      ELSE
          VAL = RNUM
      ENDIF
      IE  = LOG10(VAL)
C
   25 CONTINUE
      IF (IE .LT. 0) THEN
         RVAL = VAL * 10.0D0**(IABS(IE) - 1) * 1000.0D0
      ELSEIF (IE .GT. 0) THEN
         RVAL = VAL * 10.0D0**(-IE + 1) * 10.0D0
      ELSE
         RVAL = VAL * 100.0D0
      ENDIF
      IF (RVAL.LT.100.0D0 .OR. RVAL.GE.1000.0D0) THEN
         IF (RVAL .LT. 100.0D0) IE = IE - 1
         IF (RVAL .GE. 1000.0D0)IE = IE + 1
         GO TO 25
      ELSE
         IVAL = NINT (RVAL)
         IF (IVAL .EQ. 1000) THEN
            IVAL = 100
            IF (IE .LE. 0) THEN
               IE = IE - 1
            ELSE
               IE = IE + 1
            ENDIF
         ENDIF
      ENDIF
C
      CALL CKI2CH (IVAL, INUM, L, IERR)
      LT = 0
      IF (IE.NE.0) THEN
         CALL CKI2CH (IABS(IE), IEXP, LEXP, IERR)
         LT = MAX(LEXP, 2) + 2
      ENDIF
      IERR = IERR.OR.(5+LT .GT. ILEN)
      IF (IERR) THEN
         KERR = .TRUE.
         RETURN
      ENDIF
C
      IF (RNUM .LT. 0.0) STR(1:) = '-'
      STR(2:) = INUM(:1)//'.'//INUM(2:3)
      IF (IE .NE. 0) THEN
         IF (IE .LT. 0) THEN
            STR(6:) = 'E-'
         ELSEIF (IE .GT. 0) THEN
            STR(6:) = 'E+'
         ENDIF
         IF (LEXP .EQ. 1) THEN
            STR(8:) = '0'//IEXP(:1)
         ELSE
            STR(8:) = IEXP(:LEXP)
         ENDIF
      ENDIF
C
      I = ILASCH(STR)
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRAEX (I, RCKWRK, RA)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRAEX (I, RCKWRK, RA)*
C     Get/put the Pre-exponential coefficient of the Ith reaction
C
C  INPUT
C     I      - Reaction number; I > 0 gets RA(I) from RCKWRK
C                               I < 0 puts RA(I) into RCKWRK
C                   Data type - integer scalar
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C     If I < 1:
C     RA     - Pre-exponential coefficient for the Ith reaction.
C                   cgs units - mole-cm-sec-K
C                   Data type - real scalar
C
C  OUTPUT
C     If I > 1:
C     RA     - Pre-exponential coefficient for Ith reaction.
C                   cgs units - mole-cm-sec-K
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION RCKWRK(*)
C
#include "ckstrt.h"
C
      NI = NcCO + (IABS(I)-1)*(NPAR+1)
      IF (I .GT. 0) THEN
         RA = RCKWRK(NI)
      ELSE
         RCKWRK(NI) = RA
      ENDIF
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRAT  (RCKWRK, ICKWRK, II, KK, MAXSP, MAXTB, RU, PATM,
     1                   T, C, NSPEC, NU, NUNK, NPAR, PAR, NREV, IREV,
     2                   RPAR, NFAL, IFAL, IFOP, KFAL, NFAR, FPAR, NLAN,
     3                   NLAR, ILAN, PLT, NRLT, IRLT, RPLT, NTHB, ITHB,
     4                   NTBS, AIK, NKTB, SMH, RKFT, RKRT, RKF, RKR,
     5                   EQK, CTB, NRNU, IRNU, RNU, NORD, IORD, MXORD, 
     6                   KORD, RORD)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRAT  (RCKWRK, ICKWRK, II, KK, MAXSP, MAXTB, RU, PATM,
C 1                   T, C, NSPEC, NU, NUNK, NPAR, PAR, NREV, IREV,
C 2                   RPAR, NFAL, IFAL, IFOP, KFAL, NFAR, FPAR, NLAN,
C 3                   NLAR, ILAN, PLT, NRLT, IRLT, RPLT, NTHB, ITHB,
C 4                   NTBS, AIK, NKTB, SMH, RKFT, RKRT, RKF, RKR, EQK, 
C 5                   EQK, CTB, NRNU, IRNU, RNU, NORD, IORD, MXORD, 
C 6                   KORD, RORD)
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION RCKWRK(*), ICKWRK(*), C(*), NSPEC(*), NU(MAXSP,*),
     1          NUNK(MAXSP,*), PAR(NPAR,*), IREV(*), RPAR(NPAR,*),
     2          ILAN(*), IRLT(*), PLT(NLAR,*), RPLT(NLAR,*),
     3          IFAL(*), IFOP(*), KFAL(*), FPAR(NFAR,*), ITHB(*),
     4          NTBS(*), AIK(MAXTB,*), NKTB(MAXTB,*), SMH(*),
     5          RKFT(*), RKRT(*), RKF(*), RKR(*), EQK(*), CTB(*), 
     6          IRNU(*), RNU(MAXSP,*), IORD(*), KORD(MXORD,*), 
     7          RORD(MXORD,*)
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      ALOGT = LOG(T)
C
      DO 20 I = 1, II
         CTB(I) = 1.0D0
         RKFT(I) = PAR(1,I) * EXP(PAR(2,I)*ALOGT - PAR(3,I)/T)
   20 CONTINUE
C
C        Landau-Teller reactions
C
      DO 25 N = 1, NLAN
         I = ILAN(N)
         TFAC = PLT(1,N)/T**(1.0D0/3.0D0) + PLT(2,N)/T**(2.0D0/3.0D0)
         RKFT(I) = RKFT(I) * EXP(TFAC)
   25 CONTINUE
C
      CALL CKSMH (T, ICKWRK, RCKWRK, SMH)
      DO 50 I = 1, II
          SUMSMH = 0.0
          DO 40 N = 1, MAXSP
             IF (NUNK(N,I).NE.0) SUMSMH=SUMSMH+NU(N,I)*SMH(NUNK(N,I))
   40     CONTINUE
          IF (SUMSMH .NE. 0.0) EQK(I) = EXP(MIN(SUMSMH,EXPARG))
   50 CONTINUE
C
      DO 55 N = 1, NRNU
         SUMSMH = 0.0
         I = IRNU(N)
         DO 45 L = 1, MAXSP
            IF (NUNK(N,I).NE.0) SUMSMH=SUMSMH+RNU(L,N)*SMH(NUNK(N,I))
   45    CONTINUE
         EQK(I) = EXP(MIN(SUMSMH,EXPARG))
   55 CONTINUE
C
      PFAC = PATM / (RU*T)
      DO 60 I = 1, II
         NUSUMK = NU(1,I)+NU(2,I)+NU(3,I)+NU(4,I)+NU(5,I)+NU(6,I)
         EQK(I) = EQK(I) * PFAC**NUSUMK
   60 CONTINUE
      DO 65 N = 1, NRNU
         RNUSUM = RNU(1,N)+RNU(2,N)+RNU(3,N)+RNU(4,N)+RNU(5,N)+RNU(6,N)
         I = IRNU(N)
         PF = PFAC ** RNUSUM
         EQK(I) = EQK(I) * PF
   65 CONTINUE
C
      DO 68 I = 1, II
C
C     RKR=0.0 for irreversible reactions, else RKR=RKF/MAX(EQK,SMALL)
C
         RKRT(I) = 0.0
         IF (NSPEC(I).GT.0) RKRT(I) = RKFT(I) / MAX(EQK(I),SMALL)
   68 CONTINUE
C
C     if reverse parameters have been given:
C
      DO 70 N = 1, NREV
         I = IREV(N)
         RKRT(I) = RPAR(1,N) * EXP(RPAR(2,N)*ALOGT - RPAR(3,N)/T)
         EQK(I) = RKFT(I)/RKRT(I)
   70 CONTINUE
C
C     if reverse Landau-Teller parameters have been given:
C
      DO 75 N = 1, NRLT
         I = IRLT(N)
         TFAC = RPLT(1,N)/T**(1.0D0/3.0D0) + RPLT(2,N)/T**(2.0D0/3.0D0)
         RKRT(I) = RKRT(I) * EXP(TFAC)
         EQK(I) = RKFT(I)/RKRT(I)
   75 CONTINUE
C
C     third-body reactions
C
      CTOT = 0.0D0
      DO 10 K = 1, KK
         CTOT = CTOT + C(K)
   10 CONTINUE
C
      DO 80 N = 1, NTHB
         CTB(ITHB(N)) = CTOT
         DO 80 L = 1, NTBS(N)
            CTB(ITHB(N)) = CTB(ITHB(N)) + (AIK(L,N)-1.0D0)*C(NKTB(L,N))
   80 CONTINUE
C
C     If fall-off (pressure dependence):
C
      DO 90 N = 1, NFAL
C
C        CONCENTRATION OF THIRD BODY
C
         IF (KFAL(N) .EQ. 0) THEN
            CTHB = CTB(IFAL(N))
            CTB(IFAL(N)) = 1.0D0
         ELSE
            CTHB = C(KFAL(N))
         ENDIF
C
         RKLOW = FPAR(1,N) * EXP(FPAR(2,N)*ALOGT - FPAR(3,N)/T)
         PR = RKLOW*CTHB / RKFT(IFAL(N))
         PRLOG = LOG10(MAX(PR,SMALL))
C
         IF (IFOP(N) .EQ. 1) THEN
C
C           LINDEMANN FORM
C
            FC = 1.0
C
         ELSE
C
            IF (IFOP(N) .EQ. 2) THEN
C
C              SRI FORM
C
               XP = 1.0D0/(1.0D0 + PRLOG**2)
               FC = ((FPAR(4,N)*EXP(-FPAR(5,N)/T) + EXP(-T/FPAR(6,N)))
     1              **XP) * FPAR(7,N) * T**FPAR(8,N)
C
            ELSE
C
C              6-PARAMETER TROE FORM
C
               FCENT = (1.0D0-FPAR(4,N)) * EXP(-T/FPAR(5,N))
     1               + FPAR(4,N) * EXP(-T/FPAR(6,N))
C
C              7-PARAMETER TROE FORM
C
               IF (IFOP(N) .EQ. 4) FCENT = FCENT + EXP(-FPAR(7,N)/T)
C
               FCLOG = LOG10(MAX(FCENT,SMALL))
               XN    = 0.75D0 - 1.27D0*FCLOG
               CPRLOG= PRLOG - (0.4D0 + 0.67D0*FCLOG)
               FLOG = FCLOG/(1.0D0 + (CPRLOG/(XN-0.14D0*CPRLOG))**2)
               FC = 10.0D0**FLOG
            ENDIF
         ENDIF
         PCOR = FC * PR/(1.0D0+PR)
         RKFT(IFAL(N)) = RKFT(IFAL(N)) * PCOR
         RKRT(IFAL(N)) = RKRT(IFAL(N)) * PCOR
   90 CONTINUE
C
C     Multiply by the product of reactants and product of products
C     PAR(4,I) is a perturbation factor
C
      DO 150 I = 1, II
         RKFT(I) = RKFT(I) * CTB(I) * PAR(4,I)
         RKRT(I) = RKRT(I) * CTB(I) * PAR(4,I)
         IF (NU(1,I) .NE. 0) THEN
            RKF(I) = RKFT(I)*C(NUNK(1,I))**IABS(NU(1,I))
            RKR(I) = RKRT(I)*C(NUNK(4,I))**NU(4,I)
            IF (NUNK(2,I) .NE. 0) THEN
               RKF(I)= RKF(I) * C(NUNK(2,I))**IABS(NU(2,I))
               IF (NUNK(3,I) .NE. 0)
     1         RKF(I) = RKF(I) * C(NUNK(3,I))**IABS(NU(3,I))
            ENDIF
            IF (NUNK(5,I) .NE. 0) THEN
               RKR(I) = RKR(I) * C(NUNK(5,I))**NU(5,I)
               IF (NUNK(6,I) .NE. 0) RKR(I)=RKR(I)*C(NUNK(6,I))**NU(6,I)
            ENDIF
         ENDIF
  150 CONTINUE
C
      DO 160 N = 1, NRNU
         I = IRNU(N)
         C1 = C(NUNK(1,I)) ** ABS(RNU(1,N))
         C4 = C(NUNK(4,I)) ** RNU(4,N)
         RKF(I) = RKFT(I) * C1
         RKR(I) = RKRT(I) * C4
         IF (NUNK(2,I) .NE. 0) THEN
            C2 = C(NUNK(2,I)) ** ABS(RNU(2,N))
            RKF(I) = RKF(I) * C2
            IF (NUNK(3,I) .NE. 0) THEN
               C3 = C(NUNK(3,I)) ** ABS(RNU(3,N))
               RKF(I) = RKF(I) * C3
            ENDIF
         ENDIF
         IF (NUNK(5,I) .NE. 0) THEN
            C5 = C(NUNK(5,I)) ** RNU(5,N)
            RKR(I) = RKR(I) * C5
            IF (NUNK(6,I) .NE. 0) THEN
               C6 = C(NUNK(6,I)) ** RNU(6,N)
               RKR(I) = RKR(I) * C6
            ENDIF
         ENDIF
  160 CONTINUE
C
      DO 200 N = 1, NORD
         I = IORD(N)
         RKF(I) = RKFT(I)
         RKR(I) = RKRT(I)
C
         DO 190 L = 1, MXORD
            NK = KORD(L,N)
            IF (NK .LT. 0) THEN
               NK = IABS(NK)
               CNK = C(NK) ** RORD(L,N)
               RKF(I) = RKF(I) * CNK
            ELSEIF (NK .GT. 0) THEN
               CNK = C(NK) ** RORD(L,N)
               RKR(I) = RKR(I) * CNK
            ENDIF
  190    CONTINUE
  200 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRATT (RCKWRK, ICKWRK, II, MAXSP, RU, PATM, T, NSPEC,
     1                   NU, NUNK, NPAR, PAR, NREV, IREV, RPAR, NLAN,
     2                   NLAR, ILAN, PLT, NRLT, IRLT, RPLT, SMH, RKFT,
     3                   RKRT, EQK, NRNU, IRNU, RNU)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRATT (RCKWRK, ICKWRK, II, MAXSP, RU, PATM, T, NSPEC,
C 1                   NU, NUNK, NPAR, PAR, NREV, IREV, RPAR, NLAN,
C 2                   NLAR, ILAN, PLT, NRLT, IRLT, RPLT, SMH, RKFT,
C 3                   RKRT, EQK, NRNU, IRNU, RNU)
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION RCKWRK(*), ICKWRK(*), NSPEC(*), NU(MAXSP,*),
     1          NUNK(MAXSP,*), PAR(NPAR,*), IREV(*), RPAR(NPAR,*),
     2          ILAN(*), IRLT(*), PLT(NLAR,*), RPLT(NLAR,*), SMH(*),
     3          RKFT(*), RKRT(*), EQK(*), IRNU(*), RNU(MAXSP,*)
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      ALOGT = LOG(T)
C
      DO 20 I = 1, II
         RKFT(I) = PAR(1,I) * EXP(PAR(2,I)*ALOGT - PAR(3,I)/T)
   20 CONTINUE
C
C        Landau-Teller reactions
C
      DO 25 N = 1, NLAN
         I = ILAN(N)
         TFAC = PLT(1,N)/T**(1.0D0/3.0D0) + PLT(2,N)/T**(2.0D0/3.0D0)
         RKFT(I) = RKFT(I) * EXP(TFAC)
   25 CONTINUE
C
      CALL CKSMH (T, ICKWRK, RCKWRK, SMH)
      DO 50 I = 1, II
          SUMSMH = 0.0D0
          DO 40 N = 1, MAXSP
             IF (NUNK(N,I).NE.0)
C*****precision > double
     1         SUMSMH = SUMSMH + DBLE(NU(N,I))*SMH(NUNK(N,I))
C*****END precision > double
C*****precision > single
C     1           SUMSMH = SUMSMH + REAL(NU(N,I))*SMH(NUNK(N,I))
C*****END precision > single
   40     CONTINUE
          IF (SUMSMH .NE. 0.0D0) EQK(I) = EXP(MIN(SUMSMH,EXPARG))
   50 CONTINUE
C
      DO 55 N = 1, NRNU
         SUMSMH = 0.0D0
         I = IRNU(N)
         DO 45 L = 1, MAXSP
            IF (NUNK(L,I).NE.0) SUMSMH=SUMSMH+RNU(L,N)*SMH(NUNK(L,I))
   45    CONTINUE
         IF (SUMSMH .NE. 0.0D0) EQK(I) = EXP(MIN(SUMSMH,EXPARG))
   55 CONTINUE
C
      PFAC = PATM / (RU*T)
      DO 60 I = 1, II
         NUSUMK = NU(1,I)+NU(2,I)+NU(3,I)+NU(4,I)+NU(5,I)+NU(6,I)
         EQK(I) = EQK(I) * PFAC**NUSUMK
   60 CONTINUE
      DO 65 N = 1, NRNU
         RNUSUM = RNU(1,N)+RNU(2,N)+RNU(3,N)+RNU(4,N)+RNU(5,N)+RNU(6,N)
         I = IRNU(N)
         PFR = PFAC ** RNUSUM
         EQK(I) = EQK(I) * PFR
   65 CONTINUE
C
      DO 68 I = 1, II
C
C     RKR=0.0 for irreversible reactions, else RKR=RKF/MAX(EQK,SMALL)
C
         RKRT(I) = 0.0D0
         IF (NSPEC(I).GT.0) RKRT(I) = RKFT(I) / MAX(EQK(I),SMALL)
   68 CONTINUE
C
C     if reverse parameters have been given:
C
      DO 70 N = 1, NREV
         I = IREV(N)
         RKRT(I) = RPAR(1,N) * EXP(RPAR(2,N)*ALOGT - RPAR(3,N)/T)
         EQK(I)  = RKFT(I)/RKRT(I)
   70 CONTINUE
C
C     if reverse Landau-Teller parameters have been given:
C
      DO 75 N = 1, NRLT
         I = IRLT(N)
         TFAC = RPLT(1,N)/T**(1.0D0/3.0D0) + RPLT(2,N)/T**(2.0D0/3.0D0)
         RKRT(I) = RKRT(I) * EXP(TFAC)
         EQK(I) = RKFT(I)/RKRT(I)
   75 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRATX (II, KK, MAXSP, MAXTB, T, C, NSPEC, NU, NUNK,
     1                   NPAR, PAR, NFAL, IFAL, IFOP, KFAL, NFAR, FPAR, 
     2                   NTHB, ITHB, NTBS, AIK, NKTB, RKFT, RKRT, RKF, 
     3                   RKR, CTB, NRNU, IRNU, RNU, NORD, IORD, MXORD,
     4                   KORD, RORD)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRATX (II, KK, MAXSP, MAXTB, T, C, NSPEC, NU, NUNK,
C 1                   NPAR, PAR, NFAL, IFAL, IFOP, KFAL, NFAR, FPAR, 
C 2                   NTHB, ITHB, NTBS, AIK, NKTB, RKFT, RKRT, RKF, 
C 3                   RKR, CTB, NRNU, IRNU, RNU, NORD, IORD, MXORD,
C 4                   KORD, RORD)
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION C(*), NSPEC(*), NU(MAXSP,*), NUNK(MAXSP,*), PAR(NPAR,*),
     1          IFAL(*), IFOP(*), KFAL(*), FPAR(NFAR,*), ITHB(*),
     2          NTBS(*), AIK(MAXTB,*), NKTB(MAXTB,*), RKFT(*),
     3          RKRT(*), RKF(*), RKR(*), CTB(*), IRNU(*), RNU(MAXSP,*),
     4          IORD(*), KORD(MXORD,*), RORD(MXORD,*)
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      DO 20 I = 1, II
         CTB(I) = 1.0D0
         RKF(I) = 0.0D0
         RKR(I) = 0.0D0
   20 CONTINUE
C
C     third-body reactions
C
      IF (NTHB .GT. 0) THEN
         CTOT = 0.0D0
         DO 10 K = 1, KK
            CTOT = CTOT + C(K)
   10    CONTINUE
         DO 80 N = 1, NTHB
            CTB(ITHB(N)) = CTOT
            DO 80 L = 1, NTBS(N)
               CTB(ITHB(N)) = CTB(ITHB(N))+(AIK(L,N)-1.0D0)*C(NKTB(L,N))
   80    CONTINUE
      ENDIF
C
C     If fall-off (pressure correction):
C
      IF (NFAL .GT. 0) THEN
         ALOGT = LOG(T)
C
         DO 90 N = 1, NFAL
C
            RKLOW = FPAR(1,N) * EXP(FPAR(2,N)*ALOGT - FPAR(3,N)/T)
C
C        CONCENTRATION OF THIRD BODY
C
            IF (KFAL(N) .EQ. 0) THEN
               PR = RKLOW * CTB(IFAL(N)) / RKFT(IFAL(N))
               CTB(IFAL(N)) = 1.0D0
            ELSE
               PR = RKLOW * C(KFAL(N)) / RKFT(IFAL(N))
            ENDIF
C
            PCOR = PR / (1.0D0 + PR)
C
            IF (IFOP(N) .GT. 1) THEN
               PRLOG = LOG10(MAX(PR,SMALL))
C
               IF (IFOP(N) .EQ. 2) THEN
C
C              8-PARAMETER SRI FORM
C
                  XP = 1.0D0/(1.0D0 + PRLOG**2)
                  FC = ((FPAR(4,N)*EXP(-FPAR(5,N)/T) 
     1                   + EXP(-T/FPAR(6,N))) **XP)
     2                  * FPAR(7,N) * T**FPAR(8,N)
C
               ELSE
C
C              6-PARAMETER TROE FORM
C
                  FCENT = (1.0D0-FPAR(4,N)) * EXP(-T/FPAR(5,N))
     1                  +       FPAR(4,N) * EXP(-T/FPAR(6,N))
C
C              7-PARAMETER TROE FORM
C
                  IF (IFOP(N) .EQ. 4) FCENT = FCENT + EXP(-FPAR(7,N)/T)
C
                  FCLOG = LOG10(MAX(FCENT,SMALL))
                  XN    = 0.75D0 - 1.27D0*FCLOG
                  CPRLOG= PRLOG - (0.4D0 + 0.67D0*FCLOG)
                  FLOG = FCLOG/(1.0D0 + (CPRLOG/(XN-0.14D0*CPRLOG))**2)
                  FC = 10.0D0**FLOG
               ENDIF
               PCOR = FC * PCOR
            ENDIF
C
            RKFT(IFAL(N)) = RKFT(IFAL(N)) * PCOR
            RKRT(IFAL(N)) = RKRT(IFAL(N)) * PCOR
   90    CONTINUE
      ENDIF
C
C     Multiply by the product of reactants and product of products
C     PAR(4,I) is a perturbation factor
C
      DO 150 I = 1, II
         RKFT(I) = RKFT(I) * CTB(I) * PAR(4,I)
         RKRT(I) = RKRT(I) * CTB(I) * PAR(4,I)
C
         IF (NU(1,I) .NE. 0) THEN
            RKF(I) = RKFT(I)*C(NUNK(1,I))**IABS(NU(1,I))
            RKR(I) = RKRT(I)*C(NUNK(4,I))**NU(4,I)
            IF (NUNK(2,I) .NE. 0) THEN
               RKF(I)= RKF(I) * C(NUNK(2,I))**IABS(NU(2,I))
               IF (NUNK(3,I) .NE. 0)
     1         RKF(I) = RKF(I) * C(NUNK(3,I))**IABS(NU(3,I))
            ENDIF
            IF (NUNK(5,I) .NE. 0) THEN
               RKR(I) = RKR(I) * C(NUNK(5,I))**NU(5,I)
               IF (NUNK(6,I) .NE. 0) RKR(I)=RKR(I)*C(NUNK(6,I))**NU(6,I)
            ENDIF
         ENDIF
  150 CONTINUE
C
      DO 160 N = 1, NRNU
         I = IRNU(N)
         C1 = C(NUNK(1,I)) ** ABS(RNU(1,N))
         C4 = C(NUNK(4,I)) ** RNU(4,N)
         RKF(I) = RKFT(I) * C1
         RKR(I) = RKRT(I) * C4
         IF (NUNK(2,I) .NE. 0) THEN
            C2 = C(NUNK(2,I)) ** ABS(RNU(2,N))
            RKF(I) = RKF(I) * C2
            IF (NUNK(3,I) .NE. 0) THEN
               C3 = C(NUNK(3,I)) ** ABS(RNU(3,N))
               RKF(I) = RKF(I) * C3
            ENDIF
         ENDIF
         IF (NUNK(5,I) .NE. 0) THEN
            C5 = C(NUNK(5,I)) ** RNU(5,N)
            RKR(I) = RKR(I) * C5
            IF (NUNK(6,I) .NE. 0) THEN
               C6 = C(NUNK(6,I)) ** RNU(6,N)
               RKR(I) = RKR(I) * C6
            ENDIF
         ENDIF
  160 CONTINUE
C
      DO 200 N = 1, NORD
         I = IORD(N)
         RKF(I) = RKFT(I)
         RKR(I) = RKRT(I)
C
         DO 190 L = 1, MXORD
            NK = KORD(L,N)
            IF (NK .LT. 0) THEN
               NK = IABS(NK)
               CNK = C(NK) ** RORD(L,N)
               RKF(I) = RKF(I) * CNK
            ELSEIF (NK .GT. 0) THEN
               CNK = C(NK) ** RORD(L,N)
               RKR(I) = RKR(I) * CNK
            ENDIF
  190    CONTINUE
  200 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRDEX (I, RCKWRK, RD)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRDEX (I, RCKWRK, RD)*
C     Get/put the perturbation factor of the Ith reaction
C
C  INPUT
C     I      - Reaction number; I > 0 gets RD(I) from RCKWRK
C                               I < 0 puts RD(I) into RCKWRK
C                   Data type - integer scalar
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C     If I < 1:
C     RD     - Perturbation factor for the Ith reaction.
C                   cgs units - mole-cm-sec-K
C                   Data type - real scalar
C
C  OUTPUT
C     If I > 1:
C     RD     - Perturbation factor for Ith reaction.
C                   cgs units - mole-cm-sec-K
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION RCKWRK(*)
C
#include "ckstrt.h"
C
      NI = NcCO + (IABS(I)-1)*(NPAR+1) + NPAR
      IF (I .GT. 0) THEN
         RD = RCKWRK(NI)
      ELSE
         RCKWRK(NI) = RD
      ENDIF
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRHEX (K, RCKWRK, A6)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRHEX (K, RCKWRK, A6)
C
C     Returns an array of the sixth thermodynamic polynomial
C     coefficients for a species, or changes their value,
C     depending on the sign of K.
C
C  INPUT
C      K      - Integer species number; K>0 gets A6(*) from RCKWRK,
C                                       K<0 puts A6(*) into RCKWRK.
C                    Data type - integer scalar
C      RCKWRK - Array of real internal work space.
C                    Data type - real array
C
C  OUTPUT
C      A6     - The array of the 6th thermodynamic polynomial
C               coefficients for the Kth species, over the number
C               of temperature ranges used in fitting thermodynamic
C               properties.
C               Dimension A6(*) at least (MXTP-1), where MXTP is
C               the maximum number of temperatures used for fitting
C               the thermodynamic properties of the species.
C                    Data type - real array
C                    cgs units:  none
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION RCKWRK(*), A6(*)
C
#include "ckstrt.h"
C
      DO 100 L = 1, MXTP-1
         NA6 = NCAA + (L-1)*NCP2 + (IABS(K)-1)*NCP2T + NCP
         IF (K .GT. 0) THEN
            A6(L) = RCKWRK(NA6)
         ELSE
            RCKWRK(NA6) = A6(L)
         ENDIF
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRHOC (P, T, C, ICKWRK, RCKWRK, RHO)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRHOC (P, T, C, ICKWRK, RCKWRK, RHO)
C     Returns the mass density of the gas mixture given the pressure,
C     temperature and molar concentrations;  see Eq. (2).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION C(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      CTOT = 0.0D0
      SUM  = 0.0D0
      DO 100 K = 1, NKK
         CTOT = CTOT + C(K)
         SUM = SUM + C(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
C
      RHO  = SUM * P / (RCKWRK(NcRU)*T*CTOT)
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRHOX (P, T, X, ICKWRK, RCKWRK, RHO)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRHOX (P, T, X, ICKWRK, RCKWRK, RHO)
C     Returns the mass density of the gas mixture given the pressure,
C     temperature and mole fractions;  see Eq. (2).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + X(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
C
      RHO = SUM * P / (RCKWRK(NcRU)*T)
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRHOY (P, T, Y, ICKWRK, RCKWRK, RHO)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRHOY (P, T, Y, ICKWRK, RCKWRK, RHO)
C     Returns the mass density of the gas mixture given the pressure,
C     temperature and mass fractions;  see Eq. (2).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
        double precision rckwrk, sumyow, y
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*)
C
#include "ckstrt.h"
C
      SUMYOW = 0.0D0
      DO 150 K = 1, NKK
         SUMYOW = SUMYOW + Y(K)/RCKWRK(NcWT + K - 1)
150   CONTINUE
      RHO = P/(SUMYOW*T*RCKWRK(NcRU))
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRP   (ICKWRK, RCKWRK, RU, RUC, PA)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRP   (ICKWRK, RCKWRK, RU, RUC, PA)
C     Returns universal gas constants and the pressure of one standard
C     atmosphere
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     RU     - Universal gas constant.
C                   cgs units - 8.314E7 ergs/(mole*K)
C                   Data type - real scalar
C     RUC    - Universal gas constant used only in conjuction with
C              activation energy.
C                   preferred units - 1.987 cal/(mole*K)
C                   Data type - real scalar
C     PA     - Pressure of one standard atmosphere.
C                   cgs units - 1.01325E6 dynes/cm**2
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      RU  = RCKWRK(NcRU)
      RUC = RCKWRK(NcRC)
      PA  = RCKWRK(NcPA)
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSAVE (LOUT, LSAVE, ICKWRK, RCKWRK, CCKWRK)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSAVE (LOUT, LSAVE, ICKWRK, RCKWRK, CCKWRK)
C     Writes to a binary file information about a Chemkin
C     binary file, pointers for the Chemkin Library, and
C     Chemkin work arrays.
C
C  INPUT
C     LOUT   - Output file for printed diagnostics.
C                   Data type - integer scalar
C     LSAVE  - Integer output unit.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace containing integer data.
C                   Data type - integer array
C     RCKWRK - Array of real workspace containing real data.
C                   Data type - real array
C     CCKWRK - Array of character workspace containing character data.
C                   Data type - CHARACTER*16 array
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      COMMON /CMIN/ CKMIN
      CHARACTER CCKWRK(*)*(*), VERS*16, PREC*16
      LOGICAL KERR
      COMMON /CKCONS/ PREC, VERS, KERR, LENI, LENR, LENC
C
      NPOINT = 72
      WRITE (LSAVE, ERR=999)
     *                NPOINT, VERS,   PREC,   LENI,   LENR,   LENC,
     *                NMM , NKK , NII , MXSP, MXTB, MXTP, NCP , NCP1,
     1                NCP2, NCP2T,NPAR, NLAR, NFAR, NLAN, NFAL, NREV,
     2                NTHB, NRLT, NWL,  NRNU, NORD, MXORD,IcMM, IcKK, 
     3                IcNC, IcPH, IcCH, IcNT, IcNU, IcNK, IcNS, IcNR, 
     4                IcLT, IcRL, IcRV, IcWL, IcFL, IcFO, IcKF, IcTB, 
     5                IcKN, IcKT, IcRNU,IcORD,IcKOR,NcAW, NcWT, NcTT, 
     6                NcAA, NcCO, NcRV, NcLT, NcRL, NcFL, NcKT, NcWL, 
     7                NcRU, NcRC, NcPA, NcKF, NcKR, NcRNU,NcKOR,NcK1, 
     8                NcK2, NcK3, NcK4, NcI1, NcI2, NcI3, NcI4, CKMIN
      WRITE (LSAVE, ERR=999) (ICKWRK(L), L = 1, LENI)
      WRITE (LSAVE, ERR=999) (RCKWRK(L), L = 1, LENR)
      WRITE (LSAVE, ERR=999) (CCKWRK(L), L = 1, LENC)
      RETURN
C
  999 CONTINUE
      WRITE (LOUT, *)
     1 ' Error writing Chemkin binary file information...'
      KERR = .TRUE.
      RETURN
      END
C---------------------------------------------------------------------
C
      SUBROUTINE CKSBML (P, T, X, ICKWRK, RCKWRK, SBML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSBML (P, T, X, ICKWRK, RCKWRK, SBML)*
C     Returns the mean entropy of the mixture in molar units,
C     given the pressure, temperature and mole fractions;
C     see Eq. (42).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     SBML   - Mean entropy in molar units.
C                   cgs units - ergs/(mole*K)
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      RLNP = RCKWRK(NcRU) * LOG(P / RCKWRK(NcPA))
      SBML = 0.0D0
      DO 100 K = 1, NKK
         SBML = SBML + X(K) * ( RCKWRK(NcK1 + K - 1) -
     1          RCKWRK(NcRU)*LOG(MAX(X(K),SMALL)) - RLNP )
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSBMS (P, T, Y, ICKWRK, RCKWRK, SBMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSBMS (P, T, Y, ICKWRK, RCKWRK, SBMS)*
C     Returns the mean entropy of the mixture in mass units,
C     given the pressure, temperature and mass fractions;
C     see Eq.(43).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     SBMS   - Mean entropy in mass units.
C                   cgs units - ergs/(gm*K)
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      COMMON /MACH/ SMALL,BIG,EXPARG
C
      CALL CKSML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
      CALL CKYTX (Y, ICKWRK, RCKWRK, RCKWRK(NcK2))
C
      RLNP = RCKWRK(NcRU) * LOG (P / RCKWRK(NcPA))
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + RCKWRK(NcK2 + K - 1) *
     1             ( RCKWRK(NcK1 + K - 1)
     2             - RCKWRK(NcRU) *
     3               LOG(MAX(RCKWRK(NcK2 + K - 1),SMALL)) - RLNP)
  100 CONTINUE
C
      CALL CKMMWY (Y, ICKWRK, RCKWRK, WTM)
      SBMS = SUM / WTM
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSMH  (T, ICKWRK, RCKWRK, SMH)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSMH  (T, ICKWRK, RCKWRK, SMH)*
C     Returns the array of entropies minus enthalpies for the species.
C     It is normally not called directly by the user.
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     SMH    - Entropy minus enthalpy for the species,
C              SMH(K) = S(K)/R - H(K)/RT.
C                   cgs units - none
C                   Data type - real array
C                   Dimension SMH(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), SMH(*), TN(10)
C
#include "ckstrt.h"
C
      TN(1) = LOG(T) - 1.0D0
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)/((N-1)*N)
 150  CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         SMH(K) = SUM + RCKWRK(NA1 + NCP2 - 1)
     1                - RCKWRK(NA1 + NCP1 - 1)/T
C
 250  CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSML  (T, ICKWRK, RCKWRK, SML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSML  (T, ICKWRK, RCKWRK, SML)
C     Returns the standard state entropies in molar units
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     SML    - Standard state entropies in molar units for the species.
C                   cgs units - ergs/(mole*K)
C                   Data type - real array
C                   Dimension SML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), SML(*), TN(10)
C
#include "ckstrt.h"
C
      TN(1) = LOG(T)
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)/(N-1)
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         SML(K) = RCKWRK(NcRU) * (SUM + RCKWRK(NA1+NCP2-1))
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSMS  (T, ICKWRK, RCKWRK, SMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSMS  (T, ICKWRK, RCKWRK, SMS)
C     Returns the standard state entropies in mass units;
C     see Eq. (28).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     SMS    - Standard state entropies in mass units for the species.
C                   cgs units - ergs/(gm*K)
C                   Data type - real array
C                   Dimension SMS(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), SMS(*), TN(10)
C
#include "ckstrt.h"
C
      TN(1) = LOG(T)
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)/(N-1)
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         SMS(K) = RCKWRK(NcRU) * (SUM+RCKWRK(NA1 + NCP2 - 1))
     1                         / RCKWRK(NcWT + K - 1)
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSNUM (LINE, NEXP, LOUT, KRAY, NN, KNUM, NVAL,
     1                   RVAL, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSNUM (LINE, NEXP, LOUT, KRAY, NN, KNUM, NVAL,
C                     RVAL, KERR)
C     This subroutine is called to parse a character string, LINE,
C     that is composed of several blank-delimited substrings.
C     It is expected that the first substring in LINE is also an
C     entry in a reference array of character strings, KRAY(*), in
C     which case the index position in KRAY(*) is returned as KNUM,
C     otherwise an error flag is returned.  The substrings following
C     the first are expected to represent numbers, and are converted
C     to elements of the array RVAL(*).  If NEXP substrings are not
C     found an error flag will be returned.  This allows format-free
C     input of combined alpha-numeric data.  For example, after
C     reading a line containing a species name followed by several
C     numerical values, the subroutine might be called to find
C     a Chemkin species index and convert the other substrings to
C     real values:
C
C     input:  LINE    = "N2  1.2"
C             NEXP    = 1, the number of values expected
C             LOUT    = 6, a logical unit number on which to write
C                       diagnostic messages.
C             KRAY(*) = "H2" "O2" "N2" "H" "O" "N" "OH" "H2O" "NO"
C             NN      = 9, the number of entries in KRAY(*)
C     output: KNUM    = 3, the index number of the substring in
C                       KRAY(*) which corresponds to the first
C                       substring in LINE
C             NVAL    = 1, the number of values found in LINE
C                       following the first substring
C             RVAL(*) = 1.200E+00, the substring converted to a number
C             KERR    = .FALSE.
C  INPUT
C     LINE   - A character string.
C                   Data type - CHARACTER*80
C     NEXP   - Number of real values to be found in character string.
C              If NEXP is negative, then IABS(NEXP) values are
C              expected.  However, it is not an error condition,
C              if less values are found.
C                   Data type - integer scalar
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C     KRAY   - Array of character strings.
C                   Data type - CHARACTER*(*)
C     NN     - Total number of character strings in KRAY.
C                   Data type - integer scalar
C
C  OUTPUT
C     KNUM   - Index number of character string in array which
C              corresponds to the first substring in LINE.
C                   Data type - integer scalar
C     NVAL   - Number of real values found in LINE.
C                   Data type - integer scalar
C     RVAL   - Array of real values found in LINE.
C                   Data type - real array
C                   Dimension RVAL(*) at least NEXP
C     KERR   - Error flag; syntax or dimensioning error,
C              corresponding string not found, or total of
C              values found is not the number of values expected,
C              will result in KERR = .TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C     A '!' will comment out a line, or remainder of the line.
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*), KRAY(*)*(*), ISTR*80
      DIMENSION RVAL(*)
      LOGICAL KERR, IERR
C
      NVAL = 0
      KERR = .FALSE.
      ILEN = MIN (IPPLEN(LINE), ILASCH(LINE))
      IF (ILEN .LE. 0) RETURN
C
      I1 = IFIRCH(LINE(:ILEN))
      I3 = INDEX(LINE(I1:ILEN),' ')
      IF (I3 .EQ. 0) I3 = ILEN - I1 + 1
      I2 = I1 + I3
      ISTR = ' '
      ISTR = LINE(I1:I2-1)
C
      CALL CKCOMP (ISTR, KRAY, NN, KNUM)
      IF (KNUM.EQ.0) THEN
         LT = MAX (ILASCH(ISTR), 1)
         WRITE (LOUT,'(A)')
     1   ' Error in CKSNUM...'//ISTR(:LT)//' not found...'
         KERR = .TRUE.
      ENDIF
C
      ISTR = ' '
      ISTR = LINE(I2:ILEN)
      IF (NEXP .NE. 0)
     1      CALL CKXNUM (ISTR, NEXP, LOUT, NVAL, RVAL, IERR)
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSOR  (T, ICKWRK, RCKWRK, SOR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSOR  (T, ICKWRK, RCKWRK, SOR)
C     Returns the nondimensional entropies;  see Eq. (21).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     SOR    - Nondimensional entropies for the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension SOR(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION TN(10), SOR(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      TN(1) = LOG(T)
      DO 150 N = 2, NCP
         TN(N) = T**(N-1)/(N-1)
150   CONTINUE
C
      DO 250 K = 1, NKK
         L = 1
         DO 220 N = 2, ICKWRK(IcNT + K - 1)-1
            TEMP = RCKWRK(NcTT + (K-1)*MXTP + N - 1)
            IF (T .GT. TEMP) L = L+1
 220     CONTINUE
C
         NA1 = NcAA + (L-1)*NCP2 + (K-1)*NCP2T
         SUM = 0.0D0
         DO 225 N = 1, NCP
            SUM = SUM + TN(N)*RCKWRK(NA1 + N - 1)
  225    CONTINUE
         SOR(K) = SUM + RCKWRK(NA1 + NCP2 - 1)
250   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSUBS (LINE, LOUT, NDIM, SUB, NFOUND, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSUBS (LINE, LOUT, NDIM, SUB, NFOUND, KERR)
C     Returns an array of substrings in a character string with blanks
C     as the delimiter
C
C  INPUT
C     LINE   - A character string.
C                   Data type - CHARACTER*(*)
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C     NDIM   - Dimension of array SUB(*)*(*)
C
C  OUTPUT
C     SUB    - The character substrings of LINE.
C                   Data type - CHARACTER*(*) array
C                   Dimension SUB(*) at least NDIM
C     NFOUND - Number of substrings found in LINE.
C                   Data type - integer
C     KERR   - Error flag; dimensioning errors will result in
C              KERR = .TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C     A '!' will comment out a line, or remainder of the line.
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER SUB(*)*(*), LINE*(*)
      LOGICAL KERR
      NFOUND = 0
      ILEN = LEN(SUB(1))
C
      IEND = 0
      KERR = .FALSE.
   25 CONTINUE
C
      ISTART = IEND + 1
      DO 100 L = ISTART, IPPLEN(LINE)
C
         IF (LINE(L:L) .NE. ' ') THEN
            IEND   = INDEX(LINE(L:), ' ')
            IF (IEND .EQ. 0) THEN
               IEND = IPPLEN(LINE)
            ELSE
               IEND = L + IEND - 1
            ENDIF
            IF (IEND-L+1 .GT. ILEN) THEN
               WRITE (LOUT,*) ' Error in CKSUBS...substring too long'
               KERR = .TRUE.
            ELSEIF (NFOUND+1 .GT. NDIM) THEN
               WRITE (LOUT,*) ' Error in CKSUBS...NDIM too small'
               KERR = .TRUE.
            ELSE
               NFOUND = NFOUND + 1
               SUB(NFOUND) = LINE(L:IEND)
            ENDIF
            GO TO 25
         ENDIF
C
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSYME (CCKWRK, LOUT, ENAME, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSYME (CCKWRK, LOUT, ENAME, KERR)*
C     Returns the character strings of element names.
C
C  INPUT
C     CCKWRK - Array of character work space.
C                   Data type - CHARACTER*16 array
C                   Dimension CCKWRK(*) at least LENCWK.
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     ENAME  - Element names.
C                   Data type - CHARACTER*(*) array
C                   Dimension ENAME at least MM, the total number of
C                   elements in the problem.
C     KERR   - Error flag; character length error will result in
C              KERR = .TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER*(*) CCKWRK(*), ENAME(*)
      LOGICAL KERR
C
#include "ckstrt.h"
C
      KERR = .FALSE.
      ILEN = LEN(ENAME(1))
      DO 150 M = 1, NMM
         LT = ILASCH(CCKWRK(IcMM+M-1))
         ENAME(M) = ' '
         IF (LT .LE. ILEN) THEN
            ENAME(M) = CCKWRK(IcMM+M-1)
         ELSE
            WRITE (LOUT,'(A)')
     1      ' Error in CKSYME...character string length too small '
            KERR = .TRUE.
         ENDIF
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSYMR (I, LOUT, ICKWRK, RCKWRK, CCKWRK, LT, ISTR,
     1                   KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSYMR (I, ICKWRK, RCKWRK, CCKWRK, LT, ISTR, KERR)*
C     Returns a character string which describes the Ith reaction,
C     and the effective length of the character string.
C
C  INPUT
C     I      - Reaction index.
C                   Data type - integer scalar
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C     CCKWRK - Array of character work space.
C                   Data type - CHARACTER*16 array
C                   Dimension CCKWRK(*) at least LENCWK.
C
C  OUTPUT
C     ISTR   - Character string describing the Ith reaction.
C                   Data type - CHARACTER*(*)
C     LT     - Number of characters in the reaction description.
C                   Data type - integer scalar
C     KERR   - Error flag;  character length error will result in
C              KERR=.TRUE.
C                    Data type - logical
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*),RCKWRK(*)
      CHARACTER CCKWRK(*)*(*), ISTR*(*), IDUM*80
      LOGICAL KERR, IERR
C
#include "ckstrt.h"
C
      COMMON /CMIN/ CKMIN
C
      ISTR = ' '
      ILEN = LEN(ISTR)
      KERR = .FALSE.
C
      IRNU = 0
      DO 10 N = 1, NRNU
         IF (I .EQ. ICKWRK(IcRNU+N-1)) IRNU = N
   10 CONTINUE
C
      DO 100 J = 1,2
         NS = 0
         DO 50 N = 1, MXSP
            NU = ICKWRK(IcNU + (I-1)*MXSP + N - 1)
            K  = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
C
            IF (IRNU .GT. 0) THEN
               RNU = RCKWRK(NcRNU + (IRNU-1)*MXSP + N - 1)
               IF (ABS(RNU) .GT. CKMIN) THEN
                  IF (J .EQ. 1) THEN
                     NU = -1
                  ELSE
                     NU = 1
                  ENDIF
               ENDIF
            ENDIF
            IF (J.EQ.1.AND.NU.LT.0 .OR. J.EQ.2.AND.NU.GT.0) THEN
               NS = NS + 1
C
               IF (NS .GT. 1) THEN
                  LT = ILASCH(ISTR)
                  IF (LT+1 .GT. ILEN) THEN
                     KERR = .TRUE.
                     ISTR = ' '
                     WRITE (LOUT, 500)
                     RETURN
                  ENDIF
                  ISTR(LT+1:) = '+'
               ENDIF
               IF (IRNU .GT. 0) THEN
                  CALL CKR2CH (ABS(RNU), IDUM, L, IERR)
               ELSE
                  CALL CKI2CH (IABS(NU), IDUM, L, IERR)
               ENDIF
               IF (IERR) THEN
                  KERR = .TRUE.
                  WRITE (LOUT,*) ' Syntax error in CKSYMR...'
                  ISTR = ' '
                  RETURN
               ENDIF
               IF (IABS(NU).NE.1 .OR. 
     1             (IRNU.GT.0 .AND. ABS(RNU).NE.1.0)) THEN
                  LT = ILASCH(ISTR)
                  IF (LT+L .GT. ILEN) THEN
                      KERR = .TRUE.
                      ISTR = ' '
                      WRITE (LOUT, 500)
                      RETURN
                  ENDIF
                  ISTR(LT+1:) = IDUM
               ENDIF
               LK = ILASCH(CCKWRK(IcKK+K-1))
               LT = ILASCH(ISTR)
               IF (LT+LK .GT. ILEN) THEN
                  KERR = .TRUE.
                  ISTR = ' '
                  WRITE (LOUT, 500)
                  RETURN
               ENDIF
               ISTR(LT+1:) = CCKWRK(IcKK+K-1)(:LK)
            ENDIF
   50    CONTINUE
C
         DO 60 N = 1, NFAL
            IF (ICKWRK(IcFL+N-1) .EQ. I) THEN
               LT = ILASCH(ISTR)
               IF (ICKWRK(IcKF+N-1) .EQ. 0) THEN
                  IF (LT+4 .GT. ILEN) THEN
                     KERR = .TRUE.
                     ISTR = ' '
                     WRITE (LOUT, 500)
                     RETURN
                  ENDIF
                  ISTR(LT+1:) = '(+M)'
               ELSE
                  IDUM = ' '
                  IDUM = CCKWRK (IcKK + ICKWRK(IcKF+N-1) - 1)
                  LK = ILASCH(IDUM)
                  IF (LT+LK+3 .GT. ILEN) THEN
                     KERR = .TRUE.
                     ISTR = ' '
                     WRITE (LOUT, 500)
                     RETURN
                  ENDIF
                  ISTR(LT+1:) ='(+'//IDUM(:LK)//')'
               ENDIF
            ENDIF
   60    CONTINUE
C
         DO 70 N = 1, NTHB
            IF (ICKWRK(IcTB+N-1).EQ.I .AND. INDEX(ISTR,'(+M)').LE.0)
     1           THEN
               LT = ILASCH(ISTR)
               IF (LT+2 .GT. ILEN) THEN
                  KERR = .TRUE.
                     ISTR = ' '
                     WRITE (LOUT, 500)
                  RETURN
               ENDIF
               ISTR(LT+1:) = '+M'
            ENDIF
   70    CONTINUE
C
         DO 80 N = 1, NWL
            IF (ICKWRK(IcWL+N-1) .EQ. I) THEN
               W = RCKWRK(NcWL+N-1)
               LT = ILASCH(ISTR)
               IF (LT+3 .GT. ILEN) THEN
                  KERR = .TRUE.
                  ISTR = ' '
                  WRITE (LOUT, 500)
                  RETURN
               ENDIF
               IF (J.EQ.1.AND.W.LT.0.0D0 .OR. J.EQ.2.AND.W.GT.0.0D0)
     1             ISTR(LT+1:) = '+HV'
            ENDIF
   80    CONTINUE
C
         IF (J.EQ.1) THEN
            LT = ILASCH(ISTR)
            IF (ICKWRK(IcNS+I-1) .LT. 0) THEN
               IF (LT+2 .GT. ILEN) THEN
                  KERR = .TRUE.
                  ISTR = ' '
                  WRITE (LOUT, 500)
                  RETURN
               ENDIF
               ISTR(LT+1:) = '=>'
            ELSE
               IF (LT+3 .GT. ILEN) THEN
                  KERR = .TRUE.
                  ISTR = ' '
                  WRITE (LOUT, 500)
                  RETURN
               ENDIF
               ISTR(LT+1:) = '<=>'
            ENDIF
         ENDIF
  100 CONTINUE
      LT = ILASCH(ISTR)
C
  500 FORMAT (' Error in CKSYMR...character string length too small')
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKSYMS (CCKWRK, LOUT, KNAME, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSYMS (CCKWRK, LOUT, KNAME, KERR)*
C     Returns the character strings of species names
C
C  INPUT
C     CCKWRK - Array of character work space.
C                   Data type - CHARACTER*16 array
C                   Dimension CCKWRK(*) at least LENCWK.
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     KNAME  - Species names.
C                   Data type - CHARACTER*(*) array
C                   Dimension KNAME(*) at least KK,
C                   the total number of species.
C     KERR   - Error flag; character length errors will result in
C              KERR = .TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER*(*) CCKWRK(*), KNAME(*)
      LOGICAL KERR
C
#include "ckstrt.h"
C
      KERR = .FALSE.
      ILEN = LEN(KNAME(1))
      DO 150 K = 1, NKK
         LT = ILASCH(CCKWRK(IcKK + K - 1))
         KNAME(K) = ' '
         IF (LT .LE. ILEN) THEN
            KNAME(K) = CCKWRK(IcKK+K-1)
         ELSE
            WRITE (LOUT,*)
     1      ' Error in CKSYM...character string length too small '
            KERR = .TRUE.
         ENDIF
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKTHB  (KDIM, ICKWRK, RCKWRK, AKI)
C
C  START PROLOGUE
C
C  SUBROUTINE CKTHB  (KDIM, ICKWRK, RCKWRK, AKI)
C     Returns matrix of enhanced third body coefficients;
C     see Eq. (58).
C
C  INPUT
C     KDIM   - First dimension of the two dimensional array AKI;
C              KDIM must be greater than or equal to the total
C              number of species, KK.
C                   Data type - integer scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     AKI    - Matrix of enhanced third body efficiencies of the
C              species in the reactions; AKI(K,I) is the enhanced
C              efficiency of the Kth species in the Ith reaction.
C                   Data type - real array
C                   Dimension AKI(KDIM,*) exactly KDIM (at least KK,
C                   the total number of species) for the first
C                   dimension and at least II for the second, the total
C                   number of reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION AKI(KDIM,*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      DO 150 I = 1, NII
         DO 140 K = 1, NKK
            AKI(K,I) = 1.0
  140    CONTINUE
  150 CONTINUE
C
      DO 250 N = 1, NTHB
         I = ICKWRK(IcTB + N - 1)
         DO 250 L = 1, ICKWRK(IcKN + N - 1)
            K  = ICKWRK(IcKT + (N-1)*MXTB + L - 1)
            AK = RCKWRK(NcKT + (N-1)*MXTB + L - 1)
            AKI(K,I) = AK
  250 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKUBML (T, X, ICKWRK, RCKWRK, UBML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKUBML (T, X, ICKWRK, RCKWRK, UBML)
C     Returns the mean internal energy of the mixture in molar units;
C     see Eq. (39).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     UBML   - Mean internal energy in molar units.
C                   cgs units - ergs/mole
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      CALL CKUML (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      UBML = 0.0D0
      DO 100 K = 1, NKK
         UBML = UBML + X(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKUBMS (T, Y, ICKWRK, RCKWRK, UBMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKUBMS (T, Y, ICKWRK, RCKWRK, UBMS)
C     Returns the mean internal energy of the mixture in mass units;
C     see Eq. (40).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     UBMS   - Mean internal energy in mass units.
C                   cgs units - ergs/gm
C                   Data type - real scalar
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      CALL CKUMS (T, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      UBMS = 0.0D0
      DO 100 K = 1, NKK
         UBMS = UBMS + Y(K)*RCKWRK(NcK1 + K - 1)
  100 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKUML  (T, ICKWRK, RCKWRK, UML)
C
C  START PROLOGUE
C
C  SUBROUTINE CKUML  (T, ICKWRK, RCKWRK, UML)
C     Returns the internal energies in molar units;  see Eq. (23).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     UML    - Internal energies in molar units for the species.
C                   cgs units - ergs/mole
C                   Data type - real array
C                   Dimension UML(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), UML(*)
C
#include "ckstrt.h"
C
      CALL CKHML (T, ICKWRK, RCKWRK, UML)
      RUT = T*RCKWRK(NcRU)
      DO 150 K = 1, NKK
         UML(K) = UML(K) - RUT
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKUMS  (T, ICKWRK, RCKWRK, UMS)
C
C  START PROLOGUE
C
C  SUBROUTINE CKUMS  (T, ICKWRK, RCKWRK, UMS)
C     Returns the internal energies in mass units;  see Eq. (30).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     UMS    - Internal energies in mass units for the species.
C                   cgs units - ergs/gm
C                   Data type - real array
C                   Dimension UMS(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), UMS(*)
C
#include "ckstrt.h"
C
      CALL CKHMS (T, ICKWRK, RCKWRK, UMS)
      RUT = T*RCKWRK(NcRU)
      DO 150 K = 1, NKK
         UMS(K) = UMS(K) - RUT/RCKWRK(NcWT+K-1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWL   (ICKWRK, RCKWRK, WL)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWL   (ICKWRK, RCKWRK, WL)
C     Returns a set of flags providing information on the wave length
C     of photon radiation
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WL     - Radiation wavelengths for the reactions.
C              WL(I)= 0.  reaction I does not have radiation as
C                         either a reactant or product
C              WL(I)=-A   reaction I has radiation of wavelength A
C                         as a reactant
C              WL(I)=+A   reaction I has radiation of wavelength A
C                         as a product
C              If A = 1.0 then no wavelength information was given;
C                   cgs units - angstrom
C                   Data type - real array
C                   Dimension WL(*) at least II, the total number of
C                   reactions.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION WL(*), ICKWRK(*), RCKWRK(*)
C
#include "ckstrt.h"
C
      DO 100 I = 1, NII
         WL(I) = 0.0D0
  100 CONTINUE
      DO 150 N = 1, NWL
         WL(ICKWRK(IcWL+N-1)) = RCKWRK(NcWL+N-1)
  150 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWC   (T, C, ICKWRK, RCKWRK, WDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWC   (T, C, ICKWRK, RCKWRK, WDOT)
C     Returns the molar production rates of the species given the
C     temperature and molar concentrations;  see Eq. (49).
C
C  INPUT
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WDOT   - Chemical molar production rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension WDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION C(*), ICKWRK(*), RCKWRK(*), WDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      DO 25 K = 1, NKK
         RCKWRK(NcK1 + K - 1) = C(K)
         WDOT(K) = 0.0D0
   25 CONTINUE
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 100 N = 1, MXSP
         DO 100 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU =
C*****precision > double
     1         DBLE (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > double
C*****precision > single
C     1         REAL (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > single
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  100 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 200 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
         DO 200 N = 1, MXSP
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            IF (K .NE. 0) THEN
               RNU = RCKWRK(NcRNU + (L-1)*MXSP + N - 1)
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWT   (ICKWRK, RCKWRK, WT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWT   (ICKWRK, RCKWRK, WT)
C     Returns the molecular weights of the species
C
C  INPUT
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WT     - Molecular weights of the species.
C                   cgs units - gm/mole
C                   Data type - real array
C                   Dimension WT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), WT(*)
C
#include "ckstrt.h"
C
      DO 100 K = 1, NKK
         WT(K) = RCKWRK(NcWT + K - 1)
  100 CONTINUE
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWXP  (P, T, X, ICKWRK, RCKWRK, WDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWXP  (P, T, X, ICKWRK, RCKWRK, WDOT)
C     Returns the molar production rates of the species given the
C     pressure, temperature and mole fractions;  see Eq. (49).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WDOT   - Chemical molar production rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension WDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*), WDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCP (P, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 50 K = 1, NKK
         WDOT(K) = 0.0D0
   50 CONTINUE
C
      DO 100 N = 1, MXSP
         DO 100 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU =
C*****precision > double
     1         DBLE (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > double
C*****precision > single
C     1         REAL (ICKWRK(IcNU + (I-1)*MXSP _ N - 1))
C*****END precision > single
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  100 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 200 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
         DO 200 N = 1, MXSP
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            IF (K .NE. 0) THEN
               RNU = RCKWRK(NcRNU + (L-1)*MXSP + N - 1)
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWXR  (RHO, T, X, ICKWRK, RCKWRK, WDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWXR  (RHO, T, X, ICKWRK, RCKWRK, WDOT)
C     Returns the molar production rates of the species given the
C     mass density, temperature and mole fractions;  see Eq. (49).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WDOT   - Chemical molar production rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension WDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X(*), ICKWRK(*), RCKWRK(*), WDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKXTCR (RHO, T, X, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 50 K = 1, NKK
         WDOT(K) = 0.0D0
   50 CONTINUE
      DO 100 N = 1, MXSP
         DO 100 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU =
C*****precision > double
     1         DBLE (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > double
C*****precision > single
C     1         REAL (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > single
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  100 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 200 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
         DO 200 N = 1, MXSP
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            IF (K .NE. 0) THEN
               RNU = RCKWRK(NcRNU + (L-1)*MXSP + N - 1)
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWYP  (P, T, Y, ICKWRK, RCKWRK, WDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWYP  (P, T, Y, ICKWRK, RCKWRK, WDOT)
C     Returns the molar production rates of the species given the
C     pressure, temperature and mass fractions;  see Eq. (49).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WDOT   - Chemical molar production rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension WDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), WDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKYTCP (P, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 50 K = 1, NKK
         WDOT(K) = 0.0D0
   50 CONTINUE
      DO 100 N = 1, MXSP
         DO 100 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU =
C*****precision > double
     1         DBLE(ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > double
C*****precision > single
C     1         REAL (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > single
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  100 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 200 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         DO 200 N = 1, MXSP
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU = RCKWRK(NcRNU + (L-1)*MXSP + N - 1)
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWYPK  (P, T, Y, RKFT, RKRT, ICKWRK, RCKWRK, WDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWYPK  (P, T, Y, RKFT, RKRT, ICKWRK, RCKWRK, WDOT)
C     Returns the molar production rates of the species given the
C     pressure, temperature and mass fractions;  see Eq. (49).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     RKFT   - Forward reaction rates for the reactions
C                   cgs units - depends on the reaction
C                   Data type - real array
C                   Dimension RKFT(*) at least II, the total number
C                   of reactions.
C     RKRT   - Referse reaction rates for the reactions
C                   cgs units - depends on the reaction
C                   Data type - real array
C                   Dimension RKRT(*) at least II, the total number
C                   of reactions
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WDOT   - Chemical molar production rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension WDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), RKFT(*), RKRT(*), WDOT(*)
C
#include "ckstrt.h"
C
      CALL CKYTCP (P, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
      DO 25 I = 1, NII
         RCKWRK(NcKF + I - 1) = RKFT(I)
         RCKWRK(NcKR + I - 1) = RKRT(I)
   25 CONTINUE
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 50 K = 1, NKK
         WDOT(K) = 0.0D0
   50 CONTINUE
      DO 100 N = 1, MXSP
         DO 100 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU =
C*****precision > double
     1         DBLE (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > double
C*****precision > single
C     1         REAL (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > single
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  100 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 200 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
         DO 200 N = 1, MXSP
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            IF (K .NE. 0) THEN
               RNU = RCKWRK(NcRNU + (L-1)*MXSP + N - 1)
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKWYR  (RHO, T, Y, ICKWRK, RCKWRK, WDOT)
C
C  START PROLOGUE
C
C  SUBROUTINE CKWYR  (RHO, T, Y, ICKWRK, RCKWRK, WDOT)
C     Returns the molar production rates of the species given the
C     mass density, temperature and mass fractions;  see Eq. (49).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     WDOT   - Chemical molar production rates of the species.
C                   cgs units - moles/(cm**3*sec)
C                   Data type - real array
C                   Dimension WDOT(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION Y(*), ICKWRK(*), RCKWRK(*), WDOT(*)
C
#include "ckstrt.h"
C
      CALL CKRATT (RCKWRK, ICKWRK, NII, MXSP, RCKWRK(NcRU),
     1             RCKWRK(NcPA), T, ICKWRK(IcNS), ICKWRK(IcNU),
     2             ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO), NREV,
     3             ICKWRK(IcRV), RCKWRK(NcRV), NLAN, NLAR, ICKWRK(IcLT),
     4             RCKWRK(NcLT), NRLT, ICKWRK(IcRL), RCKWRK(NcRL),
     5             RCKWRK(NcK1), RCKWRK(NcKF), RCKWRK(NcKR),
     6             RCKWRK(NcI1), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU))
C
      CALL CKYTCR (RHO, T, Y, ICKWRK, RCKWRK, RCKWRK(NcK1))
C
      CALL CKRATX (NII, NKK, MXSP, MXTB, T, RCKWRK(NcK1), ICKWRK(IcNS),
     1             ICKWRK(IcNU), ICKWRK(IcNK), NPAR+1, RCKWRK(NcCO),
     2             NFAL, ICKWRK(IcFL), ICKWRK(IcFO), ICKWRK(IcKF), NFAR, 
     3             RCKWRK(NcFL), NTHB, ICKWRK(IcTB), ICKWRK(IcKN), 
     4             RCKWRK(NcKT), ICKWRK(IcKT), RCKWRK(NcKF), 
     5             RCKWRK(NcKR), RCKWRK(NcI1), RCKWRK(NcI2), 
     6             RCKWRK(NcI3), NRNU, ICKWRK(IcRNU), RCKWRK(NcRNU),
     7             NORD, ICKWRK(IcORD), MXORD, ICKWRK(IcKOR), 
     8             RCKWRK(NcKOR))
C
      DO 50 K = 1, NKK
         WDOT(K) = 0.0D0
   50 CONTINUE
      DO 100 N = 1, MXSP
         DO 100 I = 1, NII
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
            IF (K .NE. 0) THEN
               RNU =
C*****precision > double
     1         DBLE (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > double
C*****precision > single
C     1         REAL (ICKWRK(IcNU + (I-1)*MXSP + N - 1))
C*****END precision > single
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  100 CONTINUE
C
      IF (NRNU .LE. 0) RETURN
      DO 200 L = 1, NRNU
         I = ICKWRK(IcRNU + L - 1)
         ROP = RCKWRK(NcI1+I-1) - RCKWRK(NcI2+I-1)
         DO 200 N = 1, MXSP
            K = ICKWRK(IcNK + (I-1)*MXSP + N - 1)
            IF (K .NE. 0) THEN
               RNU = RCKWRK(NcRNU + (L-1)*MXSP + N - 1)
               WDOT(K) = WDOT(K) + RNU * ROP
            ENDIF
  200 CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKXNUM (LINE, NEXP, LOUT, NVAL, RVAL, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKXNUM (LINE, NEXP, LOUT, NVAL, RVAL, KERR)
C     This subroutine is called to parse a character string, LINE,
C     that is composed of several blank-delimited substrings.
C     Each substring is expected to represent a number, which
C     is converted to entries in the array of real numbers, RVAL(*).
C     NEXP is the number of values expected, and NVAL is the
C     number of values found.  This allows format-free input of
C     numerical data.  For example:
C
C     input:  LINE    = " 0.170E+14 0 47780.0"
C             NEXP    = 3, the number of values requested
C             LOUT    = 6, a logical unit number on which to write
C                       diagnostic messages.
C     output: NVAL    = 3, the number of values found
C             RVAL(*) = 1.700E+13, 0.000E+00, 4.778E+04
C             KERR    = .FALSE.
C
C  INPUT
C     LINE   - A character string.
C                   Data type - CHARACTER*80
C     NEXP   - Number of real values to be found in character string.
C              If NEXP is negative, then IABS(NEXP) values are
C              expected.  However, it is not an error condition,
C              if less values are found.
C                   Data type - integer scalar
C     LOUT   - Output unit for printed diagnostics.
C                   Data type - integer scalar
C
C  OUTPUT
C     NVAL   - Number of real values found in character string.
C                   Data type - integer scalar
C     RVAL   - Array of real values found.
C                   Data type - real array
C                   Dimension RVAL(*) at least NEXP
C     KERR   - Error flag;  syntax or dimensioning error results
C              in KERR = .TRUE.
C                   Data type - logical
C
C  END PROLOGUE
C
C     A '!' will comment out a line, or remainder of the line.
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*), ITEMP*80
      DIMENSION RVAL(*), RTEMP(80)
      LOGICAL KERR
C
C----------Find Comment String (! signifies comment)
C
      ILEN = IPPLEN(LINE)
      NVAL = 0
      KERR = .FALSE.
C
      IF (ILEN .LE. 0) RETURN
      IF (ILEN .GT. 80) THEN
         WRITE (LOUT,*)     ' Error in CKXNUM...line length > 80 '
         WRITE (LOUT,'(A)') LINE
         KERR = .TRUE.
         RETURN
      ENDIF
C
      ITEMP = LINE(:ILEN)
      IF (NEXP .LT. 0) THEN
         CALL IPPARR (ITEMP, -1, NEXP, RTEMP, NVAL, IERR, LOUT)
      ELSE
         CALL IPPARR (ITEMP, -1, -NEXP, RTEMP, NVAL, IERR, LOUT)
         IF (IERR .EQ. 1) THEN
            WRITE (LOUT, *)    ' Syntax errors in CKXNUM...'
            WRITE (LOUT,'(A)') LINE
            KERR = .TRUE.
         ELSEIF (NVAL .NE. NEXP) THEN
            WRITE (LOUT,*) ' Error in CKXNUM...'
            WRITE (LOUT,'(A)') LINE
            KERR = .TRUE.
            WRITE (LOUT,*) NEXP,' values expected, ',
     1                     NVAL,' values found.'
         ENDIF
      ENDIF
      IF (NVAL .LE. IABS(NEXP)) THEN
         DO 20 N = 1, NVAL
            RVAL(N) = RTEMP(N)
   20    CONTINUE
      ENDIF
C
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKXTCP (P, T, X, ICKWRK, RCKWRK, C)
C
C  START PROLOGUE
C
C  SUBROUTINE CKXTCP (P, T, X, ICKWRK, RCKWRK, C)
C     Returns the molar concentrations given the pressure,
C     temperature and mole fractions;  see Eq. (10).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), C(*)
C
#include "ckstrt.h"
C
      PRUT = P/(RCKWRK(NcRU)*T)
      DO 150 K = 1, NKK
         C(K) = X(K)*PRUT
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKXTCR (RHO, T, X, ICKWRK, RCKWRK, C)
C
C  START PROLOGUE
C
C  SUBROUTINE CKXTCR (RHO, T, X, ICKWRK, RCKWRK, C)
C     Returns the molar concentrations given the mass density,
C     temperature and mole fractions;  see Eq. (11).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), C(*)
C
#include "ckstrt.h"
C
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + X(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
      RHOW = RHO / SUM
      DO 200 K = 1, NKK
         C(K) = X(K)*RHOW
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKXTY  (X, ICKWRK, RCKWRK, Y)
C
C  START PROLOGUE
C
C  SUBROUTINE CKXTY  (X, ICKWRK, RCKWRK, Y)
C     Returns the mass fractions given the mole fractions;
C     see Eq. (9).
C
C  INPUT
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), X(*), Y(*)
C
#include "ckstrt.h"
C
      SUM = 0.0D0
      DO 100 K = 1, NKK
         SUM = SUM + X(K)*RCKWRK(NcWT + K - 1)
  100 CONTINUE
C
      DO 200 K = 1, NKK
         Y(K) = X(K)*RCKWRK(NcWT + K - 1)/SUM
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKYTCP (P, T, Y, ICKWRK, RCKWRK, C)
C
C  START PROLOGUE
C
C  SUBROUTINE CKYTCP (P, T, Y, ICKWRK, RCKWRK, C)
C     Returns the molar concentrations given the pressure,
C     temperature and mass fractions;  see Eq. (7).
C
C  INPUT
C     P      - Pressure.
C                   cgs units - dynes/cm**2
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), C(*)
C
#include "ckstrt.h"
C
      SUMYOW = 0.0D0
      DO 150 K = 1, NKK
         SUMYOW = SUMYOW + Y(K) / RCKWRK(NcWT + K - 1)
150   CONTINUE
      SUMYOW = SUMYOW * T * RCKWRK(NcRU)
      DO 200 K = 1, NKK
         C(K) = P * Y(K) / (SUMYOW * RCKWRK(NcWT + K - 1))
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKYTCR (RHO,T, Y, ICKWRK, RCKWRK, C)
C
C  START PROLOGUE
C
C  SUBROUTINE CKYTCR (RHO,T, Y, ICKWRK, RCKWRK, C)
C     Returns the molar concentrations given the mass density,
C     temperature and mass fractions;  see Eq. (8).
C
C  INPUT
C     RHO    - Mass density.
C                   cgs units - gm/cm**3
C                   Data type - real scalar
C     T      - Temperature.
C                   cgs units - K
C                   Data type - real scalar
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     C      - Molar concentrations of the species.
C                   cgs units - mole/cm**3
C                   Data type - real array
C                   Dimension C(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), C(*)
C
#include "ckstrt.h"
C
      DO 150 K = 1, NKK
         C(K) = RHO*Y(K)/RCKWRK(NcWT + K - 1)
150   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKYTX  (Y, ICKWRK, RCKWRK, X)
C
C  START PROLOGUE
C
C  SUBROUTINE CKYTX  (Y, ICKWRK, RCKWRK, X)
C     Returns the mole fractions given the mass fractions;  see Eq. (6).
C
C  INPUT
C     Y      - Mass fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension Y(*) at least KK, the total number of
C                   species.
C     ICKWRK - Array of integer workspace.
C                   Data type - integer array
C                   Dimension ICKWRK(*) at least LENIWK.
C     RCKWRK - Array of real work space.
C                   Data type - real array
C                   Dimension RCKWRK(*) at least LENRWK.
C
C  OUTPUT
C     X      - Mole fractions of the species.
C                   cgs units - none
C                   Data type - real array
C                   Dimension X(*) at least KK, the total number of
C                   species.
C
C  END PROLOGUE
C
C*****precision > double
        IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C        IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION ICKWRK(*), RCKWRK(*), Y(*), X(*)
C
#include "ckstrt.h"
C
      SUMYOW = 0.0D0
      DO 150 K = 1, NKK
         SUMYOW = SUMYOW + Y(K)/RCKWRK(NcWT + K - 1)
150   CONTINUE
      DO 200 K = 1, NKK
         X(K) = Y(K)/(SUMYOW*RCKWRK(NcWT + K - 1))
200   CONTINUE
      RETURN
      END
C
C----------------------------------------------------------------------C
      SUBROUTINE IPPARI(STRING, ICARD, NEXPEC, IVAL, NFOUND, IERR, LOUT)
C   BEGIN PROLOGUE  IPPARI
C   REFER TO  IPGETI
C   DATE WRITTEN  850625   (YYMMDD)
C   REVISION DATE 851725   (YYMMDD)
C   CATEGORY NO.  J3.,J4.,M2.
C   KEYWORDS  PARSE
C   AUTHOR  CLARK,G.L.,GROUP C-3 LOS ALAMOS NAT'L LAB
C   PURPOSE  Parses integer variables from a character variable.  Called
C            by IPGETI, the IOPAK routine used for interactive input.
C   DESCRIPTION
C
C-----------------------------------------------------------------------
C  IPPARI may be used for parsing an input record that contains integer
C  values, but was read into a character variable instead of directly
C  into integer variables.
C  The following benefits are gained by this approach:
C    - specification of only certain elements of the array is allowed,
C      thus letting the others retain default values
C    - variable numbers of values may be input in a record, up to a
C      specified maximum
C    - control remains with the calling program in case of an input
C      error
C    - diagnostics may be printed by IPPARI to indicate the nature
C      of input errors
C
C   The contents of STRING on input indicate which elements of IVAL
C   are to be changed from their entry values, and values to which
C   they should be changed on exit.  Commas and blanks serve as
C   delimiters, but multiple blanks are treated as a single delimeter.
C   Thus, an input record such as:
C     '   1,   2,,40000   , ,60'
C   is interpreted as the following set of instructions by IPGETR:
C
C     (1) set IVAL(1) = 1
C     (2) set IVAL(2) = 2
C     (3) leave IVAL(3) unchanged
C     (4) set IVAL(4) = 40000
C     (5) leave IVAL(5) unchanged
C     (6) set IVAL(6) = 60
C
C   IPPARI will print diagnostics on the default output device, if
C   desired.
C
C   IPPARI is part of IOPAK, and is written in ANSI FORTRAN 77
C
C   Examples:
C
C      Assume IVAL = (0, 0, 0) and NEXPEC = 3 on entry:
C
C   input string           IVAL on exit            IERR    NFOUND
C   -------------          ----------------------  ----    ------
C  '  2 ,   3 45 '         (2, 3, 45)                0       3
C  '2.15,,3'               (2, 0, 3)                 1       0
C  '3X, 25, 2'             (0, 0, 0)                 1       0
C  '10000'                 (10000, 0, 0)             2       1
C
C      Assume IVAL = (0, 0, 0, 0) and NEXPEC = -4 on entry:
C
C   input string           IVAL on exit            IERR    NFOUND
C   -------------          ----------------------  ----    ------
C  '1, 2'                  (1, 2)                    0       2
C  ',,37  400'             (0, 0, 37, 400)           0       4
C  ' 1,,-3,,5'             (1, 0, -3, 0)             3       4
C
C  arguments: (I=input,O=output)
C  -----------------------------
C  STRING (I) - the character string to be parsed.
C
C  ICARD  (I) - data statement number, and error processing flag
C         < 0 : no error messages printed
C         = 0 : print error messages, but not ICARD
C         > 0 : print error messages, and ICARD
C
C  NEXPEC (I) - number of real variables expected to be input.  If
C         < 0, the number is unknown, and any number of values
C         between 0 and abs(nexpec) may be input.  (see NFOUND)
C
C  PROMPT (I) - prompting string, character type.  A question
C         mark will be added to form the prompt at the screen.
C
C  IVAL (I,O) - the integer value or values to be modified.  On entry,
C       the values are printed as defaults.  The formal parameter
C       corresponding to IVAL must be dimensioned at least NEXPEC
C       in the calling program if NEXPEC > 1.
C
C  NFOUND (O) - the number of real values represented in STRING,
C         only in the case that there were as many or less than
C         NEXPEC.
C
C  IERR (O) - error flag:
C       = 0 if no errors found
C       = 1 syntax errors or illegal values found
C       = 2 for too few values found (NFOUND < NEXPEC)
C       = 3 for too many values found (NFOUND > NEXPEC)
C-----------------------------------------------------------------------
C
C   REFERENCES  (NONE)
C   ROUTINES CALLED  IFIRCH,ILASCH
C   END PROLOGUE  IPPARI
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
C
      CHARACTER STRING*(*), ITEMP*80
      DIMENSION IVAL(*)
      CHARACTER *8 FMT(14)
      LOGICAL OKINCR
C
C   FIRST EXECUTABLE STATEMENT  IPPARI
      IERR   = 0
      NFOUND = 0
      NEXP = IABS(NEXPEC)
      IE = ILASCH(STRING)
      IF (IE .EQ. 0) GO TO 500
      NC = 1
C
C--- OKINCR is a flag that indicates it's OK to increment
C--- NFOUND, the index of the array into which the value
C--- should be read.  It is set false when a space follows
C--- an integer value substring, to keep incrementing from
C--- occurring if a comma should be encountered before the
C--- next value.
C
      OKINCR = .TRUE.
C
C--- begin overall loop on characters in string
C
100   CONTINUE
C
      IF (STRING(NC:NC) .EQ. ',') THEN
         IF (OKINCR .OR. NC .EQ. IE) THEN
            NFOUND = NFOUND + 1
         ELSE
            OKINCR = .TRUE.
         ENDIF
C
         GO TO 450
      ENDIF
      IF (STRING(NC:NC) .EQ. ' ') GO TO 450
C
C--- first good character (non-delimeter) found - now find
C--- last good character
C
      IBS = NC
160   CONTINUE
      NC = NC + 1
      IF (NC .GT. IE) GO TO 180
      IF (STRING(NC:NC) .EQ. ' ')THEN
         OKINCR = .FALSE.
      ELSEIF (STRING(NC:NC) .EQ. ',')THEN
         OKINCR = .TRUE.
      ELSE
         GO TO 160
      ENDIF
C
C--- end of substring found - read value into integer array
C
180   CONTINUE
      NFOUND = NFOUND + 1
      IF (NFOUND .GT. NEXP) THEN
         IERR = 3
         GO TO 500
      ENDIF
C
      IES = NC - 1
      NCH = IES - IBS + 1
      DATA FMT/' (I1)', ' (I2)', ' (I3)', ' (I4)', ' (I5)',
     1   ' (I6)', ' (I7)', ' (I8)', ' (I9)', '(I10)',
     2   '(I11)', '(I12)', '(I13)', '(I14)'/
      ITEMP = ' '
      ITEMP = STRING(IBS:IES)
      READ (ITEMP(1:NCH), FMT(NCH), ERR = 400) IVAL(NFOUND)
      GO TO 450
400   CONTINUE
      IERR = 1
      GO TO 510
450   CONTINUE
      NC = NC + 1
      IF (NC .LE. IE) GO TO 100
C
500   CONTINUE
      IF (NEXPEC .GT. 0 .AND. NFOUND .LT. NEXP) IERR = 2
510   CONTINUE
C
      IF (IERR .EQ. 0 .OR. ICARD .LT. 0)RETURN
      IF (ICARD .NE. 0) WRITE(LOUT,'(A,I3)')
     1   '!! ERROR IN DATA STATEMENT NUMBER', ICARD
      IF (IERR .EQ. 1) WRITE(LOUT,'(A)')'SYNTAX ERROR, OR ILLEGAL VALUE'
      IF (IERR .EQ. 2) WRITE(LOUT,'(A,I2, A, I2)')
     1   ' TOO FEW DATA ITEMS.  NUMBER FOUND = ' , NFOUND,
     2   '  NUMBER EXPECTED = ', NEXPEC
      IF (IERR .EQ. 3) WRITE(LOUT,'(A,I2)')
     1   ' TOO MANY DATA ITEMS.  NUMBER EXPECTED = ', NEXPEC
      END
C
C----------------------------------------------------------------------C
C
      SUBROUTINE IPPARR (STRING,ICARD,NEXPEC,RVAL,NFOUND,IERR,LOUT)
C   BEGIN PROLOGUE  IPPARR
C   REFER TO  IPGETR
C   DATE WRITTEN  850625   (YYMMDD)
C   REVISION DATE 851625   (YYMMDD)
C   CATEGORY NO.  J3.,J4.,M2.
C   KEYWORDS  PARSE
C   AUTHOR  CLARK,G.L.,GROUP C-3 LOS ALAMOS NAT'L LAB
C   PURPOSE  Parses real variables from a character variable.  Called
C            by IPGETR, the IOPAK routine used for interactive input.
C   DESCRIPTION
C
C-----------------------------------------------------------------------
C  IPPARR may be used for parsing an input record that contains real
C  values, but was read into a character variable instead of directly
C  into real variables.
C  The following benefits are gained by this approach:
C    - specification of only certain elements of the array is allowed,
C      thus letting the others retain default values
C    - variable numbers of values may be input in a record, up to a
C      specified maximum
C    - control remains with the calling program in case of an input
C      error
C    - diagnostics may be printed by IPPARR to indicate the nature
C      of input errors
C
C   The contents of STRING on input indicate which elements of RVAL
C   are to be changed from their entry values, and values to which
C   they should be changed on exit.  Commas and blanks serve as
C   delimiters, but multiple blanks are treated as a single delimeter.
C   Thus, an input record such as:
C     '   1.,   2,,4.e-5   , ,6.e-6'
C   is interpreted as the following set of instructions by IPGETR:
C
C     (1) set RVAL(1) = 1.0
C     (2) set RVAL(2) = 2.0
C     (3) leave RVAL(3) unchanged
C     (4) set RVAL(4) = 4.0E-05
C     (5) leave RVAL(5) unchanged
C     (6) set RVAL(6) = 6.0E-06
C
C   IPPARR will print diagnostics on the default output device, if
C   desired.
C
C   IPPARR is part of IOPAK, and is written in ANSI FORTRAN 77
C
C   Examples:
C
C      Assume RVAL = (0., 0., 0.) and NEXPEC = 3 on entry:
C
C   input string           RVAL on exit            IERR    NFOUND
C   -------------          ----------------------  ----    ------
C  '  2.34e-3,  3 45.1'    (2.34E-03, 3.0, 45.1)     0       3
C  '2,,3.-5'               (2.0, 0.0, 3.0E-05)       0       3
C  ',1.4,0.028E4'          (0.0, 1.4, 280.0)         0       3
C  '1.0, 2.a4, 3.0'        (1.0, 0.0, 0.0)           1       1
C  '1.0'                   (1.0, 0.0, 0.0)           2       1
C
C      Assume RVAL = (0.,0.,0.,0.) and NEXPEC = -4 on entry:
C
C   input string           RVAL on exit            IERR    NFOUND
C   -------------          ----------------------  ----    ------
C  '1.,2.'                 (1.0, 2.0)                0       2
C  ',,3  4.0'              (0.0, 0.0, 3.0, 4.0)      0       4
C  '1,,3,,5.0'             (0.0, 0.0, 3.0, 0.0)      3       4
C
C  arguments: (I=input,O=output)
C  -----------------------------
C  STRING (I) - the character string to be parsed.
C
C  ICARD  (I) - data statement number, and error processing flag
C         < 0 : no error messages printed
C         = 0 : print error messages, but not ICARD
C         > 0 : print error messages, and ICARD
C
C  NEXPEC (I) - number of real variables expected to be input.  If
C         < 0, the number is unknown, and any number of values
C         between 0 and abs(nexpec) may be input.  (see NFOUND)
C
C  PROMPT (I) - prompting string, character type.  A question
C         mark will be added to form the prompt at the screen.
C
C  RVAL (I,O) - the real value or values to be modified.  On entry,
C       the values are printed as defaults.  The formal parameter
C       corresponding to RVAL must be dimensioned at least NEXPEC
C       in the calling program if NEXPEC > 1.
C
C  NFOUND (O) - the number of real values represented in STRING,
C         only in the case that there were as many or less than
C         NEXPEC.
C
C  IERR (O) - error flag:
C       = 0 if no errors found
C       = 1 syntax errors or illegal values found
C       = 2 for too few values found (NFOUND < NEXPEC)
C       = 3 for too many values found (NFOUND > NEXPEC)
C-----------------------------------------------------------------------
C
C   REFERENCES  (NONE)
C   ROUTINES CALLED  IFIRCH,ILASCH
C   END PROLOGUE  IPPARR
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER STRING*(*), ITEMP*80
      DIMENSION RVAL(*)
      CHARACTER *8 FMT(16)
      LOGICAL OKINCR
C
C   FIRST EXECUTABLE STATEMENT  IPPARR
      IERR   = 0
      NFOUND = 0
      NEXP = IABS(NEXPEC)
      IE = ILASCH(STRING)
      IF (IE .EQ. 0) GO TO 500
      NC = 1
C
C--- OKINCR is a flag that indicates it's OK to increment
C--- NFOUND, the index of the array into which the value
C--- should be read.  It is set negative when a space follows
C--- a real value substring, to keep incrementing from
C--- occurring if a comma should be encountered before the
C--- next value.
C
      OKINCR = .TRUE.
C
C--- begin overall loop on characters in string
C
100   CONTINUE
C
      IF (STRING(NC:NC) .EQ. ',') THEN
         IF (OKINCR) THEN
            NFOUND = NFOUND + 1
         ELSE
            OKINCR = .TRUE.
         ENDIF
C
         GO TO 450
      ENDIF
      IF (STRING(NC:NC) .EQ. ' ') GO TO 450
C
C--- first good character (non-delimeter) found - now find
C--- last good character
C
      IBS = NC
160   CONTINUE
      NC = NC + 1
      IF (NC .GT. IE) GO TO 180
      IF (STRING(NC:NC) .EQ. ' ')THEN
         OKINCR = .FALSE.
      ELSEIF (STRING(NC:NC) .EQ. ',')THEN
         OKINCR = .TRUE.
      ELSE
         GO TO 160
      ENDIF
C
C--- end of substring found - read value into real array
C
180   CONTINUE
      NFOUND = NFOUND + 1
      IF (NFOUND .GT. NEXP) THEN
         IERR = 3
         GO TO 500
      ENDIF
C
      DATA FMT/     ' (E1.0)', ' (E2.0)', ' (E3.0)', ' (E4.0)',
     1   ' (E5.0)', ' (E6.0)', ' (E7.0)', ' (E8.0)', ' (E9.0)',
     2   '(E10.0)', '(E11.0)', '(E12.0)', '(E13.0)', '(E14.0)',
     3   '(E15.0)', '(E16.0)'/
      IES = NC - 1
      NCH = IES - IBS + 1
      ITEMP = ' '
      ITEMP = STRING(IBS:IES)
      READ (ITEMP(:NCH), FMT(NCH), ERR = 400) RVAL(NFOUND)
      GO TO 450
400   CONTINUE
      IERR = 1
      GO TO 510
450   CONTINUE
      NC = NC + 1
      IF (NC .LE. IE) GO TO 100
C
500   CONTINUE
      IF (NEXPEC .GT. 0 .AND. NFOUND .LT. NEXP) IERR = 2
510   CONTINUE
C
      IF (IERR .EQ. 0 .OR. ICARD .LT. 0) RETURN
      IF (ICARD .NE. 0) WRITE(LOUT,'(A,I3)')
     1   '!! ERROR IN DATA STATEMENT NUMBER', ICARD
      IF (IERR .EQ. 1) WRITE(LOUT,'(A)')'SYNTAX ERROR, OR ILLEGAL VALUE'
      IF (IERR .EQ. 2) WRITE(LOUT,'(A,I2, A, I2)')
     1   ' TOO FEW DATA ITEMS.  NUMBER FOUND = ' , NFOUND,
     2   '  NUMBER EXPECTED = ', NEXPEC
      IF (IERR .EQ. 3) WRITE(LOUT,'(A,I2)')
     1   ' TOO MANY DATA ITEMS.  NUMBER EXPECTED = ', NEXPEC
      END
C
C----------------------------------------------------------------------C
C
      FUNCTION IFIRCH   (STRING)
C   BEGIN PROLOGUE  IFIRCH
C   DATE WRITTEN   850626
C   REVISION DATE  850626
C   CATEGORY NO.  M4.
C   KEYWORDS  CHARACTER STRINGS,SIGNIFICANT CHARACTERS
C   AUTHOR  CLARK,G.L.,GROUP C-3 LOS ALAMOS NAT'L LAB
C   PURPOSE  Determines first significant (non-blank) character
C            in character variable
C   DESCRIPTION
C
C-----------------------------------------------------------------------
C  IFIRCH locates the first non-blank character in a string of
C  arbitrary length.  If no characters are found, IFIRCH is set = 0.
C  When used with the companion routine ILASCH, the length of a string
C  can be determined, and/or a concatenated substring containing the
C  significant characters produced.
C-----------------------------------------------------------------------
C
C   REFERENCES  (NONE)
C   ROUTINES CALLED  (NONE)
C   END PROLOGUE IFIRCH
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER* (*)STRING
C
C   FIRST EXECUTABLE STATEMENT IFIRCH
      NLOOP = LEN(STRING)
C
      IF (NLOOP.EQ.0 .OR. STRING.EQ.' ') THEN
         IFIRCH = 0
         RETURN
      ENDIF
C
      DO 100 I = 1, NLOOP
         IF (STRING(I:I) .NE. ' ') GO TO 120
100   CONTINUE
C
      IFIRCH = 0
      RETURN
120   CONTINUE
      IFIRCH = I
      END
C
C----------------------------------------------------------------------C
C
      FUNCTION ILASCH   (STRING)
C   BEGIN PROLOGUE  ILASCH
C   DATE WRITTEN   850626
C   REVISION DATE  850626
C   CATEGORY NO.  M4.
C   KEYWORDS  CHARACTER STRINGS,SIGNIFICANT CHARACTERS
C   AUTHOR  CLARK,G.L.,GROUP C-3 LOS ALAMOS NAT'L LAB
C   PURPOSE  Determines last significant (non-blank) character
C            in character variable
C   DESCRIPTION
C
C-----------------------------------------------------------------------
C  IFIRCH locates the last non-blank character in a string of
C  arbitrary length.  If no characters are found, ILASCH is set = 0.
C  When used with the companion routine IFIRCH, the length of a string
C  can be determined, and/or a concatenated substring containing the
C  significant characters produced.
C  Note that the FORTRAN intrinsic function LEN returns the length
C  of a character string as declared, rather than as filled.  The
C  declared length includes leading and trailing blanks, and thus is
C  not useful in generating 'significant' substrings.
C-----------------------------------------------------------------------
C
C   REFERENCES  (NONE)
C   ROUTINES CALLED  (NONE)
C   END PROLOGUE IFIRCH
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER*(*) STRING
C
C   FIRST EXECUTABLE STATEMENT ILASCH
      NLOOP = LEN(STRING)
      IF (NLOOP.EQ.0 .OR. STRING.EQ.' ') THEN
         ILASCH = 0
         RETURN
      ENDIF
C
      DO 100 I = NLOOP, 1, -1
         ILASCH = I
         IF (STRING(I:I) .NE. ' ') RETURN
100   CONTINUE
C
      END
C
C----------------------------------------------------------------------C
C
      FUNCTION IPPLEN (LINE)
C
C  BEGIN PROLOGUE
C
C  FUNCTION IPPLEN (LINE)
C     Returns the effective length of a character string, i.e.,
C     the index of the last character before an exclamation mark (!)
C     indicating a comment.
C
C  INPUT
C     LINE  - A character string.
C                  Data type - CHARACTER*(*)
C
C  OUTPUT
C     IPPLEN - The effective length of the character string.
C                   Data type - integer scalar
C
C  END PROLOGUE
C
C*****precision > double
       IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*)
C
      IN = IFIRCH(LINE)
      IF (IN.EQ.0 .OR. LINE(IN:IN) .EQ. '!') THEN
         IPPLEN = 0
      ELSE
         IN = INDEX(LINE,'!')
         IF (IN .EQ. 0) THEN
            IPPLEN = ILASCH(LINE)
         ELSE
            IPPLEN = ILASCH(LINE(:IN-1))
         ENDIF
      ENDIF
      RETURN
      END
C
C----------------------------------------------------------------------C

