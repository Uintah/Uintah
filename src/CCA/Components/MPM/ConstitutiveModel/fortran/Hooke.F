C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C

C
C 
C 
! This file contains subroutines for Hooke's law
! HOOKECHK: Checks validity of user input and sets defaults
! HOOKERXV: Defines internal state variable requirements
! HOOKE_INCREMENTAL: Main physics driver
! HOOKE_INIT: state variable initialization if MG EOS is used
! MGEOS: Driver for Mie Gruneisen EOS
! EOSMGI: Data check routine for MG EOS
! EOSMGR: Energy as function of temperature and density
! EOSMGV: Pressure and temperature as function of energy and density
!
C#include <CCA/Components/MPM/ConstitutiveModels/fortran/HookeChk_fort.h>
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7-|--.----
      SUBROUTINE HOOKE_INCREMENTAL( NBLK, NINSV, DT, PROP,
     .                              SIGARG, D, SVARG, USM)
C PUBLIC: called by host code
C         This is the main physics routine, called every timestep for
C         every Hooke's Law element that is to be evaluated in rate form.
!
!***********************************************************************
!
!     DESCRIPTION: HOOKE'S LAW IN INCREMENTAL FORM.
!
!     FORMAL PARAMETERS:
!i      NBLK   INTEGER   NUMBER OF ELEMENTS IN THIS BLOCK
!                        Parallel codes normally send NBLK=1
!                        Vectorized codes send more than one element
!                        at a time (NBLK>1).
!i      NINSV  INTEGER   NUMBER OF INTERNAL STATE VARIABLES
!i      DT     REAL      TIME STEP
!i      PROP   REAL      MATERIAL PROPERTIES (see below)
!io     SIGARG REAL      STRESSES
!                        On input, send stress at beginning of step
!                        Output is stress at end of step
!i      D      REAL      STRAIN RATES (preferably at the half step)
!io     SVARG  REAL      INTERNAL STATE VARIABLES (see below)
!o      USM    REAL      Uniaxial strain modulus (with which the host code
!                        may compute an UPPER BOUND on the longitudinal
!                        wave speed by dividing USM by density and taking
!                        the square root of the result)
!
!     CALLED BY:host code during physics loop over elements/cells
!
!     REF: Any undergraduate mechanics of materials book.
!
!     PROGRAMMED BY R.M. BRANNON
!     The FORTRAN-90 version is available upon request
!
!***********************************************************************
!
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#else
#include "precision.Blk"
#include "numbers.Blk"
#endif
#include "hookepnt.Blk"
!
!***********************************************************************
!
! Material Property and control Parameters
!
!
!***********************************************************************
!
!  State variables
!
!***********************************************************************
!
C...parameters
C...parameters (numbers)
C...passed
      DIMENSION SVARG(NINSV,NBLK),SIGARG(6,NBLK),USM(NBLK)
      DIMENSION PROP(*),D(6,NBLK)
C...local
      character*18 iam
      parameter (iam='HOOKE_INCREMENTAL')
      DIMENSION DE(6),DEDEV(6),DS(6)
      DIMENSION STRESSP(6),STRESSN(6),S(6),SN(6),SP(6)
      DIMENSION IDENT(6) 
      SAVE IDENT
      DATA IDENT/PONE,PONE,PONE,PZERO,PZERO,PZERO/
C
#ifdef BRANNON_DEBUG
#include "debug.h"
#endif
C
! Statement functions
C Specialized version of exponential, apparently implemented to prevent
C underflow replacement of EXP(Z) by zero when Z is an extraordinarily
C large negative number. For all practical purposes, EXPS should be
C regarded as equivalent to EXP.
      PARAMETER (EUNDERFLOW=-34.53877639491D0*PONE)
      PARAMETER (EOVERFLOW=92.1034037D0*PONE)
      EXPS(ARG) = EXP(MIN(MAX(ARG,EUNDERFLOW),EOVERFLOW))
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      IBLK=1 
*
*  ...Pass mtl props to local vars
*
      THREEK=PTHREE*PROP(IPB0)
      TWOG=PTWO*PROP(IPG0)

*
*  ...save values at beginning of step to local vars
*
C     PRINT *, SIGARG(1,1)

#ifdef MG_EOS
      RHON = SVARG(KRHO,IBLK)
      PRESN = SVARG(KPRES,IBLK) 
      !PRESN = -PTHIRD*(SIGARG(1,IBLK)+SIGARG(2,IBLK)+SIGARG(3,IBLK))
#else
      PRESN = -PTHIRD*(SIGARG(1,IBLK)+SIGARG(2,IBLK)+SIGARG(3,IBLK))
#endif
*  ...total and deviatoric stress at beginning of the increment
      DO IJ=1,6
        STRESSN(IJ) = SIGARG(IJ,IBLK)
        SN(IJ)      = SIGARG(IJ,IBLK) + PRESN*IDENT(IJ)
      ENDDO
      
*  ...strain inc.
      DO IJ=1,6
        DE(IJ) = D(IJ,IBLK)*DT
      ENDDO

*  ...volumetric strain inc.
      DEV = DE(1)+DE(2)+DE(3)

*  ...deviatoric strain inc.
      DO IJ=1,6
        DEDEV(IJ) = DE(IJ) - PTHIRD*DEV*IDENT(IJ) 
      ENDDO
 
*
*  ...UPDATE STRESS
*

*  ...pressure
#ifdef MG_EOS
*     if EOS is used, pressure is computed by it
      CALL MGEOS(PROP, SVARG, DEV,
     .           PRES, TMPR, SNDSP)
      !PRES = PRESN - PTHREE*(RHON*SNDSP**2-PTHIRD*TWOG)*PTHIRD*DEV
#else
      PRES = PRESN - THREEK*PTHIRD*DEV
#endif

*  ...deviatoric stress
      DO IJ=1,6
        SP(IJ) = SN(IJ) + TWOG*DEDEV(IJ)
      ENDDO

*  ...stress at end of step
      DO IJ=1,6
        STRESSP(IJ) = -PRES*IDENT(IJ) + SP(IJ) 
      ENDDO
      
#ifdef MG_EOS
      RHOP = RHON*EXP(-DEV)

*  ...Energy update
      ENINC = PZERO
      DO IJ=1,3
        ENINC = ENINC+PHALF*(STRESSN(IJ)/RHON + STRESSP(IJ)/RHOP)*DE(IJ)
      ENDDO
      DO IJ=4,6
        ENINC = ENINC+      (STRESSN(IJ)/RHON + STRESSP(IJ)/RHOP)*DE(IJ)
      ENDDO
      ENRGYP = SVARG(KENRGY,IBLK) + ENINC

*  ...Distoritional energy update
*      ENINC = PZERO
*      DO IJ=1,3
*        ENINC = ENINC+PHALF*(SN(IJ)/RHON + SP(IJ)/RHOP)*DEDEV(IJ)
*      ENDDO
*      DO IJ=4,6
*        ENINC = ENINC+      (SN(IJ)/RHON + SP(IJ)/RHOP)*DEDEV(IJ)
*      ENDDO
*      DISTENRGYP = DISTENRGYN + ENINC
      distenrgyp = pzero
*      print*,enrgyp,distenrgyp
*      call bombed('here')
#endif


*  ...ISV update
      SVARG(KMSG,IBLK) = SVARG(KMSG,IBLK) + 0.1D-1
#ifdef MG_EOS
      SVARG(KENRGY,IBLK) = ENRGYP
      SVARG(KDISTENRGY,IBLK) = DISTENRGYP
      SVARG(KSNDSP,IBLK) = SNDSP
      SVARG(KTMPR,IBLK)  = TMPR
      SVARG(KPRES,IBLK)  = PRES
      SVARG(KRHO,IBLK)   = RHOP
#endif

*  ...Pass updated stress and compute USM
      DO IJ=1,6
        SIGARG(IJ,IBLK) = STRESSP(IJ)
      ENDDO
      USM(IBLK) = PTHIRD*(PTWO*TWOG+THREEK)

C
      RETURN
      END !SUBROUTINE HOOKE_INCREMENTAL
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE HOOKECHK ( UI, GC, DC)
C
C PUBLIC: called by host code after user inputs have been obtained from
C the user and assembled into the array UI (called PROP in the main
C physics routine).
C***********************************************************************
C     REQUIRED MIG DATA CHECK ROUTINE
C     Checks validity of user inputs for Hooke's Law model.
C     Sets defaults for unspecified user input.
C     Adjusts user input to be self-consistent.
C
C     input
C     -----
C        UI: user input as read and stored by host code.
C
C        Upon entry, the UI array contains the user inputs EXACTLY
C        as read from the user. These inputs must be ordered in the
C        UI array as follows:
C
C
C         DC: Not used with this model
C
C     Other output
C     ------------
C         GC: Not used with this model
C         DC: Not used with this model
C         Because GC and DC are not used, you may call this routine
C         with a line of the form “CALL HOOKECHK(UI,UI,UI)”
C
C***********************************************************************
C  author: Rebecca Brannon
C
C  yymmdd:usernam: m o d i f i c a t i o n
C  ---------------------------------------------------------------------
C  111111:rmbrann:Created original data check
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#include <src/material/Smod_numbers.blk>
#else
#include "precision.Blk"
#include "numbers.Blk"
#endif
#include "hookepnt.Blk"
C ...local
      CHARACTER*6 IAM
      PARAMETER( IAM = 'HOOKECHK' )
CCCC  character*60 jnkstr
      LOGICAL DEJAVU
      DATA DEJAVU/.FALSE./
C
C
      DIMENSION UI(*), GC(*), DC(*)
c....................................................statement functions
      PARAMETER (EUNDERFLOW=-34.53877639491D0*PONE)
      PARAMETER (EOVERFLOW=92.1034037D0*PONE)
      EXPS(ARG) = EXP(MIN(MAX(ARG,EUNDERFLOW),EOVERFLOW))
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C      if(dejavu)return
C
c     DEJAVU=(NINT(UI(IPGMM+IPDEJAVU)).NE.0)
C     The logical DEJAVU is true if the GeoModel parameters have been
C     checked. By extension, if they have been checked, then so have
C     the effective stress parameters, and there is no need to remain.
!     IF(.NOT.DEJAVU)THEN
!       CALL LOGMES('############# Hookes Law version 000000')
!     ELSE
!       RETURN
!     ENDIF
C
!     call elaspar(rho,0,0,ui(iplam))
C
C     Derived constants
C      call elaspar(ui(iprho),0,0,ui(1))

C      PRINT *, 'UI(IPB0) = ', UI(IPB0)
C      PRINT *, 'UI(IPG0) = ', UI(IPG0)
      IF(UI(IPB0).LE.PZERO)CALL BOMBED('B0.LE.0')
      IF(UI(IPG0).LE.PZERO)CALL BOMBED('G0.LE.0')
      RPOISSON = (PTHREE*UI(IPB0)-PTWO*UI(IPG0))/
     .           (PSIX*UI(IPB0) + PTWO*UI(IPG0))
      IF(RPOISSON.LE.-PONE)CALL BOMBED('unrealistic b0 and g0 combo')
      IF(RPOISSON.LE.PZERO)CALL LOGMES('negative poisson ratio')


      RETURN
      END !SUBROUTINE HOOKECHK
C
C
C
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE HOOKERXV(UI,GC,DC,
     & NX, NAMEA, KEYA, RINIT, RDIM, IADVCT, ITYPE)
C PUBLIC: called by host code
C For overall meanings of the calling arguments, see the prolog of
C subroutine GEORXV. This routine does the same thing as that one, but
C prepends the ISV array with state variables needed for effective stress
C modeling, and it also changes the input/output specification for a
C couple of the GeoModel state variables.
C***************************************************************STEP E
C
C  author: Rebecca Brannon
C
C    who yymmdd M O D I F I C A T I O N
C  ------- ------ ----------------------------------------------------
C  rmbrann 111111 Created original extra variable routine
C
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#include <src/material/Smod_numbers.blk>
#else
#include "precision.Blk"
#include "numbers.Blk"
#endif
#include "hookepnt.Blk"
C
      INTEGER MMCN,MMCK,MNUNIT,MMCNA,MMCKA
      PARAMETER (MMCN=40,MMCK=10,MNUNIT=7)
      PARAMETER (MMCNA=NHOOKEISV*MMCN,MMCKA=NHOOKEISV*MMCK)
C
      CHARACTER*(MMCN) NAME(NHOOKEISV)
      CHARACTER*(MMCK) KEY(NHOOKEISV)
      CHARACTER*1 NAMEA(*), KEYA(*)
      DIMENSION IADVCT(*),ITYPE(*)
      DIMENSION UI(*), GC(*), DC(*), RINIT(*), RDIM(7,*)
C
      CHARACTER*6 IAM
      PARAMETER(IAM='HOOKERXV')
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      CALL LOGMES('############# Requesting Hookes Law ISVs')
C
#ifdef MG_EOS
      CALL HOOKE_INIT(UI,RINIT)
#endif

      NX=0
C ##################################################################
      NX=NX+1
      IF(NX.NE.KMSG)CALL BOMBED(IAM//' KMSG pointer wrong')
      NAME(NX)='Message'
      KEY(NX)='MSG'
      IADVCT(NX)=1        ! input and output
#ifdef MG_EOS
!OC---------------------------------------------------------------------2
      NX=NX+1                                                       !KTMPR
      IF(NX.NE.KTMPR)CALL BOMBED(IAM//' KTMPR pointer wrong')
      NAME(NX)='Temperature'
      KEY(NX)='TMPR'
      IADVCT(NX)=1        ! input and output
      RDIM(4,NX)=1        ! units: temperature
!OC---------------------------------------------------------------------3
      NX=NX+1                                                      !KENRGY
      IF(NX.NE.KENRGY)CALL BOMBED(IAM//' KENRGY pointer wrong')
      NAME(NX)='Specific internal energy'
      KEY(NX)='ENRGY'
      IADVCT(NX)=1        ! input and output
      RDIM(1,NX)=PTWO     ! units: energy
      RDIM(2,NX)=PONE 
      RDIM(3,NX)=-PTWO
!OC---------------------------------------------------------------------4
      NX=NX+1                                                  !KDISTENRGY
      IF(NX.NE.KDISTENRGY)CALL BOMBED(IAM//' KDISTENRGY pointer wrong')
      NAME(NX)='Specific distortional energy'
      KEY(NX)='DISTENRGY'
      IADVCT(NX)=1        ! input and output
      RDIM(1,NX)=PTWO     ! units: energy
      RDIM(2,NX)=PONE 
      RDIM(3,NX)=-PTWO
!OC---------------------------------------------------------------------4
      NX=NX+1                                                        !KRHO
      IF(NX.NE.KRHO)CALL BOMBED(IAM//' KRHO pointer wrong')
      NAME(NX) = 'Density'
      KEY(NX) = 'RHO'
      IADVCT(NX)=1        ! output only
*     itype=1  (scalar)
      RDIM(1,NX)= -PTHREE   ! units: mass/length^3
      RDIM(2,NX)=  PONE 
      RDIM(3,NX)=  PZERO
!OC---------------------------------------------------------------------5
      NX=NX+1                                                      !KSNDSP
      IF(NX.NE.KSNDSP)CALL BOMBED(IAM//' KSNDSP pointer wrong')
      NAME(NX) = 'Sound speed'
      KEY(NX) = 'SNDSP'
      IADVCT(NX)=1        ! output only
*     itype=1  (scalar)
      RDIM(1,NX)=  PONE   ! units: length/time
      RDIM(2,NX)=  PZERO 
      RDIM(3,NX)=  PONE
!OC---------------------------------------------------------------------6
      NX=NX+1                                                       !KPRES
      IF(NX.NE.KPRES)CALL BOMBED(IAM//' KPRES pointer wrong')
      NAME(NX) = 'Pressure'
      KEY(NX) = 'PRES'
      IADVCT(NX)=1        ! output only
*     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
!OC---------------------------------------------------------------------7
      NX=NX+1                                                      !KALPHA
      IFLGPORO=0
      IF(NX.NE.KALPHA)CALL BOMBED(IAM//' KALPHA pointer wrong')
      NTYP = NINT(PTEN*UI(IPTYP))
      IF (UI(IDCEOSMGCT+7).GT.PONE) THEN
        IFLGPORO=1
        NAME(NX) = 'Alpha Parameter'
        IF (NTYP.EQ.20 .OR. NTYP.EQ.40) THEN
          KEY(NX) = 'ALPI'
        ELSEIF (NTYP.EQ.21 .OR. NTYP.EQ.41) THEN
          KEY(NX) = 'ALPF'
        ELSE
          KEY(NX) = 'ALP'
        ENDIF
        RINIT(NX) = UI(IDCEOSMGCT+7)
      ENDIF

#endif
*************************************************************************
************************************************************endif MG_EOS

#ifdef BRANNON_DEBUG
      CALL LOGMES(IAM//' WARNING: iadvct globally set to 1 for debug')
      WRITE(66,*) IAM//' WARNING: iadvct globally set to 1 for debug'
      DO IX=1,NX
         IADVCT(IX)=1
      ENDDO
#endif
C     ##################################################################
      IF(NX.GT.NHOOKEISV)CALL BOMBED
     & ('INCREASE NHOOKEISV IN ROUTINE GEORXV AND IN DATA FILE')
C     convert NAME and KEY to character streams NAMEA and KEYA
C     (See note about TOKENS in prolog of this routine)
      CALL TOKENS(NX,NAME,NAMEA)
      CALL TOKENS(NX,KEY ,KEYA )

C
c      CALL LOGMES(‘############# exiting HOOKERXV’)
      RETURN
      END !SUBROUTINE HOOKERXV
C
#ifdef MG_EOS
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE HOOKE_INIT(PROP, SV)
C PUBLIC:Called by host code after GeoModel inputs have been read and
C        checked (in routine GEOCHK).
C
C***********************************************************************
C
C     DESCRIPTION:
C       This routine initializes GeoModel internal state variables.
C
C     FORMAL PARAMETERS:
C       PROP     REAL      material properties for this material
C       SV       REAL      array containing the internal state
C                          variables which must be intialized
C
C***********************************************************************
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#include <src/material/Smod_numbers.blk>
#else
#include "precision.Blk"
#include "numbers.Blk"
#endif
#include "hookepnt.Blk"
C
      DIMENSION SV(*),PROP(*)
C
C parameters and functions from svinit.f need for this model:
C
      PARAMETER (
     *  NIT  = 1000,
     *  CONV = TOL1M6
     *  )
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      DO ISV=1,NHOOKEISV
        SV(ISV)=PZERO
      ENDDO
*      
* ... Pass initial values from user input array to sv array of
*     quantities to be tracked by the material model:
*
      IF(PROP(IPRHO0).EQ.0.0)THEN
        CALL BOMBED('INITIAL DENSITY NOT SPECIFIED')
      ELSE
        SV(KRHO)=PROP(IPRHO0)
      ENDIF
      IF(PROP(IPTMPR0).EQ.0)THEN
        SV(KTMPR)=293.D0
      ELSE
        SV(KTMPR)=PROP(IPTMPR0)
      ENDIF
      IF(PROP(IPSNDSP0).EQ.0)THEN
        CALL LOGMES('NO INITIAL SOUND SPEED SPECIFIED, 
     .               USING DEFAULT SNDSP=SQRT(K/RHO)')
        SV(KSNDSP)=SQRT(PROP(IPB0)/PROP(IPRHO0))
      ELSE
        SV(KSNDSP)=PROP(IPSNDSP0)
      ENDIF

*
*     Initialize energy using Mie-Gruneisen EOS
*
* ... IFLGPORO specifies if porosity option is used (=1) or not (=0)
      IFLGPORO=0
      IF (PROP(IDCEOSMGCT+7).GT.PONE) THEN
        IFGLPORO=1
        SV(KALPHA)=PROP(IDCEOSMGCT+7)
      ENDIF
*
* ... Set initial energy with EOSMGR.  EOSMGR takes density and temperature
*     and returns pressure and energy.
*
      CALL EOSMGR(1,1,PROP(IPRHO0),PROP(IPRHO0),PROP(IPA1MG),
*     Input
     . SV(KRHO), SV(KTMPR), SV(KALPHA), IFLGPORO,
*     Output
     . PRESS, ENRGY0, CS, 
*     Scratch
     . SCRATCH )
      CALL FATRET(NERR)
      IF(NERR.GT.0)CALL BOMBED('ERROR IN EOSMGR')
      if(enrgy0.lt.pzero)then
              sv(kenrgy)=prop(ipesft)
      else
      SV(KENRGY)=ENRGY0
      endif


      RETURN
      END !SUBROUTINE HOOKE_INIT

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE MGEOS(PROP,SV,DEV,
*    Output 
     . PRES, TMPR, SNDSP)
C***********************************************************************
C     PURPOSE: This routine evaluates the equation of state
C
C input
C -----
C    VAR: description
C
C input and output
C ----------------
C    VAR: description
C
C output
C -----
C    VAR: description
C
C
C  AUTHORS
C  rmb:Rebecca Brannon:theory, algorithm, and code
C
C  MODIFICATION HISTORY
C  yymmdd|who|what was done
C  ------ --- -------------
C  chngme|rmb|created routine
C
C
#ifdef BRANNON_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
#include "hookepnt.Blk"
#include "hookeProp.h"
C
C.............................................................parameters
c#include "numbers.Blk"
#ifdef BRANNON_IMPLNONE
      DOUBLE PRECISION PZERO,PONE,PTWO,PTHREE, PFOUR
#endif
      PARAMETER (PZERO=0.0D0,PONE=0.1D1,PTWO=0.2D1,PTHREE=0.3D1)
      PARAMETER (PFOUR=0.4D1)
      PARAMETER (PHALF   =0.5D0)
      PARAMETER (PTHIRD  =0.33333333333333333333333333333333333333333D0)
      PARAMETER (P4THIRD =1.33333333333333333333333333333333333333333D0)
      PARAMETER (TOL=1D-16, BET=0.D0, ALF=0.D0)
      PARAMETER (KNTMAX=20)
C
C.................................................................common
C.................................................................passed
#ifdef BRANNON_IMPLNONE
      DOUBLE PRECISION PROP,SV,PRES,TMPR,SNDSP
#endif
      DIMENSION PROP(*),SV(*)
C..............................................................functions
C...............................................................external
C..........................................................local scalars
#ifdef BRANNON_IMPLNONE
      DOUBLE PRECISION UBEG,UPSBEG,UPSEND,RHOBEG,PBEG,RHOEND
      DOUBLE PRECISION C1MG,C2MG,C3MG,C4MG
      DOUBLE PRECISION X,XLAST,TOP,BOT,EREF
      DOUBLE PRECISION PRESS,TEMP,SNDSPD,DPDT,DEDT
      INTEGER KNT,KNTMAX
#endif
C...........................................................local arrays
#ifdef BRANNON_IMPLNONE
      DOUBLE PRECISION SCRATCH
#endif
#ifdef GEO_EOS
      DIMENSION SCRATCH(9)
#endif
C..................................................................saved
C...................................................................data
C....................................................statement functions
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
*
*     Collect information about the time step
*

      IFLGPORO=0
* ... Energies
      UBEG     = SV(KENRGY)
      UPSBEG   = SV(KDISTENRGY)
      EBEG     = UBEG-UPSBEG
      UPSEND   = SV(KDISTENRGY)

* ... "Mass" stuff
      RHOBEG   = SV(KRHO)
      PBEG     = SV(KPRES)
      SNDSPBEG = SV(KSNDSP)
      RHOEND   = RHOBEG*EXP(-DEV)

*
*     Set up for sperical energy iteration
*
      ZBEG     = -PBEG/RHOBEG
      DEBEG    = ZBEG*DEV
      DDEBEG   = PHALF*(SNDSPBEG**2+ZBEG)*DEV*DEV
      EEND     = EBEG+DEBEG+DDEBEG
      UEND     = EEND + UPSEND
      EENDSAVE = EEND

      DO KNT=1,20

        CALL EOSMGV(1,1,
     *    PROP(IPRHO0),PROP,PROP(IPA1MG),
*      Input: DENSITY, ENERGY, ALPHA PARAMETER, POROSITY FLAG
     *    RHOEND, UEND, SV(KALPHA), IFLGPORO,
*      Output: PRESS, TMPR, SOUNDSPEED, SCRATCH
     *    PRES, TMPR, SNDSP, SCRATCH)
      

*
* ... Set up for Newton iteration
*
        ZEND=-PRESS/RHOEND
        DDEEND=PHALF*(SNDSPD**2+ZEND)*DEV*DEV
        DDE=PHALF*(DDEBEG+DDEEND)
        EEND=EBEG+DEBEG+DDE
        UEND=EEND+UPSEND

* ... Check for convergence
        EENDDIF=ABS(EEND-EENDSAVE)
        EENDSAVE=EEND
        EENDREF=MAX(ABS(EENDSAVE),0.1D-5*SCRATCH(1))
        IF(EENDDIF.LT.TOL*EENDREF)GO TO 8
      ENDDO
* ... If energy iterations failed, call bombed
      CALL BOMBED('EOS did not converge')

8     CONTINUE 


      RETURN
      END ! SUBROUTINE EOSMOD
      SUBROUTINE HOOKE_INIT(PROP, SV)
C PUBLIC:Called by host code after GeoModel inputs have been read and
C        checked (in routine GEOCHK).
C
C***********************************************************************
C
C     DESCRIPTION:
C       This routine initializes GeoModel internal state variables.
C
C     FORMAL PARAMETERS:
C       PROP     REAL      material properties for this material
C       SV       REAL      array containing the internal state
C                          variables which must be intialized
C
C***********************************************************************
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#include <src/material/Smod_numbers.blk>
#else
#include "precision.Blk"
#include "numbers.Blk"
#endif
#include "hookepnt.Blk"
C
      DIMENSION SV(*),PROP(*)
C
C parameters and functions from svinit.f need for this model:
C
      PARAMETER (
     *  NIT  = 1000,
     *  CONV = TOL1M6
     *  )
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      DO ISV=1,NHOOKEISV
        SV(ISV)=PZERO
      ENDDO
*      
* ... Pass initial values from user input array to sv array of
*     quantities to be tracked by the material model:
*
      IF(PROP(IPRHO0).EQ.0.0)THEN
        CALL BOMBED('INITIAL DENSITY NOT SPECIFIED')
      ELSE
        SV(KRHO)=PROP(IPRHO0)
      ENDIF
      IF(PROP(IPTMPR0).EQ.0)THEN
        SV(KTMPR)=293.D0
      ELSE
        SV(KTMPR)=PROP(IPTMPR0)
      ENDIF
      IF(PROP(IPSNDSP0).EQ.0)THEN
        CALL LOGMES('NO INITIAL SOUND SPEED SPECIFIED, 
     .               USING DEFAULT SNDSP=SQRT(K/RHO)')
        SV(KSNDSP)=SQRT(PROP(IPB0)/PROP(IPRHO0))
      ELSE
        SV(KSNDSP)=PROP(IPSNDSP0)
      ENDIF

*
*     Initialize energy using Mie-Gruneisen EOS
*
* ... IFLGPORO specifies if porosity option is used (=1) or not (=0)
      IFLGPORO=0
      IF (PROP(IDCEOSMGCT+7).GT.PONE) THEN
        IFGLPORO=1
        SV(KALPHA)=PROP(IDCEOSMGCT+7)
      ENDIF
*
* ... Set initial energy with EOSMGR.  EOSMGR takes density and temperature
*     and returns pressure and energy.
*
      CALL EOSMGR(1,1,PROP(IPRHO0),PROP(IPRHO0),PROP(IPA1MG),
*     Input
     . SV(KRHO), SV(KTMPR), SV(KALPHA), IFLGPORO,
*     Output
     . PRESS, ENRGY0, CS, 
*     Scratch
     . SCRATCH )
      CALL FATRET(NERR)
      IF(NERR.GT.0)CALL BOMBED('ERROR IN EOSMGR')
      if(enrgy0.lt.pzero)then
              sv(kenrgy)=prop(ipesft)
      else
      SV(KENRGY)=ENRGY0
      endif


      RETURN
      END !SUBROUTINE HOOKE_INIT
C  @(#)$Id: eosmgi.F,v 1.7 2006/03/13 22:41:09 rgschmi Exp $
*DECK EOSMGI
      SUBROUTINE EOSMGI (UI,GC,DC,
C     Extended MIG Input
     &  UC,MDC,NDC,VI)
C**********************************************EOS Package**************
C
C   PURPOSE.   EOSMGI: data check routine for Mie-Gruneisen EOS.
C
C   STANDARD MIG INPUT.
C     UI       = user input array
C     GC       = global constants (not used here)
C     DC       = derived material constants array
C
C   EXTENDED MIG INPUT.
C     UC       = character input array
C     MDC      = amount of storage available in array DC
C     NDC      = amount of storage used in array DC (here set to 12)
C     VI       = default initial values and melting parameters
C        VI(1) = initial density
C        VI(2) = initial temperature
C        VI(3) = minimum solid density
C        VI(4) = melting temperature
C        VI(5) = melting density
C
C   AUTHOR.    G. I. Kerley, 05/24/91
C   MODIFIED.  GIK-08/25/97-replaced with MIG version
C   MODIFIED.  GIK-11/20/97-added low-pressure modifications to Hugoniot
C   MODIFIED.  GIK-01/10/98-added default for p-alpha parameter PS
C   MODIFIED.  GIK-05/26/98-MIG modifications to MP model
C   MODIFIED.  GIK-03/16/00-added trapping of EOS errors
C   modified   dac-12/20/05-added fatal error if B0<0
C
C**********************************************GIK 03/16/00*************
C
C   Input parameters are:
C
C     UI(1)  = R0    - initial density (required)
C     UI(2)  = T0    - initial temperature (default is T0=0.02568 eV)
C     UI(3)  = CS    - sound speed (required)
C     UI(4)  = S1    - coefficient of linear term US-UP fit
C     UI(5)  = G0    - Gruneisen parameter
C     UI(6)  = CV    - heat capacity (required)
C     UI(7)  = ESFT  - shift in energy zero (see below)
C     UI(8)  = RP    - initial porous density (no default if porous)
C     UI(9)  = PS    - crushup pressure
C     UI(10) = PE    - pressure at elastic limit
C     UI(11) = CE    - sound speed in foam
C     UI(12) = NSUB  - number of subcycles in time step
C     UI(13) = S2    - coefficient of quad. term in US-UP fit
C     UI(14) = TYP   - used for 2-state and MP models (see below)
C     UI(15) = RO    - alias for R0
C     UI(16) = TO    - alias for T0
C     UI(17) = S     - alias for S1
C     UI(18) = GO    - alias for G0
C     UI(19) = B     - coefficient of low-pressure term
C     UI(20) = XB    - constant in low-pressure term
C     UI(21) = NB    - power of compression in low-pressure term
C     UI(22) = PWR   - power for alpha integration, default=2.0

C
C   --Setup will fail unless R0, CS, and CV are specified.
C   --Units: any consistent set of units can be used.
C   --Quadratic fit to Hugoniot is US = CS+S1*UP+(S2/CS)*UP**2. This
C     form makes S2 unitless.
C   --B-dependent term adds following correction to quadratic Hugoniot:
C     US -> US-B*EXP(-(X/XB)**NB), where X=1-R0/RHO.
C   --The default value of ESFT is chosen to make T=0 the energy zero,
C     but this value is overriden when any positive value is input.
C   --For 2-state models, set TYP=2.0 (initial state) or TYP=2.1 (final
C     state. ESFT=0 in this case.
C   --For MP model, set TYP=4.0 for parent, TYP=4.1 for daughter.
C
C   Derived parameters are:
C
C     DC(1)  = A(1)  - coefficient in temperature fit
C     DC(2)  = A(2)  - coefficient in temperature fit
C     DC(3)  = A(3)  - coefficient in temperature fit
C     DC(4)  = A(4)  - coefficient in temperature fit
C     DC(5)  = A(5)  - coefficient in temperature fit
C     DC(6)  = A0    - R0/RP
C     DC(7)  = AE    - (CE/CS-1)/(A0-1)
C     DC(8)  = FK0   - 1/(R0*CS**2)
C     DC(9)  = AF    - (A0-1)/(PS-PE)**2
C     DC(10) = PF    - parameter in temperature fit
C     DC(11) = XF    - parameter in temperature fit
C     DC(12) = CF    - parameter in temperature fit
C     DC(13) = RMX   - maximum allowed density
C
C***********************************************************************
C
#include "impdoubl.h"
#ifdef CTH_DYN
#include "iofils.h"
#include "mpcthl.h"
#endif
C
      CHARACTER*1 UC(*)
      DIMENSION UI(*),GC(*),DC(*),VI(*),A(8)
      PARAMETER (ZERO=0.0D0,HALF=0.5D0,ONE=1.0D0,TWO=2.0D0,FIV=5.0D0,
     & TEN=10.0D0,PD8=0.8D0,PM4=1.0D-4,PE9=1.0D9,PM6=1.0D-6)
      NDC = 13
      IF (MDC.LT.NDC) GO TO 2
*
*  Check for aliases--override other values if non-zero.
*
      IF (UI(15).NE.ZERO) UI(1)=UI(15)
      IF (UI(16).NE.ZERO) UI(2)=UI(16)
      IF (UI(17).NE.ZERO) UI(4)=UI(17)
      IF (UI(18).NE.ZERO) UI(5)=UI(18)
*
*  Check for required parameters.
*
      IF (UI(1).LE.ZERO) GO TO 3
      IF (UI(3).LE.ZERO) GO TO 3
      IF (UI(6).LE.ZERO) GO TO 3
*
*  Set defaults for low-pressure region.
*
      IF (UI(19).EQ.ZERO) THEN
        UI(20) = ONE
        UI(21) = ONE
      ELSE
        IF (UI(20).LE.ZERO) UI(20)=PM4
        IF (UI(21).LE.ZERO) UI(21)=ONE
      ENDIF
*
*  Compute fit coefficients for temperature function.
*
      CS = UI(3)
      S1 = UI(4)
      S2 = UI(13)
      G0 = UI(5)
      B = UI(19)
      XB = UI(20)
      FB = UI(21)
      PWR= UI(22)
      CALL EOSMGJ(CS,S1,S2,G0,B,XB,FB,A(1),A(6))
*
*  Compute derived parameters.
*
      DC(1) = A(1)
      DC(2) = A(2)
      DC(3) = A(3)
      DC(4) = A(4)
      DC(5) = A(5)
      DC(10) = A(6)
      DC(11) = A(7)
      DC(12) = A(8)
*
*  Compute default energy shift--ignore if TYP=2.1 or TYP=2.2.
*
      NTYP = NINT(UI(14))
      IF (NTYP.NE.2 .AND. UI(7).LE.ZERO) UI(7)=UI(6)*UI(2)
*
*  Set derived constants for p-alpha model.
*
      R0 = UI(1)
      RP = UI(8)
      IF (UI(22).LT.ONE+PM4) THEN
        CALL LOGMES('P-Alpha exponent reset to 1.0001')
        UI(22) = ONE+PM4
      ENDIF
*
      IF (RP.GT.ZERO .AND. RP.LT.R0) THEN
        IF (UI(9).LE.ZERO) UI(9)=PE9
        IF (UI(12).LE.ZERO) UI(12)=TEN
        PS = UI(9)
        PE = UI(10)
        CE = UI(11)
        C0 = CS-B
        IF (CE.LE.ZERO) CE=C0
        A0 = R0/RP
        DC(6) = A0
        DC(7) = (CE/C0-ONE)/(A0-ONE)
        DC(8) = ONE/(R0*C0**2)
        DC(9) = (A0-ONE)/(PS-PE)**PWR
      ELSE
        DC(6) = ONE
        DC(7) = ZERO
        DC(8) = ZERO
        DC(9) = ZERO
      ENDIF
*
*  Check for B0>0 consistency
*
      CV = UI(6)
      T0 = UI(2)
      B0 = R0*(CS*(ONE-B/CS))**2-G0**2*R0*CV*T0
      IF (B0.LT.ZERO) THEN
        CALL FATERR('EOSMGI',
     &   'MGRUN CONSTRAINT G0<CS*(1-B/CS)/SQRT(CV*T0) VIOLATED')
        GOTO 1
      ENDIF
*
*  Set maximum allowed density, used to trap unreasonable densities.
*
      IF (S2.GE.ZERO) THEN
        XMX = ONE/MAX(ONE+PM6,S1+TWO*SQRT(S2))
      ELSE
        XMX = MAX(ZERO,MIN(ONE-PM6,S1/(HALF*S1*S1-TWO*S2)))
      ENDIF
      DC(13) = R0/(ONE-XMX)
*
*  Set defaults for initial state and melting parameters.
*
      VI(1) = R0/DC(6)
      VI(2) = UI(2)
      VI(3) = ZERO
      VI(4) = FIV*UI(2)
      VI(5) = PD8*VI(1)

*
* Echo User processed input parameters
*
 1    CONTINUE
#ifdef CTH_DYN
      IF(LPARNT) THEN
        WRITE(KPT6,*) 'Mie-Gruneisen:'
        WRITE(KPT6,*) 'UI(1)  = R0   = ',UI(1)
        WRITE(KPT6,*) 'UI(2)  = T0   = ',UI(2)
        WRITE(KPT6,*) 'UI(3)  = CS   = ',UI(3)
        WRITE(KPT6,*) 'UI(4)  = S1   = ',UI(4)
        WRITE(KPT6,*) 'UI(5)  = G0   = ',UI(5)
        WRITE(KPT6,*) 'UI(6)  = CV   = ',UI(6)
        WRITE(KPT6,*) 'UI(7)  = ESFT = ',UI(7)
        WRITE(KPT6,*) 'UI(8)  = RP   = ',UI(8)
        WRITE(KPT6,*) 'UI(9)  = PS   = ',UI(9)
        WRITE(KPT6,*) 'UI(10) = PE   = ',UI(10)
        WRITE(KPT6,*) 'UI(11) = CE   = ',UI(11)
        WRITE(KPT6,*) 'UI(12) = NSUB = ',UI(12)
        WRITE(KPT6,*) 'UI(13) = S2   = ',UI(13)
        WRITE(KPT6,*) 'UI(14) = TYP  = ',UI(14)
        WRITE(KPT6,*) 'UI(15) = RO   = ',UI(15)
        WRITE(KPT6,*) 'UI(16) = TO   = ',UI(16)
        WRITE(KPT6,*) 'UI(17) = S    = ',UI(17)
        WRITE(KPT6,*) 'UI(18) = GO   = ',UI(18)
        WRITE(KPT6,*) 'UI(19) = B    = ',UI(19)
        WRITE(KPT6,*) 'UI(20) = XB   = ',UI(20)
        WRITE(KPT6,*) 'UI(21) = NB   = ',UI(21)
        WRITE(KPT6,*) 'UI(22) = PWR  = ',UI(22)
C
        WRITE(KPT6,*) 'Derived parameters are:'
        WRITE(KPT6,*) 'DC(1)  = A(1)  = ',DC(1)
        WRITE(KPT6,*) 'DC(2)  = A(2)  = ',DC(2)
        WRITE(KPT6,*) 'DC(3)  = A(3)  = ',DC(3)
        WRITE(KPT6,*) 'DC(4)  = A(4)  = ',DC(4)
        WRITE(KPT6,*) 'DC(5)  = A(5)  = ',DC(5)
        WRITE(KPT6,*) 'DC(6)  = A0    = ',DC(6)
        WRITE(KPT6,*) 'DC(7)  = AE    = ',DC(7)
        WRITE(KPT6,*) 'DC(8)  = FK0   = ',DC(8)
        WRITE(KPT6,*) 'DC(9)  = AF    = ',DC(9)
        WRITE(KPT6,*) 'DC(10) = PF    = ',DC(10)
        WRITE(KPT6,*) 'DC(11) = XF    = ',DC(11)
        WRITE(KPT6,*) 'DC(12) = CF    = ',DC(12)
        WRITE(KPT6,*) 'DC(13) = RMX   = ',DC(13)
      ENDIF
#endif
*
      GO TO 4
*
*  Input errors
*
 2    CALL FATERR('EOSMGI','INSUFFICIENT STORAGE IN DC ARRAY')
      GO TO 4
 3    CALL FATERR('EOSMGI','REQUIRED INPUT PARAMETERS MISSING')
 4    RETURN
      END
C  @(#)$Id: eosmgr.F,v 1.9 2004/05/20 23:47:44 rgschmi Exp $
C
      SUBROUTINE EOSMGR (MC,NC,UI,GC,DC,
C     Input
     &  RHO,TEMP,ALPH,NX,
C     Output
     &  PRES,ENRG,CS,
C     Scratch
     &  S)
C
C**********************************************EOS Package**************
C
C   PURPOSE.   EOSMGR: pressure & energy as functions of density
C              & temperature using Mie-Gruneisen EOS.
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
C     TEMP     = ABSOLUTE_TEMPERATURE
C     ALPH     = EXTRA~1 (porosity parameter)
C     NX       = NX=1 if using porosity option, else NX=0
C
C   OUTPUT.
C     PRES     = THERMODYNAMIC_PRESSURE
C     ENRG     = SPECIFIC_INTERNAL_ENERGY
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
C   MODIFIED.  RMB-05/28/08-revised to fit into Brannon's driver
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
#include "impdoubl.h"
#ifdef CTH_DYN 
#include "mxsarr.h"
#include "eoserr.h"
#endif
C
      CHARACTER*10 DEN
      PARAMETER (ZERO=0.D0,ONE=1.0D0,TWO=2.0D0,THR=3.0D0,FOR=4.0D0,
     & FIV=5.0D0,SIX=6.0D0,HALF=0.5D0)
      DIMENSION UI(*),GC(*),DC(*)
      DIMENSION RHO(MC), TEMP(MC), ALPH(MC),
     &          PRES(MC), ENRG(MC), CS(MC), S(MC,8)
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
*
      R0 = UI(1)
      T0 = UI(2)
      C0 = UI(3)
      S0 = R0*C0**2
      S1 = UI(4)
      S2 = UI(13)
      G0 = UI(5)
      CV = UI(6)
      DPT = G0*R0*CV
      BX = UI(19)/C0
      XB = UI(20)
      FB = UI(21)
      PWR = UI(22)
      CX = TWO*FB
      B0 = R0*(C0*(ONE-BX))**2-G0*DPT*T0
      A1 = DC(1)
      A2 = DC(2)
      A3 = DC(3)
      A4 = DC(4)
      A5 = DC(5)
      PF = DC(10)
      XF = DC(11)
      CF = DC(12)
      RMX = DC(13)
      AF = DC(9)
      PS = UI(9)
      CE = UI(11)

*
*  Check for realistic density.
*
      DO 1 N=1,NC
       IF(RHO(N).GE.RMX .OR. RHO(N).LE.ZERO)  THEN
C        Flag the error so we can get graceful shutdown or at least
C        an error report of some kind
#ifndef CTH_DYN
C        This produces a fatal error with an ungraceful shutdown
         WRITE(DEN,'(1PE10.3)')  RHO(N)
#ifdef BRANNON_RECOMMENDATION
c        If faterr is called, the calculation will not terminate.
c        Consequently, RHO(N) will be assigned a new value, making it
c        now an output variable, which is contrary to the subroutine
c        prolog and which could therefore corrupt the database of
c        the calling routine if it had special coding that assumed
c        RHO truly was an input-only.
         CALL BOMBED('EOSMGR: UNATTAINABLE DENSITY, RHO='//DEN)
#else
         CALL FATERR('EOSMGR','UNATTAINABLE DENSITY, RHO='//DEN)
#endif
#else
C        This variable is in a common block so EOSMAP will issue the
C        correct warning message
         IEOSRR(N)=1
#endif
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
        S(N,5) = T0*EXP(G0*S(N,1))
        ENRG(N) = S0*(HALF-S(N,3))*(S(N,1)*S(N,2))**2/R0-CV*S(N,5)
        PRES(N) = S0*S(N,1)*(ONE-G0*S(N,1)*S(N,3))*S(N,2)**2-DPT*S(N,5)
        CS(N) = S0*S(N,2)*(CS(N)*(ONE-G0*S(N,1)*S(N,3))
     &          -S(N,1)*G0*S(N,4)*S(N,2))-DPT*G0*S(N,5)
       ELSE
        ENRG(N) = HALF*B0*S(N,1)**2/R0-DPT*T0*S(N,1)/R0-CV*T0
        PRES(N) = B0*S(N,1)-DPT*T0
        CS(N) = B0
       ENDIF
*
*  Add temperature terms.
*
       ENRG(N) = ENRG(N)+CV*TEMP(N)+UI(7)
       PRES(N) = PRES(N)+DPT*TEMP(N)
       IF (NX.GT.0) PRES(N)=PRES(N)/MAX(ONE,ALPH(N))
       S(N,1) = CS(N)*R0/S(N,6)**2
       S(N,2) = DPT
       IF (NX.GT.0) S(N,2)=S(N,2)/MAX(ONE,ALPH(N))
       S(N,3) = CV
       !print*,1,enrg(n),n
       S(N,4) = (PRES(N)-TEMP(N)*S(N,2))/RHO(N)**2
       !print*,2,enrg(n),n
       CS(N) = MAX(ZERO,S(N,1)+TEMP(N)*(S(N,2)/RHO(N))**2/CV)
       CS(N) = SQRT(CS(N))
       IF (NX.GT.0) THEN
        IF (CE.GT.CS(N).AND.ALPH(N).LT.ONE+AF*MAX(ZERO,PS-PRES(N))**PWR)
     &     CS(N)=CE
       ENDIF
 2    CONTINUE
      RETURN
      END
C  @(#)$Id: eosmgv.F,v 1.10 2004/05/20 23:47:44 rgschmi Exp $
C
      SUBROUTINE EOSMGV (MC,NC,UI,GC,DC,
C     Input
     &  RHO,ENRG,ALPH,NX,
C     Output
     &  PRES,TEMP,CS,
C     Scratch
     &  S)
C
C**********************************************EOS Package**************
C
C   PURPOSE.   EOSMGV: pressure & temperature as functions of
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
#include "impdoubl.h"
#ifdef CTH_DYN
#include "mxsarr.h"
#include "eoserr.h"
#endif
C
      CHARACTER*10 DEN
      PARAMETER (ZERO=0.D0,ONE=1.0D0,TWO=2.0D0,THR=3.0D0,FOR=4.0D0,
     &           FIV=5.0D0,SIX=6.0D0,HALF=0.5D0)
      DIMENSION UI(*),GC(*),DC(*)
      DIMENSION RHO(MC), ENRG(MC), ALPH(MC),
     &          PRES(MC), TEMP(MC), CS(MC), S(MC,8)
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
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
#ifndef CTH_DYN
C        This produces a fatal error with an ungraceful shutdown
         WRITE(DEN,'(1PE10.3)')  RHO(N)
         CALL FATERR('EOSMGV','UNATTAINABLE DENSITY, RHO='//DEN)
#else
C        This variable is in a common block so EOSMAP will issue the
C        correct warning message
         IEOSRR(N)=1
#endif
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
#endif
