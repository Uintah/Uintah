//----- ArchesFort.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ArchesFort_h
#define Uintah_Component_Arches_ArchesFort_h

/**************************************

HEADER
   ArchesFort
   
   Contains the header files to define interfaces between Fortran and
   C++ for Arches.

GENERAL INFORMATION

   ArchesFort.h

   Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 University of Utah

KEYWORDS
   Arches Fortran Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

// GROUP: Function Definitions:
////////////////////////////////////////////////////////////////////////
//
#define FORT_INIT init_
#define FORT_INIT_SCALAR initscal_
#define FORT_CELLTYPEINIT celltypeinit_
#define FORT_CELLG cellg_
#define FORT_AREAIN areain_
#define FORT_PROFV profv_
#define FORT_PROFSCALAR profscalar_
#define FORT_SMAGMODEL smagmodel_
#define FORT_CALPBC calpbc_
#define FORT_INLBCS inlbcs_
#define FORT_UVELCOEF uvelcoef_
#define FORT_VVELCOEF vvelcoef_
#define FORT_WVELCOEF wvelcoef_
#define FORT_VELCOEF velcoef_
#define FORT_UVELSOURCE uvelsrc_
#define FORT_VVELSOURCE vvelsrc_
#define FORT_WVELSOURCE wvelsrc_
#define FORT_VELSOURCE vsource_
#define FORT_BCUVEL bcuvel_
#define FORT_BCVVEL bcvvel_
#define FORT_BCWVEL bcwvel_
#define FORT_MASCAL mascal_
#define FORT_APCAL apcal_
#define FORT_PRESSCOEFF prescoef_
#define FORT_PRESSOURCE pressrc_
#define FORT_PRESSBC bcpress_
#define FORT_ADDPRESSGRAD addpressgrad_
#define FORT_SCALARCOEFF scalcoef_
#define FORT_SCALARSOURCE scalsrc_
#define FORT_SCALARBC bcscalar_
#define FORT_COMPUTERESID rescal_
#define FORT_COLDPROPS cprops_
#define FORT_UNDERRELAX urelax_
#define FORT_RBGLISOLV lisolv_
#define FORT_BCUTURB bcut_
#define FORT_BCVTURB bcvt_
#define FORT_BCWTURB bcwt_

// GROUP: Function Declarations:
////////////////////////////////////////////////////////////////////////
//

extern "C"
{
    ////////////////////////////////////////////////////////////////////////
    //
    // Initialize basic variables :
    //
    void
    FORT_INIT(const int* domLoU, const int* domHiU, 
	      const int* idxLoU, const int* idxHiU,
	      double* uVelocity, const double* uVelocityVal,
	      const int* domLoV, const int* domHiV, 
	      const int* idxLoV, const int* idxHiV,
	      double* vVelocity, const double* vVelocityVal,
	      const int* domLoW, const int* domHiW, 
	      const int* idxLoW, const int* idxHiW,
	      double* wVelocity, const double* wVelocityVal,
	      const int* domLo, const int* domHi, 
	      const int* idxLo, const int* idxHi,
	      double* pressure, const double* pressureVal,
	      double* density, const double* densityVal,
	      double* viscosity, const double* viscosityVal);

    ////////////////////////////////////////////////////////////////////////
    //
    // Initialize scalar variables :
    //
    void
    FORT_INIT_SCALAR(const int* domainLow, const int* domainHigh, 
		     const int* indexLow, const int* indexHigh,
		     double* scalar, const double* scalarVal);

    ////////////////////////////////////////////////////////////////////////
    //
    // Initialize celltype variables :
    //
    void
    FORT_CELLTYPEINIT(const int* domainLow, const int* domainHigh, 
		     const int* indexLow, const int* indexHigh,
		     int* celltype, const int* celltypeval);

    ////////////////////////////////////////////////////////////////////////
    //
    // Initialize geometry variables :
    //
    void
    FORT_CELLG(const int* domainLow, const int* domainHigh, 
	       const int* indexLow, const int* indexHigh,
	       double* sew,double* sns, double* stb,
	       double* sewu, double* snsv, double* stbw,
	       double* dxep, double* dynp, double* dztp,
	       double* dxepu,double*  dynpv,double*  dztpw,
	       double* dxpw, double* dyps, double* dzpb,
	       double* dxpwu, double* dypsv, double* dzpbw,
	       double* cee, double* cwe,double*  cww,
	       double* ceeu, double* cweu, double* cwwu,
	       double* cnn, double* csn,double*  css,
	       double* cnnv,double*  csnv, double* cssv,
	       double* ctt, double* cbt, double* cbb,
	       double* cttw, double* cbtw, double* cbbw,
	     //	     rr, ra, rv, rone,
	     //	     rcv, rcva,
	       double* xx,double*  xu, double* yy, double* yv, double* zz, double* zw,
	       double* efac, double* wfac, double* nfac, double* sfac,double*  tfac, double* bfac,
	       double* fac1u, double* fac2u, double* fac3u, double* fac4u,
	       double* fac1v, double* fac2v, double* fac3v, double* fac4v,
	       double* fac1w, double* fac2w, double* fac3w, double* fac4w,
	       int* iesdu, int* iwsdu, int* jnsdv, int* jssdv, int* ktsdw,
	       int*  kbsdw);



    ////////////////////////////////////////////////////////////////////////
    //
    // Initialize celltype variables :
    //
    void
    FORT_AREAIN(const int* domainLow, const int* domainHigh, 
		const int* indexLow, const int* indexHigh,
		  double* sew, double* sns,
		double* stb, double* area, int* celltype, 
		int* celltypeID);

    ////////////////////////////////////////////////////////////////////////
    //
    // Cset flat profiles:
    //
    void
    FORT_PROFV(const int* domLoU, const int* domHiU, 
	       const int* idxLoU, const int* idxHiU,
	       double* uVelocity, 
	       const int* domLoV, const int* domHiV, 
	       const int* idxLoV, const int* idxHiV,
	       double* vVelocity, 
	       const int* domLoW, const int* domHiW, 
	       const int* idxLoW, const int* idxHiW,
	       double* wVelocity, 
	       const int* domLo, const int* domHi, 
	       const int* idxLo, const int* idxHi,
	       int* celltype, double * area, const int* celltypeval,
	       double* flowrate, double* density);

    ////////////////////////////////////////////////////////////////////////
    //
    // set flat profiles for scalars:
    //
    void
    FORT_PROFSCALAR(const int* domainLow, const int* domainHigh, 
		    const int* indexLow, const int* indexHigh,
		    double* scalar, int* cellType,
		    double * sValue, const int* celltypeval);

    ////////////////////////////////////////////////////////////////////////
    //
    // turbulence model
    //
    void
    FORT_SMAGMODEL(const int* domLoU, const int* domHiU, 
		   double* uVelocity, 
		   const int* domLoV, const int* domHiV, 
		   double* vVelocity, 
		   const int* domLoW, const int* domHiW, 
		   double* wVelocity, 
		   const int* domLo, const int* domHi, 
		   double* density,
		   const int* domLoVis, const int* domHiVis, 
		   const int* idxLoVis, const int* idxHiVis,
		   double* viscosity,
		   double* sew, double * sns, double* stb, double* mol_visc,
		   double* cf, double* fac_msh, double* filterl);

    ////////////////////////////////////////////////////////////////////////
    //
    // Update inlet velocities in order to match total flow rates while
    // inlet area densities are changing
    //
    void
    FORT_INLBCS(const int* domLoU, const int* domHiU, 
		double* uVelocity, 
		const int* domLoV, const int* domHiV, 
		double* vVelocity, 
		const int* domLoW, const int* domHiW, 
		double* wVelocity, 
		const int* domLo, const int* domHi, 
		const int* idxLo, const int* idxHi, 
		const double* density,
		const int* cellType,
		const int* cellTypeVal);

    ////////////////////////////////////////////////////////////////////////
    //
    // set pressure BC:
    //
    void
    FORT_CALPBC(const int* domLoU, const int* domHiU, 
		double* uVelocity, 
		const int* domLoV, const int* domHiV, 
		double* vVelocity, 
		const int* domLoW, const int* domHiW, 
		double* wVelocity, 
		const int* domLo, const int* domHi, 
		const int* idxLo, const int* idxHi,
		double* pressure, double* density,
		int* celltype, const int* celltypeval,
		double* refPressure);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the U-velocity coeffs and convection coeffs
    //
    void
    FORT_UVELCOEF(const int* domLoU, const int* domHiU,
		  const int* idxLoU, const int* idxHiU,
		  const double* uVelocity,
		  double* uVelocityConvectCoeff_AE, 
		  double* uVelocityConvectCoeff_AW, 
		  double* uVelocityConvectCoeff_AN, 
		  double* uVelocityConvectCoeff_AS, 
		  double* uVelocityConvectCoeff_AT, 
		  double* uVelocityConvectCoeff_AB, 
		  double* uVelocityCoeff_AP,
		  double* uVelocityCoeff_AE,
		  double* uVelocityCoeff_AW,
		  double* uVelocityCoeff_AN,
		  double* uVelocityCoeff_AS,
		  double* uVelocityCoeff_AT,
		  double* uVelocityCoeff_AB,
		  double* variableCalledDU,
		  const int* domLoV, const int* domHiV,
		  const double* vVelocity,
		  const int* domLoW, const int* domHiW,
		  const double* wVelocity,
		  const int* domLo, const int* domHi,
		  const double* density,
		  const double* viscosity,
		  const double* deltaT,
		  const double* ceeu, const double* cweu, const double* cwwu,
		  const double* cnn, const double* csn, const double* css,
		  const double* ctt, const double* cbt, const double* cbb,
		  const double* sewu, const double* sew,
		  const double* sns, const double* stb,
		  const double* dxepu, const double* dxpwu,
		  const double* dxpw,
		  const double* dynp, const double* dyps,
		  const double* dztp, const double* dzpb,
		  const double* fac1u, const double* fac2u,
		  const double* fac3u, const double* fac4u,
		  const int* iesdu, const int* iwsdu, 
		  const double* nfac, const double* sfac,
		  const double* tfac, const double* bfac);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the V-velocity coeffs and convection coeffs
    //
    void
    FORT_VVELCOEF(const int* domLoV, const int* domHiV,
		  const int* idxLoV, const int* idxHiV,
		  const double* vVelocity,
		  double* vVelocityConvectCoeff_AE, 
		  double* vVelocityConvectCoeff_AW, 
		  double* vVelocityConvectCoeff_AN, 
		  double* vVelocityConvectCoeff_AS, 
		  double* vVelocityConvectCoeff_AT, 
		  double* vVelocityConvectCoeff_AB, 
		  double* vVelocityCoeff_AP,
		  double* vVelocityCoeff_AE,
		  double* vVelocityCoeff_AW,
		  double* vVelocityCoeff_AN,
		  double* vVelocityCoeff_AS,
		  double* vVelocityCoeff_AT,
		  double* vVelocityCoeff_AB,
		  double* variableCalledDV,
		  const int* domLoU, const int* domHiU,
		  const double* uVelocity,
		  const int* domLoW, const int* domHiW,
		  const double* wVelocity,
		  const int* domLo, const int* domHi,
		  const double* density,
		  const double* viscosity,
		  const double* deltaT,
		  const double* cee, const double* cwe, const double* cww,
		  const double* cnnv, const double* csnv, const double* cssv,
		  const double* ctt, const double* cbt, const double* cbb,
		  const double* sew, const double* snsv, const double* sns,
		  const double* stb,
		  const double* dxep, const double* dxpw,
		  const double* dynpv, const double* dypsv,
		  const double* dyps,
		  const double* dztp, const double* dzpb,
		  const double* fac1v, const double* fac2v,
		  const double* fac3v, const double* fac4v,
		  const int* jnsdv, const int* jssdv, 
		  const double* efac, const double* wfac,
		  const double* tfac, const double* bfac);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the W-velocity coeffs and convection coeffs
    //
    void
    FORT_WVELCOEF(const int* domLoW, const int* domHiW,
		  const int* idxLoW, const int* idxHiW,
		  const double* wVelocity,
		  double* wVelocityConvectCoeff_AE, 
		  double* wVelocityConvectCoeff_AW, 
		  double* wVelocityConvectCoeff_AN, 
		  double* wVelocityConvectCoeff_AS, 
		  double* wVelocityConvectCoeff_AT, 
		  double* wVelocityConvectCoeff_AB, 
		  double* wVelocityCoeff_AP,
		  double* wVelocityCoeff_AE,
		  double* wVelocityCoeff_AW,
		  double* wVelocityCoeff_AN,
		  double* wVelocityCoeff_AS,
		  double* wVelocityCoeff_AT,
		  double* wVelocityCoeff_AB,
		  double* variableCalledDW,
		  const int* domLoU, const int* domHiU,
		  const double* uVelocity,
		  const int* domLoV, const int* domHiV,
		  const double* vVelocity,
		  const int* domLo, const int* domHi,
		  const double* density,
		  const double* viscosity,
		  const double* deltaT,
		  const double* cee, const double* cwe, const double* cww,
		  const double* cnn, const double* csn, const double* css,
		  const double* cttw, const double* cbtw, const double* cbbw,
		  const double* sew, const double* sns, const double* stbw,
		  const double* stb, const double* dxep, const double* dxpw,
		  const double* dynp, const double* dyps,
		  const double* dztpw, const double* dzpbw,
		  const double* dzpb,
		  const double* fac1w, const double* fac2w,
		  const double* fac3w, const double* fac4w,
		  const int* ktsdw, const int* kbsdw, 
		  const double* efac, const double* wfac,
		  const double* nfac, const double* sfac);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the U-velocity linear and non-linear source terms
    //
    void
    FORT_UVELSOURCE(const int* domLoU, const int* domHiU,
		    const int* idxLoU, const int* idxHiU,
		    const double* uVelocity,  const double* old_uVelocity,
		    double* uvelnlinSrc, double* uvellinSrc,
		    const int* domLoV, const int* domHiV,
		    const double* vVelocity,
		    const int* domLoW, const int* domHiW,
		    const double* wVelocity,
		    const int* domLo, const int* domHi,
		    const double* density, const double* old_density,
		    const double* viscosity,
		    const double* gravity,
		    const double* deltaT, const double* den_ref,
		    const double* ceeu, const double* cweu, const double* cwwu,
		    const double* cnn, const double* csn, const double* css,
		    const double* ctt, const double* cbt, const double* cbb,
		    const double* sewu, const double* sew, const double* sns,
		    const double* stb,
		    const double* dxpw, 
		    const double* fac1u, const double* fac2u,
		    const double* fac3u, const double* fac4u,
		    const int* iesdu, const int* iwsdu);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the V-velocity linear and non-linear source terms
    //
    void
    FORT_VVELSOURCE(const int* domLoV, const int* domHiV,
		    const int* idxLoV, const int* idxHiV,
		    const double* vVelocity,  const double* old_vVelocity,
		    double* vvelnlinSrc, double* vvellinSrc,
		    const int* domLoU, const int* domHiU,
		    const double* uVelocity,
		    const int* domLoW, const int* domHiW,
		    const double* wVelocity,
		    const int* domLo, const int* domHi,
		    const double* density, const double* old_density,
		    const double* viscosity,
		    const double* gravity,
		    const double* deltaT, const double* den_ref,
		    const double* cee, const double* cwe, const double* cww,
		    const double* cnnv, const double* csnv, const double* cssv,
		    const double* ctt, const double* cbt, const double* cbb,
		    const double* sew, const double* snsv, const double* sns,
		    const double* stb,
		    const double* dyps, 
		    const double* fac1v, const double* fac2v,
		    const double* fac3v, const double* fac4v,
		    const int* jnsdv, const int* jssdv);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the W-velocity linear and non-linear source terms
    //
    void
    FORT_WVELSOURCE(const int* domLoW, const int* domHiW,
		    const int* idxLoW, const int* idxHiW,
		    const double* wVelocity,  const double* old_wVelocity,
		    double* wvelnlinSrc, double* wvellinSrc,
		    const int* domLoU, const int* domHiU,
		    const double* uVelocity,
		    const int* domLoV, const int* domHiV,
		    const double* vVelocity,
		    const int* domLo, const int* domHi,
		    const double* density, const double* old_density,
		    const double* viscosity,
		    const double* gravity,
		    const double* deltaT, const double* den_ref,
		    const double* cee, const double* cwe, const double* cww,
		    const double* cnn, const double* csn, const double* css,
		    const double* cttw, const double* cbtw, const double* cbbw,
		    const double* sew, const double* sns, const double* stbw,
		    const double* stb,
		    const double* dzpb, 
		    const double* fac1w, const double* fac2w,
		    const double* fac3w, const double* fac4w,
		    const int* ktsdw, const int* kbsdw);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the velocity mass source terms
    //
    void
    FORT_MASCAL(const int* domLo, const int* domHi,
		const int* idxLo, const int* idxHi,
		const double* velocity,  
		const double* velCoefAE,
		const double* velCoefAW,
		const double* velCoefAN,
		const double* velCoefAS,
		const double* velCoefAT,
		const double* velCoefAB,
		double* velNonLinSrc, double* velLinSrc,
		double* velConvectCoefAE,
		double* velConvectCoefAW,
		double* velConvectCoefAN,
		double* velConvectCoefAS,
		double* velConvectCoefAT,
		double* velConvectCoefAB);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the U-velocity bc
    //
    void
    FORT_BCUVEL(const int* domLoU, const int* domHiU,
		const int* idxLoU, const int* idxHiU,
		const double* uVelocity,
		double* uVelocityCoeff_AP,
		double* uVelocityCoeff_AE,
		double* uVelocityCoeff_AW,
		double* uVelocityCoeff_AN,
		double* uVelocityCoeff_AS,
		double* uVelocityCoeff_AT,
		double* uVelocityCoeff_AB,
		double* nlsource, double* linsource,
		const int* domLo, const int* domHi,
		const int* pcell,
		const int* wall, const int* ffield,
		const double* viscosity,
		const double* sewu, const double* sns, const double* stb,
		const double* yy, const double* yv,
		const double* zz, const double* zw);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the V-velocity bc
    //
    void
    FORT_BCVVEL(const int* domLoV, const int* domHiV,
		const int* idxLoV, const int* idxHiV,
		const double* vVelocity,
		double* vVelocityCoeff_AP,
		double* vVelocityCoeff_AE,
		double* vVelocityCoeff_AW,
		double* vVelocityCoeff_AN,
		double* vVelocityCoeff_AS,
		double* vVelocityCoeff_AT,
		double* vVelocityCoeff_AB,
		double* nlsource, double* linsource,
		const int* domLo, const int* domHi,
		const int* pcell,
		const int* wall, const int* ffield,
		const double* viscosity,
		const double* sew, const double* snsv, const double* stb,
		const double* xx, const double* xu,
		const double* zz, const double* zw);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the W-velocity bc
    //
    void
    FORT_BCWVEL(const int* domLoW, const int* domHiW,
		const int* idxLoW, const int* idxHiW,
		const double* wVelocity,
		double* wVelocityCoeff_AP,
		double* wVelocityCoeff_AE,
		double* wVelocityCoeff_AW,
		double* wVelocityCoeff_AN,
		double* wVelocityCoeff_AS,
		double* wVelocityCoeff_AT,
		double* wVelocityCoeff_AB,
		double* nlsource, double* linsource,
		const int* domLo, const int* domHi,
		const int* pcell,
		const int* wall, const int* ffield,
		const double* viscosity,
		const double* sew, const double* sns, const double* stbw,
		const double* xx, const double* xu,
		const double* yy, const double* yv);



    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the velocity diagonal
    //
    void
    FORT_APCAL(const int* domLo, const int* domHi,
	       const int* idxLo, const int* idxHi,
	       double* velCoefAP,
	       const double* velCoefAE,
	       const double* velCoefAW,
	       const double* velCoefAN,
	       const double* velCoefAS,
	       const double* velCoefAT,
	       const double* velCoefAB,
	       const double* velLinSrc); 

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the pressure stencil coefficients
    //
    void
    FORT_PRESSCOEFF(const int* domLo, const int* domHi,
		    const int* idxLo, const int* idxHi,
		    const double* density,
		    double* pressCoefAE,
		    double* pressCoefAW,
		    double* pressCoefAN,
		    double* pressCoefAS,
		    double* pressCoefAT,
		    double* pressCoefAB,
		    const int* domLoU, const int* domHiU,
		    const double* uVelCoefAP,
		    const int* domLoV, const int* domHiV,
		    const double* vVelCoefAP,
		    const int* domLoW, const int* domHiW,
		    const double* wVelCoefAP,
		    const double* sew, const double* sns, const double* stb,
		    const double* sewu, const double* dxep, const double* dxpw,
		    const double* snsv, const double* dynp, const double* dyps,
		    const double* stbw, const double* dztp, const double* dzpb); 

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the pressure source terms
    //
    void
    FORT_PRESSOURCE(const int* domLo, const int* domHi,
		    const int* idxLo, const int* idxHi,
		    double* pressureLinSrc,
		    double* pressureNonLinSrc,
		    const double* density, const double* old_density,
		    const int* domLoU, const int* domHiU,
		    const double* uVelocity,
		    const double* uVelCoefAP,
		    const double* uVelCoefAE,
		    const double* uVelCoefAW,
		    const double* uVelCoefAN,
		    const double* uVelCoefAS,
		    const double* uVelCoefAT,
		    const double* uVelCoefAB,
		    const double* uVelNonLinSrc,
		    const int* domLoV, const int* domHiV,
		    const double* vVelocity,
		    const double* vVelCoefAP,
		    const double* vVelCoefAE,
		    const double* vVelCoefAW,
		    const double* vVelCoefAN,
		    const double* vVelCoefAS,
		    const double* vVelCoefAT,
		    const double* vVelCoefAB,
		    const double* vVelNonLinSrc,
		    const int* domLoW, const int* domHiW,
		    const double* wVelocity,
		    const double* wVelCoefAP,
		    const double* wVelCoefAE,
		    const double* wVelCoefAW,
		    const double* wVelCoefAN,
		    const double* wVelCoefAS,
		    const double* wVelCoefAT,
		    const double* wVelCoefAB,
		    const double* wVelNonLinSrc,
		    const double* sew, const double* sns, const double* stb,
		    const int* cellType, const int* cellTypeID,
		    const double* delta_t);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the pressure BC
    //
    void
    FORT_PRESSBC(const int* domLo, const int* domHi,
		 const int* idxLo, const int* idxHi,
		 double* pressure,
		 double* pressCoeffAE,
		 double* pressCoeffAW,
		 double* pressCoeffAN,
		 double* pressCoeffAS,
		 double* pressCoeffAT,
		 double* pressCoeffAB,
		 double* pressNonlinearSrc,
		 double* pressLinearSrc,
		 int* cellType,
		 int* wall_celltypeval, int* symmetry_celltypeval,
		 int* flow_celltypeval);
    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the pressure grad for [u,v,w] source
    //
    void
    FORT_ADDPRESSGRAD(const int* domLoU, const int* domHiU,
		      const int* idxLo, const int* idxHiU,
		      const double* uVelocity,
		      double* nlsource, double* velcoeff_AP,
		      const int* domLo, const int* domHi,
		      const double* pressure,
		      const double* old_density,
		      const double* delta_t, const int* ioff, const int* joff,
		      const int* koff,
		      const double* sew, const double* sns, const double* stbw,
		      const double* dxpw);



    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the Mixing Scalar coeffs and convection terms
    //
    void
    FORT_SCALARCOEFF(const int* domLo, const int* domHi,
		     const int* idxLo, const int* idxHi,
		     const double* density,
		     const double* viscosity,
		     double* scalarCoeff_AE,
		     double* scalarCoeff_AW,
		     double* scalarCoeff_AN,
		     double* scalarCoeff_AS,
		     double* scalarCoeff_AT,
		     double* scalarCoeff_AB,
		     double* scalarConvectCoeff_AE, 
		     double* scalarConvectCoeff_AW, 
		     double* scalarConvectCoeff_AN, 
		     double* scalarConvectCoeff_AS, 
		     double* scalarConvectCoeff_AT, 
		     double* scalarConvectCoeff_AB, 
		     const int* domLoU, const int* domHiU,
		     const double* uVelocity,
		     const int* domLoV, const int* domHiV,
		     const double* vVelocity,
		     const int* domLoW, const int* domHiW,
		     const double* wVelocity,
		     const double* sew, const double* sns, const double* stb,
		     const double* cee, const double* cwe, const double* cww,
		     const double* cnn, const double* csn, const double* css,
		     const double* ctt, const double* cbt, const double* cbb,
		     const double* efac, const double* wfac,
		     const double* nfac, const double* sfac,
		     const double* tfac, const double* bfac,
		     const double* dxpw, const double* dyps, const double* dzpb);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the scalar source terms
    //
    void
    FORT_SCALARSOURCE(const int* domLo, const int* domHi,
		      const int* idxLo, const int* idxHi,
		      double* scalarLinSrc,
		      double* scalarNonLinSrc,
		      const double* old_density, const double* old_scalar,
		      const double* sew, const double* sns, const double* stb,
		      const double* delta_t);

    ////////////////////////////////////////////////////////////////////////
    //
    // Calculate the scalar BC
    //
    void
    FORT_SCALARBC(const int* domLo, const int* domHi,
		  const int* idxLo, const int* idxHi,
		  double* scalar,
		  double* scalarCoeffAE,
		  double* scalarCoeffAW,
		  double* scalarCoeffAN,
		  double* scalarCoeffAS,
		  double* scalarCoeffAT,
		  double* scalarCoeffAB,
		  double* scalarNonlinearSrc,
		  double* scalarLinearSrc,
		  double* density,
		  double* fmixin,
		  const int* domLoU, const int* domHiU,
		  double* uVelocity,
		  const int* domLoV, const int* domHiV,
		  double* vVelocity,
		  const int* domLoW, const int* domHiW,
		  double* wVelocity,
		  const double* sew, const double* sns, const double* stb,
		  int* cellType,
		  int* wall_celltypeval, int* symmetry_celltypeval,
		  int* flow_celltypeval, const int* ffield, const int* sfield,
		  const int* outletfield);

    ////////////////////////////////////////////////////////////////////////
    //
    // Compute the Residual of the Linearized System
    //
    void
    FORT_COMPUTERESID(const int* domLo, const int* domHi,
		      const int* idxLo, const int* idxHi,
		      double* variable,
		      double* residualArray,
		      double* coeffEast,
		      double* coeffWest,
		      double* coeffNorth,
		      double* coeffSouth,
		      double* coeffTop,
		      double* coeffBottom,
		      double* coeffDiagonal,
		      double* nonlinearSrc,
		      double* residualNorm);
}

#endif

//
// $Log$
// Revision 1.27  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.26  2000/07/30 22:59:31  bbanerje
// Added scalar source term calcs.
//
// Revision 1.25  2000/07/30 22:21:21  bbanerje
// Added bcscalar.F (originally bcf.f in Kumar's code) needs more work
// in C++ side.
//
// Revision 1.24  2000/07/28 02:30:59  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.23  2000/07/17 22:06:57  rawat
// modified momentum source
//
// Revision 1.22  2000/07/14 05:23:49  bbanerje
// Added scalcoef.F and updated related stuff in C++. scalcoef ==> coefs.f
// in Kumar's code.
//
// Revision 1.21  2000/07/14 03:45:44  rawat
// completed velocity bc and fixed some bugs
//
// Revision 1.20  2000/07/13 04:51:32  bbanerje
// Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
//
// Revision 1.19  2000/07/12 23:59:20  rawat
// added wall bc for u-velocity
//
// Revision 1.18  2000/07/12 23:23:23  bbanerje
// Added pressure source .. modified Kumar's version a bit.
//
// Revision 1.17  2000/07/12 22:15:01  bbanerje
// Added pressure Coef .. will do until Kumar's code is up and running
//
// Revision 1.16  2000/07/12 19:55:43  bbanerje
// Added apcal stuff in calcVelDiagonal
//
// Revision 1.15  2000/07/12 07:35:46  bbanerje
// Added stuff for mascal : Rawat: Labels and dataWarehouse in velsrc need to be corrected.
//
// Revision 1.14  2000/07/12 05:14:25  bbanerje
// Added vvelsrc and wvelsrc .. some changes to uvelsrc.
// Rawat :: Labels are getting hopelessly muddled unless we can do something
// about the time stepping thing.
//
// Revision 1.13  2000/07/11 15:46:26  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.12  2000/07/08 23:08:54  bbanerje
// Added vvelcoef and wvelcoef ..
// Rawat check the ** WARNING ** tags in these files for possible problems.
//
// Revision 1.11  2000/07/08 08:03:33  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.10  2000/07/07 23:07:44  rawat
// added inlet bc's
//
// Revision 1.9  2000/07/03 05:30:13  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.8  2000/06/30 04:19:16  rawat
// added turbulence model and compute properties
//
// Revision 1.7  2000/06/29 06:22:47  bbanerje
// Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
// changes to profv.  Code is broken until the changes are reflected
// thru all the files.
//
// Revision 1.6  2000/06/28 08:14:52  bbanerje
// Changed the init routines a bit.
//
// Revision 1.5  2000/06/20 20:42:36  rawat
// added some more boundary stuff and modified interface to IntVector. Before
// compiling the code you need to update /SCICore/Geometry/IntVector.h
//
// Revision 1.4  2000/06/15 23:47:56  rawat
// modified Archesfort to fix function call
//
// Revision 1.3  2000/06/15 22:13:21  rawat
// modified boundary stuff
//
// Revision 1.2  2000/06/14 20:40:48  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.1  2000/06/10 06:30:37  bbanerje
// Arches : Fortran wrappers in C++
//
//
