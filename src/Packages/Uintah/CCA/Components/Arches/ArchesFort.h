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
#define FORT_INIT init_
#define FORT_INIT_SCALAR initscal_
#define FORT_CELLTYPEINIT celltypeinit_
#define FORT_CELLG cellg_
#define FORT_AREAIN areain_
#define FORT_PROFV profv_
#define FORT_PROFSCALAR profscalar_
#define FORT_SMAGMODEL smagmodel_
#define FORT_SCALARVARMODEL scalarvarmodel_
#define FORT_CALPBC calpbc_
#define FORT_INLBCS inlbcs_
#define FORT_DENACCUM denaccum_
#define FORT_OUTAREA outarea_
#define FORT_BCINOUT bcinout_
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
#define FORT_MASCALSCALAR mascalscalar_
#define FORT_APCAL apcal_
#define FORT_APCAL_VEL apcalvel_
#define FORT_PRESSCOEFF prescoef_
#define FORT_PRESSOURCE pressrc_
#define FORT_PRESSBC bcpress_
#define FORT_ADDPRESSGRAD addpressgrad_
#define FORT_CALCPRESSGRAD calcpressgrad_
#define FORT_ADDPRESSUREGRAD addpressuregrad_
#define FORT_ADDTRANSSRC addtranssrc_
#define FORT_SCALARCOEFF scalcoef_
#define FORT_SCALARSOURCE scalsrc_
#define FORT_SCALARBC bcscalar_
#define FORT_COMPUTERESID rescal_
#define FORT_COLDPROPS cprops_
#define FORT_UNDERELAX underelax_
#define FORT_RBGLISOLV lisolv_
#define FORT_BCUTURB bcut_
#define FORT_BCVTURB bcvt_
#define FORT_BCWTURB bcwt_
#define FORT_LINEGS linegs_
#define FORT_NORMPRESS normpress_
#define FORT_EXPLICIT explicit_
// for multimaterial
#define FORT_MMMOMSRC mmmomsrc_
#define FORT_MMBCVELOCITY mmbcvelocity_
#define FORT_MMWALLBC mmwallbc_
#define FORT_MMCELLTYPEINIT mmcelltypeinit_
#define FORT_MM_MODIFY_PRESCOEF mm_modify_prescoef_
#define FORT_ADD_HYDRO_TO_PRESSURE add_hydrostatic_term_topressure_
// GROUP: Function Declarations:
////////////////////////////////////////////////////////////////////////

extern "C"
{
    ////////////////////////////////////////////////////////////////////////
    // Initialize basic variables :
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
    // Initialize scalar variables :
    void
    FORT_INIT_SCALAR(const int* domainLow, const int* domainHigh, 
		     const int* indexLow, const int* indexHigh,
		     double* scalar, const double* scalarVal);

    ////////////////////////////////////////////////////////////////////////
    // Initialize celltype variables :
    void
    FORT_CELLTYPEINIT(const int* domainLow, const int* domainHigh, 
		     const int* indexLow, const int* indexHigh,
		     int* celltype, const int* celltypeval);

    ////////////////////////////////////////////////////////////////////////
    // Initialize geometry variables :
    void
    FORT_CELLG(const int* domainLow, const int* domainHigh, 
	       const int* indexLow, const int* indexHigh,
	       const int* indexLowU, const int* indexHighU,
	       const int* indexLowV, const int* indexHighV,
	       const int* indexLowW, const int* indexHighW,
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
    // Initialize celltype variables :
    void
    FORT_AREAIN(const int* domainLow, const int* domainHigh, 
		const int* indexLow, const int* indexHigh,
		double* sew, double* sns,
		double* stb, double* area, int* celltype, 
		int* celltypeID,
		int* xminus, int* xplus, int* yminus, int* yplus,
		int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // Cset flat profiles:
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
	       double* flowrate, double* density,
	       int* xminus, int* xplus, int* yminus, int* yplus,
	       int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // set flat profiles for scalars:
    void
    FORT_PROFSCALAR(const int* domainLow, const int* domainHigh, 
		    const int* domainLowCT, const int* domainHighCT, 
		    const int* indexLow, const int* indexHigh,
		    double* scalar, int* cellType,
		    double * sValue, const int* celltypeval,
		    int* xminus, int* xplus, int* yminus, int* yplus,
		    int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // turbulence model
    void
    FORT_SMAGMODEL(const int* domLoU, const int* domHiU, 
		   double* uVelocity, 
		   const int* domLoV, const int* domHiV, 
		   double* vVelocity, 
		   const int* domLoW, const int* domHiW, 
		   double* wVelocity, 
		   const int* domLoDen, const int* domHiDen, 
		   double* density,
		   const int* domLoVis, const int* domHiVis, 
		   const int* idxLoVis, const int* idxHiVis,
		   double* viscosity,
		   const int* domLo, const int* domHi, 
		   double* sew, double * sns, double* stb, double* mol_visc,
		   double* cf, double* fac_msh, double* filterl);


    ////////////////////////////////////////////////////////////////////////
    // sub-grid scale scalar variance model
    void
    FORT_SCALARVARMODEL ( const int* domLoScalar, const int* domHiScalar, 
			  double* scalar,
			  const int* domLoScalarVar, const int* domHiScalarVar, 
			  const int* idxLoScalVar, const int* idxHiScalVar,
			  double* scalarVar,
			  const int* domLo, const int* domHi, 
			  double* dxpw, double * dyps, double* dzpb,
			  double* sew, double* sns, double* stb,
			  double* cfvar, double* fac_msh, double* filterl);

    ////////////////////////////////////////////////////////////////////////
    // Update inlet velocities in order to match total flow rates while
    // inlet area densities are changing
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
		const int* cellTypeVal,
		int* xminus, int* xplus, int* yminus, int* yplus,
		int* zminus, int* zplus);

    void
    FORT_BCINOUT(const int* domLoU, const int* domHiU, 
		 double* uVelocity, 
		 const int* domLoV, const int* domHiV, 
		 double* vVelocity, 
		 const int* domLoW, const int* domHiW, 
		 double* wVelocity, 
		 const int* domLoD, const int* domHiD, 
		 const int* idxLo, const int* idxHi, 
		 const double* density,
		 const int* domLo, const int* domHi, 
		 const int* cellType,
		 const int* cellTypeVal,
		 double* delta_t, double* flowin, double* flowout,
		 const double* sew,const double* sns,const double* stb,
		 int* xminus, int* xplus, int* yminus, int* yplus,
		 int* zminus, int* zplus);



    void
    FORT_DENACCUM(const int* domLoD, const int* domHiD, 
		  const int* idxLo, const int* idxHi, 
		  const double* density,
		  const int* domLoD_old, const int* domHiD_old, 
		  const double* old_den,
		  const int* domLo, const int* domHi, 
		  double* denAccum,
		  double* delta_t,
		  const double* sew,const double* sns,const double* stb);

    void
    FORT_OUTAREA(const int* domLo, const int* domHi, 
		 const int* idxLo, const int* idxHi, 
		 const int* cellType,
		 const int* domLoD, const int* domHiD, 
		 const double* density,
		 const double* sew,const double* sns,const double* stb,
		 double* areaOUT, const int* cellTypeVal,
		 int* xminus, int* xplus, int* yminus, int* yplus,
		 int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // set pressure BC:
    void
    FORT_CALPBC(const int* domLoU, const int* domHiU, 
		double* uVelocity, 
		const int* domLoV, const int* domHiV, 
		double* vVelocity, 
		const int* domLoW, const int* domHiW, 
		double* wVelocity, 
		const int* domLoden, const int* domHiden, 
		const int* domLopress, const int* domHipress, 
		const int* domLoct, const int* domHict, 
		const int* idxLo, const int* idxHi,
		double* pressure, double* density,
		int* celltype, const int* celltypeval,
		double* refPressuren,
		int* xminus, int* xplus, int* yminus, int* yplus,
		int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // Calculate the U-velocity coeffs and convection coeffs
    void
    FORT_UVELCOEF(const int* domLoU, const int* domHiU,
		  const int* domLoUng, const int* domHiUng,
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
		  //		  double* variableCalledDU,
		  const int* domLoV, const int* domHiV,
		  const double* vVelocity,
		  const int* domLoW, const int* domHiW,
		  const double* wVelocity,
		  const int* domLoeg, const int* domHieg,
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
    // Calculate the V-velocity coeffs and convection coeffs
    void
    FORT_VVELCOEF(const int* domLoV, const int* domHiV,
		  const int* domLoVng, const int* domHiVng,
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
		  //		  double* variableCalledDV,
		  const int* domLoU, const int* domHiU,
		  const double* uVelocity,
		  const int* domLoW, const int* domHiW,
		  const double* wVelocity,
		  const int* domLoeg, const int* domHieg,
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
    // Calculate the W-velocity coeffs and convection coeffs
    void
    FORT_WVELCOEF(const int* domLoW, const int* domHiW,
		  const int* domLoWng, const int* domHiWng,
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
		  //		  double* variableCalledDW,
		  const int* domLoU, const int* domHiU,
		  const double* uVelocity,
		  const int* domLoV, const int* domHiV,
		  const double* vVelocity,
		  const int* domLoeg, const int* domHieg,
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
    // Calculate the U-velocity linear and non-linear source terms
    void
    FORT_UVELSOURCE(const int* domLoU, const int* domHiU,
		    const int* domLoUng, const int* domHiUng,
		    const int* idxLoU, const int* idxHiU,
		    const double* uVelocity,  const double* old_uVelocity,
		    double* uvelnlinSrc, double* uvellinSrc,
		    const int* domLoV, const int* domHiV,
		    const double* vVelocity,
		    const int* domLoW, const int* domHiW,
		    const double* wVelocity,
		    const int* domLoeg, const int* domHieg,
		    const int* domLo, const int* domHi,
		    const double* density, 
		    const double* viscosity,
		    const int* domLong, const int* domHing,
		    const double* old_density,
		    const int* domLodenref, const int* domHidenref,
		    const double* den_ref,
		    const double* gravity,
		    const double* deltaT, 
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
    // Calculate the V-velocity linear and non-linear source terms
    void
    FORT_VVELSOURCE(const int* domLoV, const int* domHiV,
		    const int* domLoVng, const int* domHiVng,
		    const int* idxLoV, const int* idxHiV,
		    const double* vVelocity,  const double* old_vVelocity,
		    double* vvelnlinSrc, double* vvellinSrc,
		    const int* domLoU, const int* domHiU,
		    const double* uVelocity,
		    const int* domLoW, const int* domHiW,
		    const double* wVelocity,
		    const int* domLoeg, const int* domHieg,
		    const int* domLo, const int* domHi,
		    const double* density, 
		    const double* viscosity,
		    const int* domLong, const int* domHing,
		    const double* old_density,
		    const int* domLodenref, const int* domHidenref,
		    const double* den_ref,
		    const double* gravity,
		    const double* deltaT, 
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
    // Calculate the W-velocity linear and non-linear source terms
    void
    FORT_WVELSOURCE(const int* domLoW, const int* domHiW,
		    const int* domLoWng, const int* domHiWng,
		    const int* idxLoW, const int* idxHiW,
		    const double* wVelocity,  const double* old_wVelocity,
		    double* wvelnlinSrc, double* wvellinSrc,
		    const int* domLoU, const int* domHiU,
		    const double* uVelocity,
		    const int* domLoV, const int* domHiV,
		    const double* vVelocity,
		    const int* domLoeg, const int* domHieg,
		    const int* domLo, const int* domHi,
		    const double* density, 
		    const double* viscosity,
		    const int* domLong, const int* domHing,
		    const double* old_density,
		    const int* domLodenref, const int* domHidenref,
		    const double* den_ref,
		    const double* gravity,
		    const double* deltaT, 
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
    // Calculate the velocity mass source terms
    void
    FORT_MASCAL(const int* domLo, const int* domHi,
		const int* domLong, const int* domHing,
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
    // Calculate the velocity mass source terms
    void
    FORT_MASCALSCALAR(const int* domLo, const int* domHi,
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
    // Calculate the U-velocity bc
    void
    FORT_BCUVEL(const int* domLoU, const int* domHiU,
		const int* domLoUng, const int* domHiUng,
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
		const int* domLoV, const int* domHiV, 
		const int* idxLoV, const int* idxHiV,
		const double* vVelocity,
		const int* domLoW, const int* domHiW, 
		const int* idxLoW, const int* idxHiW,
		const double* wVelocity,
		const int* domLo, const int* domHi,
		const int* idxLo, const int* idxHi,
		const int* pcell,
		const int* wall, const int* ffield, const int* pfield,
		const double* viscosity,
		const double* sewu, const double* sns, const double* stb,
		const double* yy, const double* yv,
		const double* zz, const double* zw,
		int* xminus, int* xplus, int* yminus, int* yplus,
		int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // Calculate the V-velocity bc
    void
    FORT_BCVVEL(const int* domLoV, const int* domHiV,
		const int* domLoVng, const int* domHiVng,
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
		const int* domLoU, const int* domHiU, 
		const int* idxLoU, const int* idxHiU,
		const double* uVelocity,
		const int* domLoW, const int* domHiW, 
		const int* idxLoW, const int* idxHiW,
		const double* wVelocity,
		const int* domLo, const int* domHi,
		const int* idxLo, const int* idxHi,
		const int* pcell,
		const int* wall, const int* ffield, const int* pfield,
		const double* viscosity,
		const double* sew, const double* snsv, const double* stb,
		const double* xx, const double* xu,
		const double* zz, const double* zw,
		int* xminus, int* xplus, int* yminus, int* yplus,
		int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // Calculate the W-velocity bc
    void
    FORT_BCWVEL(const int* domLoW, const int* domHiW,
		const int* domLoWng, const int* domHiWng,
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
		const int* domLoU, const int* domHiU, 
		const int* idxLoU, const int* idxHiU,
		const double* uVelocity,
		const int* domLoV, const int* domHiV, 
		const int* idxLoV, const int* idxHiV,
		const double* vVelocity,
		const int* domLo, const int* domHi,
		const int* idxLo, const int* idxHi,
		const int* pcell,
		const int* wall, const int* ffield, const int* pfield,
		const double* viscosity,
		const double* sew, const double* sns, const double* stbw,
		const double* xx, const double* xu,
		const double* yy, const double* yv,
		int* xminus, int* xplus, int* yminus, int* yplus,
		int* zminus, int* zplus);



    ////////////////////////////////////////////////////////////////////////
    // Calculate the velocity diagonal
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
    // Calculate the velocity diagonal
    void
    FORT_APCAL_VEL(const int* domLo, const int* domHi,
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
    // Calculate the pressure stencil coefficients
    void
    FORT_PRESSCOEFF(const int* domLo, const int* domHi,
		    const int* domLong, const int* domHing,
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
    // Calculate the pressure source terms
    void
    FORT_PRESSOURCE(const int* domLo, const int* domHi,
		    const int* domLong, const int* domHing,
		    const int* idxLo, const int* idxHi,
		    double* pressureNonLinSrc,
		    double* pressureLinSrc,
		    const double* density, const double* old_density,
		    const int* domLoU, const int* domHiU,
		    const int* domLoUng, const int* domHiUng,
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
		    const int* domLoVng, const int* domHiVng,
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
		    const int* domLoWng, const int* domHiWng,
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
    // Calculate the pressure BC
    void
    FORT_PRESSBC(const int* domLo, const int* domHi,
		 const int* domLong, const int* domHing,
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
		 int* flow_celltypeval,
		 int* xminus, int* xplus, int* yminus, int* yplus,
		 int* zminus, int* zplus);
    ////////////////////////////////////////////////////////////////////////
    // Calculate the pressure grad for [u,v,w] source
    void
    FORT_ADDPRESSGRAD(const int* domLoU, const int* domHiU,
		      const int* domLoUng, const int* domHiUng,
		      const int* idxLo, const int* idxHiU,
		      const double* uVelocity,
		      double* nlsource, double* velcoeff_AP,
		      const int* domLo, const int* domHi,
		      const int* domLong, const int* domHing,
		      const double* pressure,
		      const double* old_density,
		      const double* delta_t, const int* ioff, const int* joff,
		      const int* koff,
		      const double* sew, const double* sns, const double* stbw,
		      const double* dxpw);


    ////////////////////////////////////////////////////////////////////////
    // Calculate the pressure grad for [u,v,w] source
    void
    FORT_ADDPRESSUREGRAD(const int* domLoU, const int* domHiU,
			 const int* domLoUng, const int* domHiUng,
			 const int* idxLo, const int* idxHiU,
			 const double* pressgrad,
			 double* nlsource, 
			 const int* domLo, const int* domHi,
			 const int* celltype, const int* mmwallid,
			 const int* ioff, const int* joff,
			 const int* koff);


    ////////////////////////////////////////////////////////////////////////
    // Calculate the pressure grad for [u,v,w] source
    void
    FORT_ADDTRANSSRC(const int* domLoU, const int* domHiU,
		     const int* domLoUng, const int* domHiUng,
		     const int* idxLo, const int* idxHiU,
		     const double* uVelocity,
		     double* nlsource, double* velcoeff_AP,
		     const int* domLo, const int* domHi,
		     const int* domLong, const int* domHing,
		     const double* old_density,
		     const double* delta_t,
		     const double* sew, const double* sns, const double* stbw);



    ////////////////////////////////////////////////////////////////////////
    // Calculate the pressure grad for [u,v,w] source
    void
    FORT_CALCPRESSGRAD(const int* domLoU, const int* domHiU,
		       const int* domLoUng, const int* domHiUng,
		       const int* idxLo, const int* idxHiU,
		       const double* uVelocity,
		       double* pressgradu,
		       const int* domLo, const int* domHi,
		       const double* pressure,
		       const int* ioff, const int* joff,
		       const int* koff,
		       const double* sew, const double* sns, const double* stbw,
		       const double* dxpw);




    ////////////////////////////////////////////////////////////////////////
    // Calculate the Mixing Scalar coeffs and convection terms
    void
    FORT_SCALARCOEFF(const int* domLo, const int* domHi,
		     const int* domLoVis, const int* domHiVis,
		     const int* domLong, const int* domHing,
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
		     const double* dxpw, const double* dxep, 
		     const double* dyps, const double* dynp,
		     const double* dzpb, const double* dztp);

    ////////////////////////////////////////////////////////////////////////
    // Calculate the scalar source terms
    void
    FORT_SCALARSOURCE(const int* domLo, const int* domHi,
		      const int* domLong, const int* domHing,
		      const int* idxLo, const int* idxHi,
		      double* scalarLinSrc,
		      double* scalarNonLinSrc,
		      const double* old_density, const double* old_scalar,
		      const double* sew, const double* sns, const double* stb,
		      const double* delta_t);

    ////////////////////////////////////////////////////////////////////////
    // Calculate the scalar BC
    void
    FORT_SCALARBC(const int* domLo, const int* domHi,
		  const int* domLong, const int* domHing,
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
		  int* flow_celltypeval, int* press_celltypeval, 
		  const int* ffield, const int* sfield,
		  const int* outletfield,
		  int* xminus, int* xplus, int* yminus, int* yplus,
		  int* zminus, int* zplus);

    ////////////////////////////////////////////////////////////////////////
    // Compute the Residual of the Linearized System
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
		      double* residualNorm,
		      double* truncNorm);

    ////////////////////////////////////////////////////////////////////////
    // underrelaxation of the eqn
  void
  FORT_UNDERELAX(const int* domLo, const int* domHi,
		   const int* domLong, const int* domHing,
		    const int* idxLo, const int* idxHi,
		    double* variable,
		    double* coeffDiagonal,
		    double* nonlinearSrc,
		    double* urelax);

  // linear solver
  void
  FORT_LINEGS(const int* domLo, const int* domHi,
	      const int* idxLo, const int* idxHi,
	      double* variable,
	      double* coeffEast,
	      double* coeffWest,
	      double* coeffNorth,
	      double* coeffSouth,
	      double* coeffTop,
	      double* coeffBottom,
	      double* coeffDiagonal,
	      double* nonlinearSrc,
	      double* e1, double* f1,
	      double* e2, double* f2,
	      double* e3, double* f3,
	      double* theta);
  //, bool* lswpwe,
  //	      bool* lswpsn, bool* lswpbt);
  
  // normalize pressure
  void
  FORT_NORMPRESS(const int* domLo, const int* domHi,
		 const int* idxLo, const int* idxHi,
		 double* pressure,
		 double* refPress);

  // explicit solver
  void 
  FORT_EXPLICIT(const int* domLo, const int* domHi,
		const int* domLong, const int* domHing,
		const int* idxLo, const int* idxHi,
		double* variable, double* old_variable,
		double* coeffEast,
		double* coeffWest,
		double* coeffNorth,
		double* coeffSouth,
		double* coeffTop,
		double* coeffBottom,
		double* coeffDiagonal,
		double* nonlinearSrc,
		const int* domLoDen, const int* domHiDen,
		const int* domLoDenwg, const int* domHiDenwg,
		double* old_density,
		double* sew, double* sns, double* stb,
		double* delta_t);
  
  // multimaterial functions
  ////////////////////////////////////////////////////////////////////////
  // Initialize mmwall celltype variables :
  void
  FORT_MMCELLTYPEINIT(const int* domainLow, const int* domainHigh, 
		      const int* indexLow, const int* indexHigh,
		      double* voidFrac, int* celltype, 
		      const int* mmwallid, const int* mmflowid, 
		      const double* cutoff);

  void
  FORT_MMMOMSRC(const int* domLoUng, const int* domHiUng,
		const int* idxLoU, const int* idxHiU,
		double* nlsource, double* linsource,
		const int* domLo, const int* domHi,
		double* mmnlsource, double* mmlinsource);
  
  ////////////////////////////////////////////////////////////////////////
  // Calculate the multimaterial velocity bc
  void
  FORT_MMBCVELOCITY(const int* domLoUng, const int* domHiUng,
		    const int* idxLoU, const int* idxHiU,
		    double* uVelocityCoeff_AE,
		    double* uVelocityCoeff_AW,
		    double* uVelocityCoeff_AN,
		    double* uVelocityCoeff_AS,
		    double* uVelocityCoeff_AT,
		    double* uVelocityCoeff_AB,
		    double* nlsource, double* linsource,
		    const int* domLo, const int* domHi,
		    const int* pcell,
		    const int* mmwallid, 
		    const int* ioff, const int* joff, const int* koff);
  
  ////////////////////////////////////////////////////////////////////////
  // Calculate the multimaterial scalar bc
  void
  FORT_MMWALLBC(const int* domLo, const int* domHi,
		const int* domLong, const int* domHing,
		const int* idxLo, const int* idxHi,
		double* uVelocityCoeff_AE,
		double* uVelocityCoeff_AW,
		double* uVelocityCoeff_AN,
		double* uVelocityCoeff_AS,
		double* uVelocityCoeff_AT,
		double* uVelocityCoeff_AB,
		double* nlsource, double* linsource,
		const int* pcell,
		const int* mmwallid);

  ////////////////////////////////////////////////////////////////////////
  // Modify pressure equation coefficients to account for voidage effects

  void 
  FORT_MM_MODIFY_PRESCOEF(
			  const int* dim_lo, const int* dim_hi,
			  const int* dim_lo_coef, const int* dim_hi_coef,
			  double* ae, 
			  double* aw,
			  double* an,
			  double* as,
			  double* at,
			  double* ab,
			  double* epsg,
			  const int* valid_lo,
			  const int* valid_hi);

  ////////////////////////////////////////////////////////////////////////
  // Add hydrostatic term to relative pressure

  void
  FORT_ADD_HYDRO_TO_PRESSURE(
			     const int* dim_lo, const int* dim_hi,
			     const int* dim_lo_ph, const int* dim_hi_ph,
			     const int* dim_lo_prel, const int* dim_hi_prel,
			     const int* dim_lo_den, const int* dim_hi_den,
			     double* p_plus_hydro, double* prel,
			     double* den_micro, 
			     double* gx, double* gy, double* gz,
			     double* xx, double* yy, double* zz,
			     const int* valid_lo, const int* valid_hi,
			     const int* pcell, const int* wall);

}

#endif

