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
#define FORT_INLBCS inlbcs_
#define FORT_CALPBC calpbc_
#define FORT_BCUVEL bcuvel_
#define FORT_BCVVEL bcvvel_
#define FORT_BCWVEL bcwvel_
#define FORT_PRESSBC pressbc_
#define FORT_VELCOEF velcoef_
#define FORT_SCALARCOEF scalcof_
#define FORT_PRESSSOURCE psource_
#define FORT_VELSOURCE vsource_
#define FORT_SCALARSOURCE ssource_
#define FORT_APCAL apcal_
#define FORT_MASCAL mascal_
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
	       double* iesdu, double* iwsdu, double* jnsdv, double* jssdv, double* ktsdw,double*  kbsdw);



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
}

#endif

//
// $Log$
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
