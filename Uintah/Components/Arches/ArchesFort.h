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
#define FORT_PROFV profv_
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
#define FORT_SMAGMODEL smodel_
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
    FORT_INIT(const int* domainLow, const int* domainHigh, 
	      const int* indexLow, const int* indexHigh,
	      double* uVelocity, const double* uVelocityVal,
	      double* vVelocity, const double* vVelocityVal,
	      double* wVelocity, const double* wVelocityVal,
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
}


#endif

//
// $Log$
// Revision 1.1  2000/06/10 06:30:37  bbanerje
// Arches : Fortran wrappers in C++
//
//
