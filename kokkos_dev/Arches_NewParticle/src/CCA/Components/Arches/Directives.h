// To include these directives in your class, put the following line in the .h file:
//#include <CCA/Components/Arches/Directives.h>

// used by:
// CoalModels/*
// TransportEqns/*
// ChemMix/*
// DQMOM
#define TINY    1e-10

// used by:
// DQMOM
//#define DEBUG_MODELS
//#define VERIFY_LINEAR_SOLVER
//#define VERIFY_AB_CONSTRUCTION
//#define DEBUG_MATRICES

// used by:
// BoundaryCond_new
// DQMOMEqn
// Discretization_new
// EqnBase
#define XDIM
#define YDIM
#define ZDIM

// used by: 
// DQMOMEqn
//#define VERIFY_DQMOM_TRANSPORT

// used by:
// ExplicitTimeInt
//#define VERIFY_TIMEINT

// used by:
// MMS_X
// MMS_Y
// MMS_Z
// MMS_XYZ
//#define VERIFICATION

