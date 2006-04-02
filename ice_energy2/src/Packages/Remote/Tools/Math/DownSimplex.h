//////////////////////////////////////////////////////////////////////
// DownSimplex.cpp - Optimize the vector using Downhill Simplex
// Code is based on Numerical Recipes, ch. 10.
// Modifications Copyright 1999 by David K. McAllister.
//////////////////////////////////////////////////////////////////////

#ifndef _downsimplex_h
#define _downsimplex_h

#include <Packages/Remote/Tools/Math/HVector.h>

namespace Remote {
extern double DownSimplex(HVector *p, HVector &y, int ndim, double ftol,
	      double (*funk)(HVector &), int &nfunk);

} // End namespace Remote


#endif
