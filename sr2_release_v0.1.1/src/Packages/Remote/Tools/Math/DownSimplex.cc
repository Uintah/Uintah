//////////////////////////////////////////////////////////////////////
// DownSimplex.cpp - Optimize the vector using Downhill Simplex.
// Code is based on Numerical Recipes, ch. 10.
// Modifications Copyright 1999 by David K. McAllister.
//////////////////////////////////////////////////////////////////////

#include <Packages/Remote/Tools/Math/DownSimplex.h>

#include <iostream>
using namespace std;

#include <math.h>

#define TINY 1.0e-10
#define NMAX 1000

namespace Remote {
// Sum each dimension's coordinates.
static inline HVector ComputePSum(HVector *p, int ndim)
{
	HVector psum(ndim);
	psum.zero();
	
	// Loop over the simplex, so ndim+1.
	for(int i=0; i<=ndim; i++)
	{
		psum += p[i];
	}
	
	return psum;
}

static inline double TrySimplex(HVector *p, HVector &y, HVector &psum, int ndim,
								double (*funk)(HVector &), int ihi, double fac)
{
	double fac1, fac2, ytry;
	
	HVector ptry(ndim);
	
	fac1 = (1.0-fac)/ndim;
	fac2 = fac1 - fac;
	
	// Compute a new vector to try.
	ptry = psum * fac1 - p[ihi] * fac2;
	
	// Try it.
	ytry = (*funk)(ptry);
	
	if (ytry < y[ihi])
	{
		// Replace the high one with this one.
		y[ihi] = ytry;
		
		psum += ptry - p[ihi];
		p[ihi] = ptry;
	}
	
	return ytry;
}

double DownSimplex(HVector *p, HVector &y, int ndim, double ftol,
				   double (*funk)(HVector &), int &nfunk)
{
	int i, ihi, ilo, inhi, mpts = ndim + 1;
	double rtol, ytry = y[0];
	
	nfunk = 0;
	HVector psum = ComputePSum(p, ndim);
	
	for(;;)
	{
		ilo = 0;
		ihi = y[0]>y[1] ? (inhi = 1, 0) : (inhi = 0, 1);
		for (i = 0; i < mpts; i++)
		{
			if (y[i] <= y[ilo]) ilo = i;
			if (y[i] > y[ihi])
			{
				inhi = ihi;
				ihi = i;
			}
			else if (y[i] > y[inhi] && i != ihi) inhi = i;
		}
		
		rtol = 2.0*fabs(y[ihi] - y[ilo]) / (fabs(y[ihi]) + fabs(y[ilo])+TINY);
		if (rtol < ftol)
		{
			double tmp = y[1]; y[1] = y[ilo]; y[ilo] = tmp;
			p[1].swap(p[ilo]);
			break;
		}
		
		if (nfunk >= NMAX)
		{
			cerr << "ERROR: Exceeded " << NMAX << " func evals.\n";
			break;
		}
		
		nfunk += 2;
		ytry = TrySimplex(p, y, psum, ndim, funk, ihi, -1.0);
		if(ytry <= y[ilo])
			ytry = TrySimplex(p, y, psum, ndim, funk, ihi, 2.0);
		else if (ytry >= y[inhi])
		{
			double ysave = y[ihi];
			ytry = TrySimplex(p, y, psum, ndim, funk, ihi, 0.5);
			if (ytry >= ysave)
			{
				for (i = 0; i < mpts; i++)
				{
					if (i != ilo)
					{
						p[i] = psum = (p[i] + p[ilo]) * 0.5;
						y[i] = (*funk)(psum);
					}
				}
				nfunk += ndim;
				psum = ComputePSum(p, ndim);
			}
			else
				nfunk--;
		}
	}
	
	return ytry;
}
} // End namespace Remote


