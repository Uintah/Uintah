/*
 *  CubicPolyRoots.h:
 *   Function for finding the real roots of a cubic polynomial.
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

// Passes back the real roots of the cubic polynomial:
// x^3 + b*x^2 + c*x + d
// and returns the number of real roots.  There will either
// be 3 real roots or one real root (in which case, only
// x0 is set and 1 is returned).
// Return values will only be 1 or 3.
int cubic_poly_roots(double b, double c, double d,
		     double& x0, double& x1, double& x2);




