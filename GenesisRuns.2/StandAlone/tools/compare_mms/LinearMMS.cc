/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <StandAlone/tools/compare_mms/LinearMMS.h>

#include <cmath>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!

LinearMMS::LinearMMS(double cu, double cv, double cw, double cp, double p_ref) {
	c_u=cu;
	c_v=cv;
	c_w=cw;
	c_p=cp;
	p_ref_=p_ref;
};
LinearMMS::~LinearMMS() {
};

double
LinearMMS::pressure( double x, double y, double z, double time )
{
  return p_ref_ + c_p*( x + y + z) + time;
}

double
LinearMMS::uVelocity( double x, double y, double z, double time )
{
  return c_u*x + time;
}
  
double
LinearMMS::vVelocity( double x, double y, double z, double time )
{
  return c_v*y + time;
}

double
LinearMMS::wVelocity( double x, double y, double z, double time )
{
  return c_w*z + time;
}

