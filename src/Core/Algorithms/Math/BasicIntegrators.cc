/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : BasicIntegrators.cc
//    Author : Allen R. Sanderson
//    Date   : July 2006


#include <Core/Algorithms/Math/BasicIntegrators.h>

namespace SCIRun {

using namespace std;

//! interpolate using the generic linear interpolator
bool
BasicIntegrators::interpolate(const VectorFieldInterfaceHandle &vfi,
			      const Point &p,
			      Vector &v)
{
  return vfi->interpolate(v, p) && (v.safe_normalize() > 0.0);
}


// LUTs for the RK-fehlberg algorithm 
static const double rkf_a[] =
  {16.0/135, 0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
static const double rkf_ab[] =
  {1.0/360, 0, -128.0/4275, -2197.0/75240, 1.0/50, 2.0/55};
//static const double rkf_c[] =
//  {0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2}; // Not used, keep for reference.
static const double rkf_d[][5]=
  {{0, 0, 0, 0, 0},
   {1.0/4, 0, 0, 0, 0},
   {3.0/32, 9.0/32, 0, 0, 0},
   {1932.0/2197, -7200.0/2197, 7296.0/2197, 0, 0},
   {439.0/216, -8.0, 3680.0/513, -845.0/4104, 0},
   {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40}};

int
BasicIntegrators::ComputeRKFTerms(Vector v[6],       // storage for terms
				  const Point &p,    // previous point
				  double s,          // current step size
				  const VectorFieldInterfaceHandle &vfi)
{
  // Already computed this one when we did the inside test.
  //  if (!interpolate(vfi, p, v[0]))
  //  {
  //    return -1;
  //  }
  v[0] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[1][0], v[1]))
    return 0;

  v[1] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[2][0] + v[1]*rkf_d[2][1], v[2]))
    return 1;

  v[2] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[3][0] + v[1]*rkf_d[3][1] +
		   v[2]*rkf_d[3][2], v[3]))
    return 2;

  v[3] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[4][0] + v[1]*rkf_d[4][1] +
		   v[2]*rkf_d[4][2] + v[3]*rkf_d[4][3], v[4]))
    return 3;

  v[4] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[5][0] + v[1]*rkf_d[5][1] +
		   v[2]*rkf_d[5][2] + v[3]*rkf_d[5][3] +
		   v[4]*rkf_d[5][4], v[5]))
    return 4;

  v[5] *= s;

  return 5;
}


void
BasicIntegrators::FindRKF(vector<Point> &v, // storage for points
			  Point x,          // initial point
			  double t2,        // square error tolerance
			  double s,         // initial step size
			  int n,            // max number of steps
			  const VectorFieldInterfaceHandle &vfi) // the field
{
  Vector terms[6];

  if (!interpolate(vfi, x, terms[0]))
    return;

  for (int i=0; i<n; i++) {
    // Compute the next set of terms.
    if (ComputeRKFTerms(terms, x, s, vfi) < 5) {
      s /= 1.5;
      continue;
    }

    // Compute the approximate local truncation error.
    const Vector err = terms[0]*rkf_ab[0] + terms[1]*rkf_ab[1]
      + terms[2]*rkf_ab[2] + terms[3]*rkf_ab[3] + terms[4]*rkf_ab[4]
      + terms[5]*rkf_ab[5];
    const double err2 = err.length2();
    
    // Is the error tolerable?  Adjust the step size accordingly.  Too
    // small?  Grow it for next time but keep small-error result.  Too
    // big?  Recompute with smaller size.
    if (err2 * 16384.0 < t2) {
      s *= 2.0;

    } else if (err2 > t2) {
      s /= 2.0;
      continue;
    }

    // Compute and add the point to the list of points found.
    x = x  +  terms[0]*rkf_a[0] + terms[1]*rkf_a[1] + terms[2]*rkf_a[2] + 
      terms[3]*rkf_a[3] + terms[4]*rkf_a[4] + terms[5]*rkf_a[5];

    // If the new point is inside the field, add it.  Otherwise stop.
    if (!interpolate(vfi, x, terms[0]))
      break;

    v.push_back(x);
  }
}


void
BasicIntegrators::FindHeun(vector<Point> &v, // storage for points
			   Point x,          // initial point
			   double t2,        // square error tolerance
			   double s,         // initial step size
			   int n,            // max number of steps
			   const VectorFieldInterfaceHandle &vfi) // the field
{
  int i;
  Vector v0, v1;

  if (!interpolate(vfi, x, v0))
    return;

  for (i=0; i < n; i ++) {
    v0 *= s;
    if (!interpolate(vfi, x + v0, v1))
      break;

    v1 *= s;
    x += 0.5 * (v0 + v1);

    if (!interpolate(vfi, x, v0))
      break;

    v.push_back(x);
  }
}


void
BasicIntegrators::FindRK4(vector<Point> &v,
			  Point x,
			  double t2,
			  double s,
			  int n,
			  const VectorFieldInterfaceHandle &vfi)
{
  Vector f[4];
  int i;

  if (!interpolate(vfi, x, f[0]))
    return;

  for (i = 0; i < n; i++) {
    f[0] *= s;
    if (!interpolate(vfi, x + f[0] * 0.5, f[1]))
      break;

    f[1] *= s;
    if (!interpolate(vfi, x + f[1] * 0.5, f[2]))
      break;

    f[2] *= s;
    if (!interpolate(vfi, x + f[2], f[3]))
      break;

    f[3] *= s;

    x += (f[0] + 2.0 * f[1] + 2.0 * f[2] + f[3]) * (1.0 / 6.0);

    // If the new point is inside the field, add it.  Otherwise stop.
    if (!interpolate(vfi, x, f[0]))
      break;
    v.push_back(x);
  }
}


void
BasicIntegrators::FindAdamsBashforth(vector<Point> &v, // storage for points
				     Point x,          // initial point
				     double t2,        // square error tolerance
				     double s,         // initial step size
				     int n,            // max number of steps
				     const VectorFieldInterfaceHandle &vfi) // the field
{
  FindRK4(v, x, t2, s, Min(n, 5), vfi);

  if (v.size() < 5) {
    return;
  }

  Vector f[5];
  int i;

  for (i = 0; i < 5; i++)
    interpolate(vfi, v[v.size() - 1 - i], f[i]);
  
  x = v[v.size() - 1];
  
  for (i = 5; i < n; i++) {
    x += (s/720.) * (1901.0 * f[0] - 2774.0 * f[1] +
		     2616.0 * f[2] - 1274.0 * f[3] +
		     251.0 * f[4]);

    f[4] = f[3];
    f[3] = f[2];
    f[2] = f[1];
    f[1] = f[0];

    if (!interpolate(vfi, x, f[0])) {
      break; 
    }

    v.push_back(x);
  }
}

}
