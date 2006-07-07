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


//    File   : BasicIntergrators.h
//    Author : Allen R. Sanderson
//    Date   : July 2006


#if !defined(Math_BasicIntergrators_h)
#define Math_BasicIntergrators_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Datatypes/Field.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

using namespace std;

class SCISHARE BasicIntegrators
{
public:
  inline bool interpolate(const VectorFieldInterfaceHandle &vfi,
			  const Point &p,
			  Vector &v);


  int ComputeRKFTerms(Vector v[6],       // storage for terms
		      const Point &p,    // previous point
		      double s,          // current step size
		      const VectorFieldInterfaceHandle &vfi);

  void FindRKF(vector<Point> &v, // storage for points
	       Point x,          // initial point
	       double t2,        // square error tolerance
	       double s,         // initial step size
	       int n,            // max number of steps
	       const VectorFieldInterfaceHandle &vfi); // the field

  void FindHeun(vector<Point> &v, // storage for points
		Point x,          // initial point
		double t2,        // square error tolerance
		double s,         // initial step size
		int n,            // max number of steps
		const VectorFieldInterfaceHandle &vfi); // the field

  void FindRK4(vector<Point> &v,
	       Point x,
	       double t2,
	       double s,
	       int n,
	       const VectorFieldInterfaceHandle &vfi);

  void FindAdamsBashforth(vector<Point> &v, // storage for points
			  Point x,          // initial point
			  double t2,        // square error tolerance
			  double s,         // initial step size
			  int n,            // max number of steps
			  const VectorFieldInterfaceHandle &vfi); // the field

};

} // End namespace SCIRun

#endif // Math_BasicIntergrators_h

