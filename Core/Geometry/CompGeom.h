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


/*
 *  CompGeom.h
 *
 *  Written by:
 *   Allen Sanderson
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 *  Copyright (C) 2005 SCI Group
 *
 */

#include <Core/Geometry/Point.h>

#include <Core/Geometry/share.h>

namespace SCIRun {


// Compute the distance squared from the point to the given line,
// where the line is specified by two end points.  This function
// actually computes the distance to the line segment
// between the given points and not to the line itself.
SHARE double
distance_to_line2(const Point &p, const Point &a, const Point &b);


// Compute the point on the triangle closest to the given point.
// The distance to the triangle will be (P - result).length())
SHARE void
closest_point_on_tri(Point &result, const Point &P,
                     const Point &A, const Point &B, const Point &C);

SHARE double
RayPlaneIntersection(const Point &p,  const Vector &dir,
		     const Point &p0, const Vector &pn);
}
