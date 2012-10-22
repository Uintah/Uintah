/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

/*
 *  CompGeom.h
 *
 *  Written by:
 *   Allen Sanderson
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 *
 */

#include <Core/Geometry/Point.h>
#include <Core/Math/MusilRNG.h>
#include <vector>

#include <Core/Geometry/share.h>

namespace SCIRun {


// Compute the distance squared from the point to the given line,
// where the line is specified by two end points.  This function
// actually computes the distance to the line segment
// between the given points and not to the line itself.
SCISHARE double
distance_to_line2(const Point &p, const Point &a, const Point &b);


// Compute the point on the triangle closest to the given point.
// The distance to the triangle will be (P - result).length())
SCISHARE void
closest_point_on_tri(Point &result, const Point &P,
                     const Point &A, const Point &B, const Point &C);

SCISHARE double
RayPlaneIntersection(const Point &p,  const Vector &dir,
		     const Point &p0, const Vector &pn);


SCISHARE bool
RayTriangleIntersection(double &t, double &u, double &v, bool backface_cull,
                        const Point &orig,  const Vector &dir,
                        const Point &p0, const Point &p1, const Point &p2);


// Compute s and t such that the distance between A0 + s * (A1 - AO)
// and B0 + t * (B1 - B0) is minimal.  Return false if the lines are
// parallel, true otherwise.
SCISHARE bool
closest_line_to_line(double &s, double &t,
                     const Point &A0, const Point &A1,
                     const Point &B0, const Point &B1);


SCISHARE void
TriTriIntersection(const Point &A0, const Point &A1, const Point &A2,
                   const Point &B0, const Point &B1, const Point &B2,
                   std::vector<Point> &results);



SCISHARE void
uniform_sample_triangle(Point &p, const Point &p0,
                        const Point &p1, const Point &p2,
                        MusilRNG &rng);


double
tetrahedra_volume(const Point &p0, const Point &p1,
                  const Point &p2, const Point &p3);


SCISHARE void
uniform_sample_tetrahedra(Point &p, const Point &p0, const Point &p1,
                          const Point &p2, const Point &p3,
                          MusilRNG &rng);


} // namespace SCIRun
