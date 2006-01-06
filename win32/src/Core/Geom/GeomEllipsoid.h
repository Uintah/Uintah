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
 * Ellipsoid.h Ellipsoid objects
 * 
 * Modified from the sphere class
 * by Eric Lundberg for Ellipsoids 1999
 *
 *
 * Sphere.h: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Ellipsoid_h
#define SCI_Geom_Ellipsoid_h 1

#include <stdio.h>

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomSphere.h>

namespace SCIRun {


class GeomEllipsoid : public GeomSphere {
public:
    double m_tensor_matrix[16];
    double mev;

    GeomEllipsoid();
    GeomEllipsoid(const Point& point, double radius, int inu, int inv,
		  double* matrix, double mev);

    virtual void draw(DrawInfoOpenGL*, Material*, double time);

    virtual ~GeomEllipsoid();
    virtual void get_bounds(BBox&);

    static PersistentTypeID type_id;
};

} // End namespace SCIRun
#endif /* SCI_Geom_Ellipsoid_h */
