/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
