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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geom/GeomSphere.h>

namespace SCICore {
namespace GeomSpace {


class GeomEllipsoid : public GeomSphere {
public:
    double m_tensor_matrix[16];
    double mev;
    GeomEllipsoid(){};
    GeomEllipsoid(const Point& point, double radius, int inu, int inv,
		  double* matrix, double mev, int index = 0x123456)
      : GeomSphere(point, radius, inu, inv, index), mev(mev)
      {
	for (short y = 0; y < 16; y++)
          m_tensor_matrix[y] = matrix[y]; 
	
      };

    virtual void draw(DrawInfoOpenGL*, Material*, double time);

    virtual ~GeomEllipsoid(){};
    virtual void get_bounds(BBox&);
    //virtual void get_bounds(BSphere&);

    static PersistentTypeID type_id;
};

} // End namespace GeomSpace
} // End namespace SCICore
#endif /* SCI_Geom_Ellipsoid_h */
