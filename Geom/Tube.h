/*
 *  Tube.h: Tube object
 *
 *  Written by:
 *   Han-Wei Shen
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tube_h 
#define SCI_Geom_Tube_h 1 

#include <Geom/Geom.h>
#include <Classlib/Array1.h>
#include <Geometry/Point.h>


class GeomTube: public GeomObj{

private:
    Array1<Point> make_circle(Point, double, Vector); 
public:
    Array1<Point>  pts;    // center points of circles
    Array1<double> rad;    // radius at each point
    Array1<Vector> normal; // the direction of each point

    GeomTube(); 
    GeomTube(const GeomTube&); 
    virtual ~GeomTube(); 

    virtual GeomObj* clone(); 
    virtual void get_bounds(BBox&); 
  
    int add(Point, double, Vector); 


#ifdef SCI_OPENGL
    virtual void objdraw(DrawInfoOpenGL*, Material*); 
#endif 
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /*SCI_Geom_Tube_h */
