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

#include <Geom/VertexPrim.h>

class SinCosTable;

class GeomTube : public GeomVertexPrim {
    int nu;
    Array1<Vector> directions;
    Array1<double> radii;
private:
    void make_circle(int which, Array1<Point>& circle,
		     const SinCosTable& tab); 
public:
    GeomTube(int nu=10); 
    GeomTube(const GeomTube&); 
    virtual ~GeomTube(); 

    virtual GeomObj* clone(); 
    virtual void get_bounds(BBox&); 
    virtual void get_bounds(BSphere&);
  
    void add(GeomVertex*, double, const Vector&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time); 
#endif 
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /*SCI_Geom_Tube_h */
