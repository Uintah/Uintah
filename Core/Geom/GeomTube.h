/*
 *  GeomTube.h: Tube object
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

#include <Core/Geom/GeomVertexPrim.h>

class SinCosTable;

namespace SCIRun {

class SCICORESHARE GeomTube : public GeomVertexPrim {
    int nu;
    Array1<Vector> directions;
    Array1<double> radii;
private:
    void make_circle(int which, Array1<Point>& circle,
		     const SinCosTable& tab); 
public:
    GeomTube(int nu=8); 
    GeomTube(const GeomTube&); 
    virtual ~GeomTube(); 

    virtual GeomObj* clone(); 
    virtual void get_bounds(BBox&); 
  
    void add(GeomVertex*, double, const Vector&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time); 
#endif 

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun



#endif /*SCI_Geom_Tube_h */
