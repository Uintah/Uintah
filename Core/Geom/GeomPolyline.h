
/*
 *  Polyline.h: Polyline object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Polyline_h
#define SCI_Geom_Polyline_h 1

#include <Core/Geom/GeomVertexPrim.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomPolyline : public GeomVertexPrim {
public:
    GeomPolyline();
    GeomPolyline(const GeomPolyline&);
    virtual ~GeomPolyline();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomPolylineTC: public GeomObj {
protected:
    Array1<float> data;
    BBox bbox;
    int drawmode;
    double drawdist;
public:
    GeomPolylineTC(int drawmode, double drawdist);
    GeomPolylineTC(const GeomPolylineTC& copy);
    virtual ~GeomPolylineTC();

    void add(double t, const Point&, const Color&);
    
    virtual GeomObj* clone();

    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun


#endif /* SCI_Geom_Polyline_h */
