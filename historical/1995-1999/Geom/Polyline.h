
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

#include <Geom/VertexPrim.h>
#include <Geometry/BBox.h>

class GeomPolyline : public GeomVertexPrim {
public:
    GeomPolyline();
    GeomPolyline(const GeomPolyline&);
    virtual ~GeomPolyline();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

class GeomPolylineTC: public GeomObj {
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
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};


#endif /* SCI_Geom_Polyline_h */
