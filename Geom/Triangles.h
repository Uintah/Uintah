
/*
 *  Triangles.h: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Triangles_h
#define SCI_Geom_Triangles_h 1

#include <Geom/VertexPrim.h>

class Color;

class GeomTriangles : public GeomVertexPrim {
private:
    void add(const Point&);
    void add(const Point&, const Vector&);
    void add(const Point&, const MaterialHandle&);
    void add(const Point&, const Vector&, const MaterialHandle&);
    void add(GeomVertex*);
public:
    Array1<Vector> normals;
    GeomTriangles();
    GeomTriangles(const GeomTriangles&);
    virtual ~GeomTriangles();

    int size(void);
    void add(const Point&, const Point&, const Point&);
    void add(const Point&, const Vector&,
	     const Point&, const Vector&,
	     const Point&, const Vector&);
    void add(const Point&, const MaterialHandle&,
	     const Point&, const MaterialHandle&,
	     const Point&, const MaterialHandle&);
    void add(const Point&, const Color&,
	     const Point&, const Color&,
	     const Point&, const Color&);
    void add(const Point&, const Vector&, const MaterialHandle&,
	     const Point&, const Vector&, const MaterialHandle&,
	     const Point&, const Vector&, const MaterialHandle&);
    void add(GeomVertex*, GeomVertex*, GeomVertex*);
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
};

#endif /* SCI_Geom_Triangles_h */
