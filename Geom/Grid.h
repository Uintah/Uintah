
/*
 *  Grid.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Grid_h
#define SCI_Geom_Grid_h 1

#include <Geom/VertexPrim.h>
#include <Classlib/Array2.h>

class GeomGrid : public GeomObj {
    Array2<double> verts;
    Array2<MaterialHandle> matls;
    Array2<Vector> normals;
    int have_matls;
    int have_normals;
    Point corner;
    Vector u, v, w;
public:
    GeomGrid(int, int, const Point&, const Vector&, const Vector&);
    GeomGrid(const GeomGrid&);
    virtual ~GeomGrid();

    virtual GeomObj* clone();

    void set(int, int, double);
    void set(int, int, double, const MaterialHandle&);
    void set(int, int, double, const Vector&);
    void set(int, int, double, const Vector&, const MaterialHandle&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /* SCI_Geom_Grid_h */
