
/*
 *  tGrid.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_tGrid_h
#define SCI_Geom_tGrid_h 1

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Classlib/Array2.h>

class TexGeomGrid : public GeomObj {
    int tmap_size;
    int tmap_dlist;
    Point corner;
    Vector u, v, w;
    void adjust();

    unsigned char* tmapdata; // texture map
    int MemDim;
    int dimU,dimV;
public:
    TexGeomGrid(int, int, const Point&, const Vector&, const Vector&);
    TexGeomGrid(const TexGeomGrid&);
    virtual ~TexGeomGrid();

    virtual GeomObj* clone();

    void set(unsigned char* data,int datadim);
    
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

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif /* SCI_Geom_Grid_h */
