
/*
 * RenderMode.h:  Object to switch rendering mode
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_RenderMode_h
#define SCI_Geom_RenderMode_h 1

#include <Geom/Container.h>
#include <Geometry/Point.h>

class GeomRenderMode : public GeomContainer {
public:
    enum DrawType {
        WireFrame,
        Flat,
        Gouraud,
        Phong,
    };
private:
    DrawType drawtype;
public:
    GeomRenderMode(DrawType, GeomObj* child);
    GeomRenderMode(const GeomRenderMode&);
    virtual ~GeomRenderMode();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif /* SCI_Geom_RenderMode_h */
