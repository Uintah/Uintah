
/*
 *  Material.h:  Material Properities for Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Material_h
#define SCI_Geom_Material_h 1

#include <Classlib/Persistent.h>
#include <Classlib/LockingHandle.h>
#include <Geom/Color.h>
#include <Geom/Container.h>
#include <Multitask/ITC.h>

class Material : public Persistent {
public:
    int ref_cnt;
    Mutex lock;

    Color ambient;
    Color diffuse;
    Color specular;
    double shininess;
    Color emission;
    double reflectivity;
    double transparency;
    double refraction_index;

    Material();
    Material(const Color&, const Color&, const Color&, double);
    Material(const Material&);
    Material& operator=(const Material&);
    ~Material();
    Material* clone();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

typedef LockingHandle<Material> MaterialHandle;

class GeomMaterial : public GeomContainer {
    MaterialHandle matl;
public:
    GeomMaterial(GeomObj*, const MaterialHandle&);
    GeomMaterial(const GeomMaterial&);
    void setMaterial(const MaterialHandle&);
    MaterialHandle getMaterial();
    virtual ~GeomMaterial();
    virtual GeomObj* clone();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);

    // For Raytracing
    virtual void intersect(const Ray& ray, Material* matl,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif
