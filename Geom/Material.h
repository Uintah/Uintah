
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
    Material();
    Material(const Color&, const Color&, const Color&, double);
    Material(const Material&);
    ~Material();
    Material* clone();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

typedef LockingHandle<Material> MaterialHandle;

#endif
