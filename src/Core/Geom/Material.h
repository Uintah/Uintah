
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

//#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomContainer.h>
#include <SCICore/Multitask/ITC.h>

namespace SCICore {
namespace Containers {
class clString;
}
} 

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::clString;
using SCICore::Multitask::Mutex;

class SCICORESHARE Material : public Persistent {
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
    Material(const Color&);
    Material(const Material&);
    Material& operator=(const Material&);
    ~Material();
    Material* clone();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

typedef LockingHandle<Material> MaterialHandle;

class SCICORESHARE GeomMaterial : public GeomContainer {
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

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log
//

#endif
