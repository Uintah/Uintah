
/*
 *  TCLGeom.h: Interface to TCL variables for Geom stuff
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_TCLGeom_h
#define SCI_Geom_TCLGeom_h 1

#include <Core/TclInterface/TCLvar.h>
#include <Core/Containers/String.h>

namespace SCIRun {


class Color;

class SCICORESHARE TCLColor : public TCLvar {
    TCLdouble r;
    TCLdouble g;
    TCLdouble b;
public:
    TCLColor(const clString& name, const clString& id, TCL* tcl);
    ~TCLColor();

    Color get();
    void set(const Color&);
    virtual void emit(std::ostream& out);
};

class Material;
class SCICORESHARE TCLMaterial : public TCLvar {
    TCLColor ambient;
    TCLColor diffuse;
    TCLColor specular;
    TCLdouble shininess;
    TCLColor emission;
    TCLdouble reflectivity;
    TCLdouble transparency;
    TCLdouble refraction_index;
 public:
    TCLMaterial(const clString& name, const clString& id, TCL* tcl);
    ~TCLMaterial();
   
    Material get();
    void set(const Material&);
    virtual void emit(std::ostream& out);
};

} // End namespace SCIRun


#endif // ifndef SCI_Geom_TCLGeom_h
