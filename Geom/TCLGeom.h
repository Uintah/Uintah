
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

#include <TCL/TCLvar.h>

class Color;
class TCLColor : public TCLvar {
    TCLdouble r;
    TCLdouble g;
    TCLdouble b;
public:
    TCLColor(const clString& name, const clString& id, TCL* tcl);
    ~TCLColor();

    Color get();
    void set(const Color&);
};

class Material;
class TCLMaterial : public TCLvar {
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
};

#endif
