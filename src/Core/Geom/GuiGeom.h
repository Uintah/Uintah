
/*
 *  GuiGeom.h: Interface to TCL variables for Geom stuff
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GuiGeom_h
#define SCI_Geom_GuiGeom_h 1

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/String.h>

namespace SCIRun {


class Color;

class SCICORESHARE GuiColor : public GuiVar {
    GuiDouble r;
    GuiDouble g;
    GuiDouble b;
public:
    GuiColor(const clString& name, const clString& id, TCL* tcl);
    ~GuiColor();

    Color get();
    void set(const Color&);
    virtual void emit(std::ostream& out);
};

class Material;
class SCICORESHARE GuiMaterial : public GuiVar {
    GuiColor ambient;
    GuiColor diffuse;
    GuiColor specular;
    GuiDouble shininess;
    GuiColor emission;
    GuiDouble reflectivity;
    GuiDouble transparency;
    GuiDouble refraction_index;
 public:
    GuiMaterial(const clString& name, const clString& id, TCL* tcl);
    ~GuiMaterial();
   
    Material get();
    void set(const Material&);
    virtual void emit(std::ostream& out);
};

} // End namespace SCIRun


#endif // ifndef SCI_Geom_GuiGeom_h
