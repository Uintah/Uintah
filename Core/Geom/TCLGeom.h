
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

#include <TclInterface/TCLvar.h>
#include <Containers/String.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::clString;
using namespace SCICore::TclInterface;

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
    virtual void emit(ostream& out);
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
    virtual void emit(ostream& out);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:52  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:14  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:13  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:18  dav
// Import sources
//
//

#endif // ifndef SCI_Geom_TCLGeom_h
