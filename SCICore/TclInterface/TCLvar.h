
/*
 *  TCLvar.h: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TCLvar_h
#define SCI_project_TCLvar_h 1

#include <SCICore/share/share.h>

#include <SCICore/Containers/String.h>

namespace SCICore {
  namespace Geometry {
    class Vector;
    class Point;
  }
}

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::clString;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

class TCL;

class SCICORESHARE TCLvar {
protected:
    clString varname;
    int is_reset;
    TCL* tcl;
public:
    TCLvar(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLvar();
    void reset();

    clString format_varname();

    clString str();
    virtual void emit(std::ostream& out)=0;
};

class SCICORESHARE TCLstring : public TCLvar {
    clString value;
public:
    TCLstring(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLstring();

    clString get();
    void set(const clString&);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLdouble : public TCLvar {
    double value;
public:
    TCLdouble(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLdouble();

    double get();
    void set(double);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLint : public TCLvar {
    int value;
public:
    TCLint(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLint();

    int get();
    void set(int);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLvardouble : public TCLvar {
    double value;
public:
    TCLvardouble(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLvardouble();

    double get();
    void set(double);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLvarint : public TCLvar {
    int value;
public:
    TCLvarint(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLvarint();

    int get();
    void set(int);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLvarintp : public TCLvar {
    int* value;
public:
    TCLvarintp(int*, const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLvarintp();

    int get();
    void set(int);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLPoint : public TCLvar {
    TCLdouble x;
    TCLdouble y;
    TCLdouble z;
public:
    TCLPoint(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLPoint();

    Point get();
    void set(const Point&);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE TCLVector : public TCLvar {
    TCLdouble x;
    TCLdouble y;
    TCLdouble z;
public:
    TCLVector(const clString& name, const clString& id, TCL* tcl);
    virtual ~TCLVector();

    Vector get();
    void set(const Vector&);
    virtual void emit(std::ostream& out);
};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/10/07 02:08:04  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:39:46  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:17  mcq
// Initial commit
//
// Revision 1.4  1999/05/17 17:14:47  kuzimmer
// Added the format_variable function from SCIRun
//
// Revision 1.3  1999/05/06 19:56:24  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:35  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
