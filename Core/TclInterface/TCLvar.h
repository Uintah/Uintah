
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

#include <Core/share/share.h>

#include <Core/Containers/String.h>

namespace SCIRun {
  class Vector;
  class Point;
}

namespace SCIRun {


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

} // End namespace SCIRun


#endif
