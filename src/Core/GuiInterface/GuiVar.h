
/*
 *  GuiVar.h: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_GuiVar_h
#define SCI_project_GuiVar_h 1

#include <Core/share/share.h>

#include <Core/Containers/String.h>

namespace SCIRun {
  class Vector;
  class Point;
}

namespace SCIRun {


class TCL;

class SCICORESHARE GuiVar {
protected:
    clString varname;
    int is_reset;
    TCL* tcl;
public:
    GuiVar(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVar();
    void reset();

    clString format_varname();

    clString str();
    virtual void emit(std::ostream& out)=0;
};

class SCICORESHARE GuiString : public GuiVar {
    clString value;
public:
    GuiString(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiString();

    clString get();
    void set(const clString&);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiDouble : public GuiVar {
    double value;
public:
    GuiDouble(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiDouble();

    double get();
    void set(double);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiInt : public GuiVar {
    int value;
public:
    GuiInt(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiInt();

    int get();
    void set(int);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiVardouble : public GuiVar {
    double value;
public:
    GuiVardouble(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVardouble();

    double get();
    void set(double);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiVarint : public GuiVar {
    int value;
public:
    GuiVarint(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVarint();

    int get();
    void set(int);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiVarintp : public GuiVar {
    int* value;
public:
    GuiVarintp(int*, const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVarintp();

    int get();
    void set(int);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiPoint : public GuiVar {
    GuiDouble x;
    GuiDouble y;
    GuiDouble z;
public:
    GuiPoint(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiPoint();

    Point get();
    void set(const Point&);
    virtual void emit(std::ostream& out);
};

class SCICORESHARE GuiVector : public GuiVar {
    GuiDouble x;
    GuiDouble y;
    GuiDouble z;
public:
    GuiVector(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVector();

    Vector get();
    void set(const Vector&);
    virtual void emit(std::ostream& out);
};

} // End namespace SCIRun


#endif
