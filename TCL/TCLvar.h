
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

#include <Classlib/String.h>
class TCL;

class TCLvar {
protected:
    clString varname;
    int is_reset;
    TCL* tcl;
public:
    TCLvar(const clString& name, const clString& id, TCL* tcl);
    ~TCLvar();
    void reset();

    clString str();
    virtual void emit(ostream& out)=0;
};

class TCLstring : public TCLvar {
    clString value;
public:
    TCLstring(const clString& name, const clString& id, TCL* tcl);
    ~TCLstring();

    clString get();
    void set(const clString&);
    virtual void emit(ostream& out);
};

class TCLdouble : public TCLvar {
    double value;
public:
    TCLdouble(const clString& name, const clString& id, TCL* tcl);
    ~TCLdouble();

    double get();
    void set(double);
    virtual void emit(ostream& out);
};

class TCLint : public TCLvar {
    int value;
public:
    TCLint(const clString& name, const clString& id, TCL* tcl);
    ~TCLint();

    int get();
    void set(int);
    virtual void emit(ostream& out);
};

class TCLvardouble : public TCLvar {
    double value;
public:
    TCLvardouble(const clString& name, const clString& id, TCL* tcl);
    ~TCLvardouble();

    double get();
    void set(double);
    virtual void emit(ostream& out);
};

class TCLvarint : public TCLvar {
    int value;
public:
    TCLvarint(const clString& name, const clString& id, TCL* tcl);
    ~TCLvarint();

    int get();
    void set(int);
    virtual void emit(ostream& out);
};

class TCLvarintp : public TCLvar {
    int* value;
public:
    TCLvarintp(int*, const clString& name, const clString& id, TCL* tcl);
    ~TCLvarintp();

    int get();
    void set(int);
    virtual void emit(ostream& out);
};

class Point;
class TCLPoint : public TCLvar {
    TCLdouble x;
    TCLdouble y;
    TCLdouble z;
public:
    TCLPoint(const clString& name, const clString& id, TCL* tcl);
    ~TCLPoint();

    Point get();
    void set(const Point&);
    virtual void emit(ostream& out);
};

class Vector;
class TCLVector : public TCLvar {
    TCLdouble x;
    TCLdouble y;
    TCLdouble z;
public:
    TCLVector(const clString& name, const clString& id, TCL* tcl);
    ~TCLVector();

    Vector get();
    void set(const Vector&);
    virtual void emit(ostream& out);
};

#endif
