
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
    int is_reset;
    clString varname;
    TCL* tcl;
public:
    TCLvar(const clString& name, const clString& id, TCL* tcl);
    ~TCLvar();
    void reset();
};

class TCLstring : public TCLvar {
    clString value;
public:
    TCLstring(const clString& name, const clString& id, TCL* tcl);
    ~TCLstring();

    clString get();
    void set(const clString&);
};

class TCLdouble : public TCLvar {
    double value;
public:
    TCLdouble(const clString& name, const clString& id, TCL* tcl);
    ~TCLdouble();

    double get();
    void set(double);
};

class TCLint : public TCLvar {
    int value;
public:
    TCLint(const clString& name, const clString& id, TCL* tcl);
    ~TCLint();

    int get();
    void set(int);
};

class TCLvardouble : public TCLvar {
    double value;
public:
    TCLvardouble(const clString& name, const clString& id, TCL* tcl);
    ~TCLvardouble();

    double get();
    void set(double);
};

class TCLvarint : public TCLvar {
    int value;
public:
    TCLvarint(const clString& name, const clString& id, TCL* tcl);
    ~TCLvarint();

    int get();
    void set(int);
};


#endif
