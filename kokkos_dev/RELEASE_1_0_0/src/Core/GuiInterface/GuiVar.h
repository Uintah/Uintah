/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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
    virtual void reset();

    clString format_varname();

    clString str();
    virtual void emit(std::ostream& out, clString& midx)=0;
};

class SCICORESHARE GuiString : public GuiVar {
    clString value;
public:
    GuiString(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiString();

    clString get();
    void set(const clString&);
    virtual void emit(std::ostream& out, clString& midx);
};

class SCICORESHARE GuiDouble : public GuiVar {
    double value;
public:
    GuiDouble(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiDouble();

    double get();
    void set(double);
    virtual void emit(std::ostream& out, clString& midx);
};

class SCICORESHARE GuiInt : public GuiVar {
    int value;
public:
    GuiInt(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiInt();

    int get();
    void set(int);
    virtual void emit(std::ostream& out, clString& midx);
};

class SCICORESHARE GuiVardouble : public GuiVar {
    double value;
public:
    GuiVardouble(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVardouble();

    double get();
    void set(double);
    virtual void emit(std::ostream& out, clString& midx);
};

class SCICORESHARE GuiVarint : public GuiVar {
    int value;
public:
    GuiVarint(const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVarint();

    int get();
    void set(int);
    virtual void emit(std::ostream& out, clString& midx);
};

class SCICORESHARE GuiVarintp : public GuiVar {
    int* value;
public:
    GuiVarintp(int*, const clString& name, const clString& id, TCL* tcl);
    virtual ~GuiVarintp();

    int get();
    void set(int);
    virtual void emit(std::ostream& out, clString& midx);
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
    virtual void emit(std::ostream& out, clString& midx);
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
    virtual void emit(std::ostream& out, clString& midx);
};

} // End namespace SCIRun


#endif
