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
#include <Core/GuiInterface/GuiManager.h>
#include <Core/GuiInterface/Remote.h>
#include <string>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
using std::string;
using std::endl;

namespace SCIRun {
  class Vector;
  class Point;
}

namespace SCIRun {

//extern GuiManager* gm_;

class TCL;

class SCICORESHARE GuiVar {
protected:
  string varname_;
  int is_reset_;
  TCL* tcl;
public:
  GuiVar(const string& name, const string& id, TCL* tcl);
  virtual ~GuiVar();
  virtual void reset();

  string format_varname();

  string str();
  virtual void emit(std::ostream& out, string& midx)=0;
};
  
  
  
template <class T>
class GuiSingle : public GuiVar
{
  T value_;
public:
  GuiSingle(const string& name, const string& id, TCL* tcl) :
    GuiVar(name, id, tcl) {}
  
  virtual ~GuiSingle() {}

  inline T get() {
    return GuiManager::get(value_, varname_, is_reset_);
  }
  inline void set(const T value) {
    if(value != value_) {
      value_ = value;
      GuiManager::set(value_, varname_, is_reset_);
    }
  }
  virtual void emit(std::ostream& out, string& midx) {
    out << "set " << midx << "-" << format_varname() << " {"
	<< get() << "}" << endl;
  }
};

typedef GuiSingle<string> GuiString;
typedef GuiSingle<double> GuiDouble;
typedef GuiSingle<double> GuiVardouble;  // NEED TO GET RID OF
typedef GuiSingle<int> GuiInt;
typedef GuiSingle<int> GuiVarint;   // NEED TO GET RID OF

template <class T>
class GuiTriple : public GuiVar
{
  GuiDouble x_;
  GuiDouble y_;
  GuiDouble z_;
public:
  GuiTriple(const string& name, const string& id, TCL* tcl) :
    GuiVar(name, id, tcl),
    x_("x", str(), tcl),
    y_("y", str(), tcl),
    z_("z", str(), tcl)
  {}
  virtual ~GuiTriple() {}

  inline T get() {
    T result;
    result.x(x_.get());
    result.y(y_.get());
    result.z(z_.get());
    return result;
  }
  inline void set(const T var) {
    if((var.x() != x_.get()) || (var.y() != y_.get()) || (var.z() != z_.get())) {
      x_.set(var.x());
      y_.set(var.y());
      z_.set(var.z());
    }
  }
  virtual void emit(std::ostream& out, string& midx) {
    x_.emit(out, midx);
    y_.emit(out, midx);
    z_.emit(out, midx);
  }
};

typedef GuiTriple<Point> GuiPoint;
typedef GuiTriple<Vector> GuiVector;


} // End namespace SCIRun


#endif
