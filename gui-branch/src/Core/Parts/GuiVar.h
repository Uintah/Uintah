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

#ifndef GuiVar_h
#define GuiVar_h 

#include <Core/share/share.h>
#include <Core/Containers/StringUtil.h>
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

class Part;
class PartPort;

class GuiVar {
protected:
  string name_;
  Part* part_;
public:
  GuiVar(const string& name, Part *part) ;
  virtual ~GuiVar();
  
  const string &name() { return name_; }
  void update();

  virtual void string_set( const string & ) {}
  virtual const string string_get() { return ""; }
  virtual void emit(std::ostream& out, string& midx)=0;

  // backward compatibility
  const string &str() { return name_; }
  void reset() {}
};
  
template<class T>
class GuiTypedVar : public GuiVar {
  T value_;
public:
  GuiTypedVar( const string &name, Part *part ) : GuiVar( name, part ) {}

  // backward compatibility
  GuiTypedVar( const string &name, const string &, Part *part ) 
    : GuiVar( name, part ) {}
  virtual ~GuiTypedVar() {}

  const T &get() { return value_; }
  void set( const T &v ) { value_ = v; update(); }
  virtual void string_set( const string &v ) { cerr << "set_string ignored\n";}
  virtual const string string_get() { return ""; }
  virtual void emit(std::ostream& out, string& midx) {
    out << "set " << midx << "-" << name_ << " {" << value_ << "}" << endl;
  }

  friend PartPort;
};

template<> void 
GuiTypedVar<string>::string_set( const string &value )
{
  value_ = value;
}

template<> const string 
GuiTypedVar<string>::string_get()
{
  return value_;
}

template<> void 
GuiTypedVar<int>::string_set( const string &value )
{
  string_to_int( value, value_ );
}

template<> const string
GuiTypedVar<int>::string_get()
{
  return to_string( value_ );
}

template<> void
GuiTypedVar<double>::string_set( const string &value )
{
  string_to_double( value, value_ );
}

template<> const string 
GuiTypedVar<double>::string_get()
{
  return to_string( value_ );
}



typedef GuiTypedVar<int> GuiInt;
typedef GuiTypedVar<double> GuiDouble;
typedef GuiTypedVar<string> GuiString;


typedef GuiTypedVar<double> GuiVardouble;  // NEED TO GET RID OF
typedef GuiTypedVar<int> GuiVarint;   // NEED TO GET RID OF

template <class T>
class GuiTriple : public GuiVar
{
  GuiDouble x_;
  GuiDouble y_;
  GuiDouble z_;
public:
  GuiTriple(const string& name, Part* part) :
    GuiVar(name, part),
    x_(name+"_x", part),
    y_(name+"_y", part),
    z_(name+"_z", part)
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


