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
#include <Core/Parts/Part.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiManager.h>
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

class SCICORESHARE GuiVar {
protected:
  string id_;
  string varname_;
  int is_reset_;
  Part* part;
public:
  GuiVar(const string& name, const string& id, Part *part);
  virtual ~GuiVar();
  virtual void reset();

  string format_varname();

  string str();
  virtual void emit(std::ostream& out, string& midx)=0;

  const string &get_var() 
  {
    return part->get_var( id_, varname_ );
//     string cmd = id_ + " get " + varname_;
//     static string res;
//     Part::tcl_eval( cmd, res );
//     return res;
  }

  void set_var( const string& v ) 
  {
    part->set_var( id_, varname_, v );
//     string cmd = id_ + " set " + varname_ + " " + v;
//     Part::tcl_execute( cmd );
  }
};
  
class GuiInt : public GuiVar {
  int value_;
public:
  GuiInt( const string &name, const string &id, Part *part ) :
    GuiVar( name, id, part ) {}
  virtual ~GuiInt() {}

  int get() { string_to_int( get_var(), value_ ); return value_; }
  void set( const int v ) { set_var( to_string( v ) ); }
  virtual void emit(std::ostream& out, string& midx) {
    out << "set " << midx << "-" << format_varname() << " {"
	<< get() << "}" << endl;
  }
};

class GuiDouble : public GuiVar {
  double value_;
public:
  GuiDouble( const string &name, const string &id, Part *part ) :
    GuiVar( name, id, part ) {}
  virtual ~GuiDouble() {}

  double get() { string_to_double( get_var(), value_ ); return value_; }
  void set( const double v ) { set_var( to_string( v ) ); }
  virtual void emit(std::ostream& out, string& midx) {
    out << "set " << midx << "-" << format_varname() << " {"
	<< get() << "}" << endl;
  }
};

class GuiString : public GuiVar {
  string value_;
public:
  GuiString( const string &name, const string &id, Part *part ) :
    GuiVar( name, id, part ) {}
  virtual ~GuiString() {}

  string get() { value_ = get_var(); return value_; }
  void set( const string &v ) { set_var( v ); }
  virtual void emit(std::ostream& out, string& midx) {
    out << "set " << midx << "-" << format_varname() << " {"
	<< get() << "}" << endl;
  }
};


template <class T>
class GuiSingle : public GuiVar
{
  T value_;
public:
  GuiSingle(const string& name, const string& id, Part* part) :
    GuiVar(name, id, part) {}
  
  virtual ~GuiSingle() {}

  inline T get() {
    cerr << "GET " << id_ << " :: " << varname_ << endl;
    return 0;
    //return gm->get(value_, varname_, is_reset_);
  }
  inline void set(const T value) {
    cerr << "SET " << id_ << " :: " << varname_ << " = " << value_ << endl;
//     if(value != value_) {
//       value_ = value;
//       gm->set(value_, varname_, is_reset_);
//     }
  }

  virtual void emit(std::ostream& out, string& midx) {
    out << "set " << midx << "-" << format_varname() << " {"
	<< get() << "}" << endl;
  }
};

#if 0
typedef GuiSingle<string> GuiString;
typedef GuiSingle<double> GuiDouble;
typedef GuiSingle<int> GuiInt;
#endif

typedef GuiSingle<double> GuiVardouble;  // NEED TO GET RID OF
typedef GuiSingle<int> GuiVarint;   // NEED TO GET RID OF

template <class T>
class GuiTriple : public GuiVar
{
  GuiDouble x_;
  GuiDouble y_;
  GuiDouble z_;
public:
  GuiTriple(const string& name, const string& id, Part* part) :
    GuiVar(name, id, part),
    x_("x", str(), part),
    y_("y", str(), part),
    z_("z", str(), part)
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
