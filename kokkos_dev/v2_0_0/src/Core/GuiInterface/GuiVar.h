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
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {
  class GuiContext;

  class SCICORESHARE GuiVar {
  protected:
    GuiContext* ctx;
  public:
    GuiVar(GuiContext* ctx);
    virtual ~GuiVar();

    void reset();
  };
  
  
  
template <class T>
class GuiSingle : public GuiVar
{
  T value_;
public:
  GuiSingle(GuiContext* ctx) : GuiVar(ctx) {}
  GuiSingle(GuiContext* ctx, const T &val) : GuiVar(ctx), value_(val)
  {
    ctx->set(value_);
  }

  virtual ~GuiSingle() {}

  inline T get() {
    ctx->get(value_);
    return value_;
  }
  inline void set(const T value) {
    value_ = value;
    ctx->set(value_);
  }
};

typedef GuiSingle<string> GuiString;
typedef GuiSingle<double> GuiDouble;
typedef GuiSingle<int> GuiInt;


template <class T>
class GuiTriple : public GuiVar
{
  GuiDouble x_;
  GuiDouble y_;
  GuiDouble z_;
public:
  GuiTriple(GuiContext* ctx) :
    GuiVar(ctx),
    x_(ctx->subVar("x")),
    y_(ctx->subVar("y")),
    z_(ctx->subVar("z"))
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
};

typedef GuiTriple<Point> GuiPoint;
typedef GuiTriple<Vector> GuiVector;


} // End namespace SCIRun


#endif
