/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {
  class GuiContext;

  class GuiVar {
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
  // Returns true if variable exists in TCL scope and is of type T
  inline bool valid() {
    return ctx->get(value_);
  }

  inline bool changed() {
    ctx->reset();
    T temp;
    ctx->get(temp);
    ctx->reset();
    return temp != value_;
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
  // Returns true if triplet exists in TCL scope and are valid doubles
  inline bool valid() {
    return (x_.valid() && y_.valid() && z_.valid());
  }
};

typedef GuiTriple<Point> GuiPoint;
typedef GuiTriple<Vector> GuiVector;


// This class is equivalent to a GuiString, except that when
// SCIRUN_NET_SUBSTITUTE_DATADIR is set it does $SCIRUN_DATA and
// $SCIRUN_DATASET variable substitution when writing to networks.
//
class GuiFilename : public GuiString
{
public:
  GuiFilename(GuiContext* ctx) : GuiString(ctx)
  {
    ctx->doSubstituteDatadir();
  }

  GuiFilename(GuiContext* ctx, const string &val) : GuiString(ctx, val)
  {
    ctx->doSubstituteDatadir();
  }

  virtual ~GuiFilename() {}
};


} // End namespace SCIRun


#endif
