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
 *  AxesObj.h: Displayable 2D object
 *
 *  Written by:
 *   Chris
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_AxesObj_h
#define SCI_AxesObj_h 

#include <Core/2d/HairObj.h>

namespace SCIRun {
  
class XAxisObj :  public HairObj {
protected:
    int num_tics_;

public:
  
  XAxisObj( const string &name="xaxis" )  
    : HairObj(name), num_tics_(7) { pos_ = .5; }
  virtual ~XAxisObj();

  virtual void get_bounds( BBox2d & ) {} 
  double at() { return pos_; }
  void recompute();
  virtual void select( double x, double y, int b );
  virtual void move( double x, double y, int b );
  virtual void release( double x, double y, int b );

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;
  
  virtual void io(Piostream&);    
  
};

class YAxisObj :  public HairObj {
protected:
  int num_tics_;

public:
  
  YAxisObj( const string &name="yaxes" )   
    : HairObj(name), num_tics_(5) { pos_ = .5; }
  virtual ~YAxisObj();

  virtual void get_bounds( BBox2d & ) {} 
  double at() { return pos_; }
  void recompute();
  virtual void select( double x, double y, int b );
  virtual void move( double x, double y, int b );
  virtual void release( double x, double y, int b );

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;
  
  virtual void io(Piostream&);    
  
};

void Pio(Piostream&, XAxisObj*&);
void Pio(Piostream&, YAxisObj*&);

} // namespace SCIRun

#endif // SCI_AxesObj_h



