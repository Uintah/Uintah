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
 * Sphere.h: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_View_Window_Clip_Frame_h
#define SCI_View_Window_Clip_Frame_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomCylinder.h>

#include <Core/Geom/share.h>

namespace SCIRun {


class SCISHARE ViewWindowClipFrame : public GeomObj {
public:

  ViewWindowClipFrame();
  ViewWindowClipFrame(const ViewWindowClipFrame& copy);
  virtual ~ViewWindowClipFrame();
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  static PersistentTypeID type_id;

  virtual void io(Piostream&);    

  void Set(Point c, Vector normal, double width, double height, double scale);
  void SetPosition(Point c, Vector normal);
  void SetSize(double width, double height);
  void SetScale( double scale );

private:
  Point center_;
  Vector normal_;
  double width_, height_, scale_;
  vector<Point> verts_;
  vector<GeomSphere*>  corners_;
  vector<GeomCylinder*> edges_;
  void adjust();
  void set_position(Point c, Vector n){ center_ = c; normal_ = n;}
  void set_size(double w, double h) { width_ = w;  height_ = h;}
  void set_scale(double s) { scale_ = s;}
  
};

} // End namespace SCIRun

#endif /* SCI_View_Window_Clip_Frame_h */
