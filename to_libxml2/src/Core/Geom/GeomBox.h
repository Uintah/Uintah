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
 * GeomBox.h:  Box object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Geom_Box_h
#define SCI_Geom_Box_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class GeomBox : public GeomObj {
  Point min, max;
  int opacity[6];
public:

  GeomBox( const Point& p, const Point& q, int op );
  GeomBox(const GeomBox&);
  virtual ~GeomBox();

  int opaque(int i) { return opacity[i]; }
  void opaque( int i, int op ) { opacity[i] = op; }
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomSimpleBox : public GeomObj {
protected:
  Point min, max;

public:

  GeomSimpleBox( const Point& p, const Point& q);
  GeomSimpleBox(const GeomSimpleBox&);
  virtual ~GeomSimpleBox();

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomCBox : public GeomSimpleBox {
public:
  GeomCBox( const Point& p, const Point& q);
  GeomCBox(const GeomCBox&);
  virtual ~GeomCBox();

  virtual GeomObj* clone();
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomBoxes : public GeomObj {
private:
  vector<Point> centers_;
  vector<double> edges_;
  vector<unsigned char> colors_;
  vector<float> indices_;
  int nu_;
  int nv_;
  double global_edge_;
  
public:

  GeomBoxes(double edge = 1.0, int nu=8, int nv=8);
  GeomBoxes(const GeomBoxes &copy);
  virtual ~GeomBoxes();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
  void add(const Point &center);
  void add(const Point &center, const MaterialHandle &mat);
  void add(const Point &center, float index);

  // If edge is too small, the box is not added and false is returned.
  bool add_edge(const Point &cen, double edge);
  bool add_edge(const Point &cen, double edge, const MaterialHandle &mat);
  bool add_edge(const Point &cen, double edge, float index);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Box_h */
