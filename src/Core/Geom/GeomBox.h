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

class SCICORESHARE GeomBox : public GeomObj {
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


class SCICORESHARE GeomSimpleBox : public GeomObj {
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


class SCICORESHARE GeomCBox : public GeomSimpleBox {
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


class SCICORESHARE GeomBoxes : public GeomObj {
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
