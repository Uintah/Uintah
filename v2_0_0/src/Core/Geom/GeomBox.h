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


} // End namespace SCIRun


#endif /* SCI_Geom_Box_h */
