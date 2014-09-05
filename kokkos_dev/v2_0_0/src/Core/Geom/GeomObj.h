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
 *  GeomObj.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GeomObj_h
#define SCI_Geom_GeomObj_h 1

#include <Core/Containers/Array1.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Persistent/Persistent.h>
#include <sci_config.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

struct DrawInfoOpenGL;
class  Material;
class  GeomSave;
class  Hit;
class  BBox;
class  Vector;
class  Point;
class  IntVector;

class SCICORESHARE GeomObj : public Persistent {
public:
  int ref_cnt;
  Mutex &lock;

  GeomObj();
  GeomObj(const GeomObj&);
  virtual ~GeomObj();
  virtual GeomObj* clone() = 0;

  virtual void reset_bbox();
  virtual void get_bounds(BBox&) = 0;

  // For OpenGL
#ifdef SCI_OPENGL
  int pre_draw(DrawInfoOpenGL*, Material*, int lit);
  virtual void draw(DrawInfoOpenGL*, Material*, double time)=0;
  int post_draw(DrawInfoOpenGL*);
#endif
  virtual void get_triangles( Array1<float> &v );
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    

  // we want to return false if value is the default value
  virtual void setId(int id) { id_int_ = id; }
  virtual void setId(const IntVector &id) { id_intvector_ = id; }
  virtual void setId(long long id) { id_longlong_ = id; }

  virtual bool getId( int &id );
  virtual bool getId( IntVector &id );
  virtual bool getId( long long &id );

private:

  int       id_int_;
  IntVector id_intvector_;
  long long  id_longlong_;
};

void Pio(Piostream&, GeomObj*&);

typedef LockingHandle<GeomObj> GeomHandle;

} // End namespace SCIRun

#endif // ifndef SCI_Geom_GeomObj_h




