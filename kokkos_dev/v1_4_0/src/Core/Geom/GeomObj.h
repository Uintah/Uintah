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
#include <Core/Containers/Handle.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/PropertyManager.h>
#include <sci_config.h>

#include <iosfwd>

namespace SCIRun {

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
  GeomObj(int id = 0x1234567);
  GeomObj(IntVector id);
  GeomObj(int id_int, IntVector id);
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
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*)=0;
  // we want to return false if value is the default value
  virtual bool getId( int& ) { return false; }
  virtual bool getId( IntVector& ){ return false; }

  PropertyManager& properties() { return _properties; }
protected:

  int id;
  IntVector _id;
  PropertyManager _properties;
};

void Pio(Piostream&, GeomObj*&);

} // End namespace SCIRun

#endif // ifndef SCI_Geom_GeomObj_h
