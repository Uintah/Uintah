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
 *  GeomBillboard.h: Billboard object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Geom_Billboard_h
#define SCI_Geom_Billboard_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomBillboard: public GeomObj {
  GeomObj* child;
  Point at;
  
  BBox bbox;
public:
  GeomBillboard(GeomObj*, const Point &);
  
  virtual ~GeomBillboard();
  
  virtual GeomObj* clone();
  //    virtual void reset_bbox();
  virtual void get_bounds(BBox&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;	
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};

} // End namespace SCIRun


#endif

