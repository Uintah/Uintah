
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
#include <Core/Containers/String.h>
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
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun


#endif

