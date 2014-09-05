
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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:40  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:18  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:05  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:37  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:03  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:54  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif

