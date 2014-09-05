
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

#ifndef SCI_Geom_Sphere_h
#define SCI_Geom_Sphere_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/IntVector.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::IntVector;

class SCICORESHARE GeomSphere : public GeomObj {
public:
  Point cen;
  double rad;
  int nu;
  int nv;
  
  void adjust();
  void move(const Point&, double, int nu=20, int nv=10);
  void move(const Point& _cen);
    
  GeomSphere(int nu, int nv, IntVector id);
  GeomSphere(int nu, int nv, int id_int, IntVector id);
  GeomSphere(int nu=20, int nv=10, int id = 0x1234567);
  GeomSphere(const Point&, double, int nu=20, int nv=10, int id = 0x1234567);
  GeomSphere(const Point&, double, int nu, int nv, int id_int, IntVector id);
  GeomSphere(const Point&, double, int nu, int nv, IntVector id);
  GeomSphere(const GeomSphere&);
  virtual ~GeomSphere();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
  virtual bool getId( int& id );
  virtual bool getId( IntVector& id);
  
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.5.2.2  2000/10/26 17:18:38  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.8  2000/09/11 22:14:46  bigler
// Added constructors that take an int and IntVector to allow unique
// identification in 4 dimensions.
//
// Revision 1.7  2000/08/11 15:38:35  bigler
// Added another constructor that took an IntVector index.
//
// Revision 1.6  2000/08/09 18:21:15  kuzimmer
// Added IntVector indexing to GeomObj & GeomSphere
//
// Revision 1.5  2000/01/03 20:12:37  kuzimmer
//  Forgot to check in these files for picking spheres
//
// Revision 1.4  1999/10/07 02:07:45  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:25  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:13  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:07  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:02  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//

#endif /* SCI_Geom_Sphere_h */
