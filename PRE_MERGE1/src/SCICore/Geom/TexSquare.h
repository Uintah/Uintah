/*
 *  TexSquare.cc: Texture-mapped square
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_TEXSQUARE_H
#define SCI_TEXSQUARE_H 1

#include <Geom/GeomObj.h>
#include <Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

class TexSquare : public GeomObj {
  Point a, b, c, d;
  unsigned char *texture;
  int numcolors, width;
  
public:
  TexSquare(const Point &p1,const Point &p2,const Point &p3,const Point &p4 );
  TexSquare(const TexSquare&);
  virtual ~TexSquare();

  void set_texture( unsigned char *tex, int num, int w = 32 );
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>& dontfree);
  virtual void preprocess();
  virtual void intersect(const Ray& ray, Material*, Hit& hit);
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:53  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:14  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:14  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
  
#endif
