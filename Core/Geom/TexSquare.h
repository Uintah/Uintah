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

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE TexSquare : public GeomObj {
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

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun

  
#endif
