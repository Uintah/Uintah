/*
 *  ColorMapTex.cc: Texture-mapped square
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_COLORMAPTEX_H
#define SCI_COLORMAPTEX_H 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE ColorMapTex : public GeomObj {
  Point a, b, c, d;
  unsigned char *texture;
  int numcolors, width;
  
public:
  ColorMapTex(const Point &p1,const Point &p2,const Point &p3,const Point &p4 );
  ColorMapTex(const ColorMapTex&);
  virtual ~ColorMapTex();

  void set_texture( unsigned char *tex, int w = 256){
    texture = tex; width = w; }
  
  virtual GeomObj* clone();
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
  
#endif
