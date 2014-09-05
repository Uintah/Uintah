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

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE ColorMapTex : public GeomObj {
  Point a_, b_, c_, d_;
  unsigned char texture_[256*4];
  int numcolors_;
  
public:
  ColorMapTex(const Point &p1, const Point &p2, 
	      const Point &p3,const Point &p4 );
  ColorMapTex(const ColorMapTex &);
  virtual ~ColorMapTex();

  void set_texture(unsigned char *tex, int w = 256)
  {
    numcolors_ = Max(Min(w, 256), 0);
    memcpy(texture_, tex, numcolors_ * 4);
  }
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

  
#endif
