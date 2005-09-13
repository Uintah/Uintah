/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  HistogramTex.cc: Texture-mapped square
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_HISTOGRAMTEX_H
#define SCI_HISTOGRAMTEX_H 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class HistogramTex : public GeomObj {
  Point a, b, c, d;
  const int *buckets;
  int nbuckets, min, max;
  const float *texture;
  int numcolors, width;
  
public:
  HistogramTex(const Point &p1,const Point &p2,
	       const Point &p3,const Point &p4);
  HistogramTex(const HistogramTex&);
  virtual ~HistogramTex();

  void set_texture( const float *tex, int w = 256){
    // This is unsafe. Colormap deletion before draw could result in a
    // crash.  However the texture is copied to OpenGL and then not used,
    // so problems would be rare.
    texture = tex; width = w; }
  void set_buckets( const int *b, int nb, int mn, int mx) {
    buckets = b; nbuckets = nb; min = mn; max = mx; }
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
