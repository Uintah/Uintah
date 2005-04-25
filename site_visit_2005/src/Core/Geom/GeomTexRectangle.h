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

#ifndef SCI_GEOMTEXRECTANGLE_H
#define SCI_GEOMTEXRECTANGLE_H 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class GeomTexRectangle : public GeomObj {
  float tex_coords_[8];
  float pos_coords_[12];
  float normal_[3];
  unsigned char *texture_;
  unsigned int sId_;
  unsigned int fId_;
  int numcolors_;
  int width_;
  int height_;
  unsigned int texname_;
  double alpha_cutoff_;
  bool interp_;
  bool trans_;
  bool use_normal_;
  FragmentProgramARB *shader_;
  FragmentProgramARB *fog_shader_;
  void	bind_texture();
public:
  GeomTexRectangle();
  GeomTexRectangle(const GeomTexRectangle&);
  virtual ~GeomTexRectangle();
  void set_coords(float *tex, float *pos);
  void set_normal(float *norm);
  void set_texture( unsigned char *tex, int num, int w, int h);
  void set_texname(unsigned int texname);
  void set_transparency(bool b){trans_ = b;}
  void set_alpha_cutoff(double alpha);
  void interpolate(bool b){ interp_ = b; }
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double );
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

  
#endif
