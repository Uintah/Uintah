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
 *  TexSquare.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (c) 199? SCI Group
 */

#include <Core/Geom/TexSquare.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>

using std::ostream;

namespace SCIRun {

Persistent *make_TexSquare() {
  return scinew TexSquare();
}

PersistentTypeID TexSquare::type_id("TexSquare", "GeomObj", make_TexSquare);

TexSquare::TexSquare()
  : GeomObj(),
    normal_(1.0, 0.0, 0.0),
    texture(0),
    numcolors(0),
    width_(2),
    height_(2),
    texname_(0),
    alpha_cutoff_(0.0)
{
}

TexSquare::TexSquare( const TexSquare &copy ) : GeomObj(copy) {
}

TexSquare::~TexSquare()
{
}

void
TexSquare::set_coords(float *tex, float *coords)
{
  memcpy(tex_coords_, tex, 8*sizeof(float));
  memcpy(pos_coords_, coords, 12*sizeof(float));
}


void 
TexSquare::set_texture( unsigned char *tex, int num, int w, int h) {
  width_ = w;
  height_ = h;
  numcolors = num;
  const int count = numcolors*width_*height_;
  texture = new unsigned char[count];
  memcpy(texture, tex, count);
}

void
TexSquare::set_texname(unsigned int texname) {
  texname_ = texname;
}

void
TexSquare::set_normal(Vector &normal) {
  normal_ = normal;
}


void
TexSquare::set_alpha_cutoff(double alpha) {
  alpha_cutoff_ = alpha;
}


GeomObj* TexSquare::clone() {
  return scinew TexSquare( *this );
}

void TexSquare::get_bounds( BBox& bb ) {
  for (int i = 0; i < 4; ++i)
    bb.extend(Point(pos_coords_[i*3+0],pos_coords_[i*3+1],pos_coords_[i*3+2]));
}

#define TEXSQUARE_VERSION 1

void TexSquare::io(Piostream& stream) {

  stream.begin_class("TexSquare", TEXSQUARE_VERSION);
  GeomObj::io(stream);
  //  Pio(stream, a);
  //Pio(stream, b);
  //Pio(stream, c);
  //Pio(stream, d);
  stream.end_class();
}

} // End namespace SCIRun

