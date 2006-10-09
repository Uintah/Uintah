//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : GeomColorMappedNrrdTextureObj.cc
//    Author : McKay Davis
//    Date   : Tue Oct  3 14:47:56 2006

#include <Core/Geom/GeomColorMappedNrrdTextureObj.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geom/DrawInfoOpenGL.h>

#include <sci_gl.h>

using std::ostream;

namespace SCIRun {

Persistent *make_GeomColorMappedNrrdTextureObj() {
  return scinew GeomColorMappedNrrdTextureObj();
}

PersistentTypeID GeomColorMappedNrrdTextureObj::type_id("GeomColorMappedNrrdTextureObj", "GeomObj", make_GeomColorMappedNrrdTextureObj);

GeomColorMappedNrrdTextureObj::GeomColorMappedNrrdTextureObj
(ColorMappedNrrdTextureObjHandle &cmnto)
  : GeomObj(),
    cmnto_(0),
    alpha_cutoff_(0.0)
{  
  cmnto_ = cmnto;
}

GeomColorMappedNrrdTextureObj::GeomColorMappedNrrdTextureObj( const GeomColorMappedNrrdTextureObj &copy ) : GeomObj(copy),
                                                                                                            cmnto_(copy.cmnto_)
{  
}

GeomColorMappedNrrdTextureObj::~GeomColorMappedNrrdTextureObj()
{
  cmnto_ = 0;
}


void
GeomColorMappedNrrdTextureObj::set_alpha_cutoff(double alpha) {
  alpha_cutoff_ = alpha;
}


GeomObj* GeomColorMappedNrrdTextureObj::clone() {
  return scinew GeomColorMappedNrrdTextureObj( *this );
}

void
GeomColorMappedNrrdTextureObj::draw(DrawInfoOpenGL* di, 
                                    Material* matl, double)
{
  double old_ambient = di->ambient_scale_;
  di->ambient_scale_ = 150;
  if (!pre_draw(di, matl, 1)) {
    di->ambient_scale_ = old_ambient;
    return;
  }
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glEnable(GL_ALPHA_TEST);  
  glAlphaFunc(GL_GREATER, 0.1);
  cmnto_->draw_quad();
  glDisable(GL_ALPHA_TEST);  
  di->ambient_scale_ = old_ambient;
  post_draw(di);
}



void GeomColorMappedNrrdTextureObj::get_bounds( BBox& bb ) {
  cmnto_->get_bounds(bb);
}

void GeomColorMappedNrrdTextureObj::io(Piostream& stream) {
  stream.begin_class("GeomColorMappedNrrdTextureObj", 1);
  GeomObj::io(stream);
  stream.end_class();
}

} // End namespace SCIRun

