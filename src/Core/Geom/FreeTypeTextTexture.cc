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
 *  FreeTypeTextTexture.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Core/Geom/FreeTypeTextTexture.h>
#include <Core/Geom/NrrdTextureObj.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/BBox.h>
#include <Core/Exceptions/InternalError.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

namespace SCIRun {  

FreeTypeTextTexture::FreeTypeTextTexture(const string &text, 
					 FreeTypeFace *face) :
  texture_(0),
  text_(text),
  face_(face),
  dirty_(true)
{
  color_[0] = color_[1] = color_[2] = color_[3] = 1.0;
}

FreeTypeTextTexture::~FreeTypeTextTexture() 
{
  if (texture_) {
    delete texture_;
    texture_ = 0;
  }
}

void
FreeTypeTextTexture::set(const string &text) 
{
  text_ = text;
  dirty_ = true;
}

int
FreeTypeTextTexture::width() {
  BBox bbox;
  bbox.reset();
  FreeTypeText fttext(text_, face_);
  fttext.get_bounds(bbox);
  fttext.set_position(Point(-bbox.min().x(), -bbox.min().y(), 0));
  bbox.reset();
  fttext.get_bounds(bbox);
  return Ceil(bbox.diagonal().x());
}

int
FreeTypeTextTexture::height() {
  BBox bbox;
  bbox.reset();
  FreeTypeText fttext(text_, face_);
  fttext.get_bounds(bbox);
  fttext.set_position(Point(-bbox.min().x(), -bbox.min().y(), 0));
  bbox.reset();
  fttext.get_bounds(bbox);
  return Ceil(bbox.diagonal().y());
}


void
FreeTypeTextTexture::set_color(double r, double g, double b, double a) 
{  
  color_[0] = r;
  color_[1] = g;
  color_[2] = b;
  color_[3] = a;
}




void
FreeTypeTextTexture::render_text_to_texture()
{
  if (!face_) 
    throw InternalError("FreeTypeTextTexture::render_text_to_texture(), face_ == 0", __FILE__, __LINE__);
  BBox bbox;
  bbox.reset();
  FreeTypeText fttext(text_, face_);
  fttext.get_bounds(bbox);
  fttext.set_position(Point(-bbox.min().x(), -bbox.min().y(), 0));
  bbox.reset();
  fttext.get_bounds(bbox);

  const int wid = Ceil(bbox.diagonal().x());
  const int hei = Ceil(bbox.diagonal().y());

  // 3 dimensions = alpha x X x Y
  size_t size[NRRD_DIM_MAX];
  size[0] = 1;
  size[1] = wid;
  size[2] = hei;
  NrrdDataHandle nrrd_handle = scinew NrrdData();
  nrrdAlloc_nva(nrrd_handle->nrrd_, nrrdTypeUChar, 3, size);
  memset(nrrd_handle->nrrd_->data, 0, wid*hei);
  fttext.render(wid, hei, (unsigned char *)nrrd_handle->nrrd_->data);

  if (texture_) {
    delete texture_;
    texture_ = 0;
  }

  texture_ = scinew NrrdTextureObj(nrrd_handle);
  dirty_ = false;
}
  

void
FreeTypeTextTexture::draw(double x, double y,
			  FreeTypeTextTexture::anchor_e anchor)
{
  if (dirty_)
    render_text_to_texture();

  if (!texture_)
    throw InternalError("FreeTypeTextTexture::draw() texture_ == 0", __FILE__, __LINE__);

  double w = texture_->width();
  double h = texture_->height();
  
  switch (anchor) {
  case FreeTypeTextTexture::s:  x -= w/2.0; break;
  case FreeTypeTextTexture::se: x -= w;     break;
  case FreeTypeTextTexture::e:  x -= w;     
				y -= h/2.0; break;
  case FreeTypeTextTexture::ne: x -= w;     
				y -= h;     break;
  case FreeTypeTextTexture::n:  x -= w/2.0; 
				y -= h;     break;
  case FreeTypeTextTexture::nw: y -= h;     break;
  case FreeTypeTextTexture::w:  y -= h/2.0; break;
  case FreeTypeTextTexture::c:  x -= w/2.0;
                                y -= h/2.0; break;
  default: // lowerleft do noting
  case FreeTypeTextTexture::sw: break;
  }
  CHECK_OPENGL_ERROR();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  CHECK_OPENGL_ERROR();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  GLint gl_viewport[4];
  glGetIntegerv(GL_VIEWPORT, gl_viewport);;
  CHECK_OPENGL_ERROR();

  x = Floor(x) / double(gl_viewport[2] - gl_viewport[0]);
  y = Floor(y) / double(gl_viewport[3] - gl_viewport[1]);
  w = w / double(gl_viewport[2] - gl_viewport[0]);
  h = h / double(gl_viewport[3] - gl_viewport[1]);
  texture_->set_color(color_[0], color_[1], color_[2], color_[3]);
  texture_->draw_quad(x, y, w, h);

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();
}


}
