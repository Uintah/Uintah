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

namespace SCIRun {  

FreeTypeTextTexture::FreeTypeTextTexture(const string &text, 
					 FreeTypeFace *face) :
  texture_(0),
  text_(text),
  face_(face),
  dirty_(true)
{  
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
}

void
FreeTypeTextTexture::render_text_to_texture()
{
  if (!face_) 
    throw "FreeTypeTextTexture::render_text_to_texture(), face_ == 0";
  BBox bbox;
  bbox.reset();
  FreeTypeText fttext(text_, face_);
  fttext.get_bounds(bbox);
  if (bbox.min().y() < 0) {
    fttext.set_position(Point(0, -bbox.min().y(), 0));
    bbox.reset();
    fttext.get_bounds(bbox);
  }

  // +1 is for some error in ft bbox
  const unsigned int wid = int(ceil(bbox.max().x()))+1;
  const unsigned int hei = int(ceil(bbox.max().y()))+1;

  // 3 dimensions = alpha x X x Y
  
  NrrdDataHandle nrrd = scinew NrrdData();
  nrrdAlloc(nrrd->nrrd, nrrdTypeUChar, 3, 1, wid, hei);
  memset(nrrd->nrrd->data, 0, wid*hei);
  fttext.render(wid, hei, (unsigned char *)nrrd->nrrd->data);

  if (texture_) {
    delete texture_;
    texture_ = 0;
  }

  Nrrd *nout = nrrdNew();
  nrrdFlip(nout, nrrd->nrrd, 2); // Is there a flip in place?

  NrrdDataHandle ndout(scinew NrrdData(nout));
  texture_ = scinew NrrdTextureObj(ndout, false, false);
  texture_->set_color(0.0, 0.0, 0.0, 1.0);
  //  texture_->set_alpha(0.76);
  dirty_ = false;
}
  

void
FreeTypeTextTexture::draw(double x, double y, double sx, double sy,
			  FreeTypeTextTexture::anchor_e anchor)
{
  if (dirty_)
    render_text_to_texture();

  if (!texture_)
    throw "FreeTypeTextTexture::draw() texture_ == 0";

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
  x = ceil(x);
  y = ceil(y);
  texture_->draw_quad(x, y, w, h);
}


}
