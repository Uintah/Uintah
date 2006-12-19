/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  NrrdTextureObj.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/NrrdTextureObj.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>


namespace SCIRun {



NrrdTextureObj::NrrdTextureObj(NrrdDataHandle nrrd_handle) :
  nrrd_handle_(nrrd_handle),
  width_(-1),
  height_(-1),
  dirty_(true),
  texture_id_(0)
{
  if (!nrrd_handle_.get_rep() ||
      !nrrd_handle_->nrrd_ ||
       nrrd_handle_->nrrd_->dim != 3) 
    throw "NrrdTextureObj::NrrdTextureObj(nrrd) nrrd not valid";

  width_  = nrrd_handle_->nrrd_->axis[1].size;
  height_ = nrrd_handle_->nrrd_->axis[2].size;
  color_[0] = color_[1] = color_[2] = color_[3] = 1.0;
  pad_to_power_of_2();
}


NrrdTextureObj::~NrrdTextureObj()
{
  if (glIsTexture(texture_id_)) {
    glDeleteTextures(1, (const GLuint*)&texture_id_);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  nrrd_handle_ = 0;
}


void
NrrdTextureObj::set_color(double r, double g, double b, double a)
{
  color_[0] = r;
  color_[1] = g;
  color_[2] = b;
  color_[3] = a;
}



void
NrrdTextureObj::pad_to_power_of_2()
{
  if (!nrrd_handle_.get_rep() || !nrrd_handle_->nrrd_) return;
  NrrdDataHandle nout_handle = scinew NrrdData();
  ptrdiff_t minp[3] = { 0, 0, 0 };
  ptrdiff_t maxp[3] = { 0, 
			Pow2(nrrd_handle_->nrrd_->axis[1].size)-1, 
			Pow2(nrrd_handle_->nrrd_->axis[2].size)-1 };

  if (nrrdPad_nva(nout_handle->nrrd_, nrrd_handle_->nrrd_,
		  minp, maxp, nrrdBoundaryBleed, 0)) {
    char *err = biffGetDone(NRRD);
    string error = string("Trouble resampling: ") + err;
    free (err);
    throw error;
  }
  nrrd_handle_ = nout_handle;
}


bool
NrrdTextureObj::bind()
{
  if (!nrrd_handle_.get_rep() || !nrrd_handle_->nrrd_) return false;

  const bool bound = glIsTexture(texture_id_);

  if (!bound)
    glGenTextures(1, (GLuint *)&texture_id_);

  glBindTexture(GL_TEXTURE_2D, texture_id_);
  CHECK_OPENGL_ERROR();
  if (bound && !dirty_) return true;
  dirty_ = false;
  Nrrd nrrd = *nrrd_handle_->nrrd_;
  int prim = 1;
  GLenum pixtype;
  if (nrrd.axis[0].size == 1) 
    pixtype = GL_ALPHA;
  
  else if (nrrd.axis[0].size == 2) 
    pixtype = GL_LUMINANCE_ALPHA;
  else if (nrrd.axis[0].size == 3) 
    pixtype = GL_RGB;
  else if (nrrd.axis[0].size == 4) 
    pixtype = GL_RGBA;
  else {
    prim = 0;
    pixtype = GL_ALPHA;
  }
  GLenum type = 0;
  switch (nrrd.type) {
  case nrrdTypeChar: type = GL_BYTE; break;
  case nrrdTypeUChar: type = GL_UNSIGNED_BYTE; break;
  case nrrdTypeShort: type = GL_SHORT; break;
  case nrrdTypeUShort: type = GL_UNSIGNED_SHORT; break;	
  case nrrdTypeInt: type = GL_INT; break;
  case nrrdTypeUInt: type = GL_UNSIGNED_INT; break;
  case nrrdTypeFloat: type = GL_FLOAT; break;
  default: throw "Cant bind nrrd"; break;
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  CHECK_OPENGL_ERROR();

  glTexImage2D(GL_TEXTURE_2D, 0, pixtype,
	       nrrd.axis[prim].size, nrrd.axis[prim+1].size, 
	       0, pixtype, type, nrrd.data);
  CHECK_OPENGL_ERROR();
  return true;
}

  
void
NrrdTextureObj::draw_quad(double x, double y, double w, double h) 
{
  if (bind()) {
    glEnable(GL_TEXTURE_2D);
    CHECK_OPENGL_ERROR();
  } else {
    glDisable(GL_TEXTURE_2D);
    CHECK_OPENGL_ERROR();
    return;
  }

  glColor4fv(color_);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  CHECK_OPENGL_ERROR();

  double tx = double(width_ )/nrrd_handle_->nrrd_->axis[1].size;
  double ty = double(height_)/nrrd_handle_->nrrd_->axis[2].size;
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0);
  glVertex3d(x, y, 0.0);

  glTexCoord2d(tx, 0.0);
  glVertex3d(x+w, y, 0.0);

  glTexCoord2d(tx, ty);
  glVertex3d(x+w, y+h, 0.0);

  glTexCoord2d(0.0, ty);
  glVertex3d(x, y+h, 0.0);

  glEnd();

  glDisable(GL_TEXTURE_2D);
  CHECK_OPENGL_ERROR();
}

}
