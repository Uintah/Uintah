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


NrrdTextureObj::NrrdTextureObj(const string &filename,
                               bool ignore_error,
                               bool repeatx,
                               bool repeaty) :
  nrrd_(scinew NrrdData()),
  filename_(filename),
  fromfile_(1),
  width_(1),
  height_(1),
  alpha_(1.0),
  dirty_(true),
  texture_id_(0),
  repeat_x_(repeatx),
  repeat_y_(repeaty)
{
  nrrd_->nrrd = nrrdNew();
  if (nrrdLoad(nrrd_->nrrd, filename.c_str(), 0))
  {
    string errstr = 
      "NrrdTextureObj::NrrdTextureObj(string filename) read error on filename: " +
      filename + biffGetDone(NRRD);
    char *err = biffGetDone(NRRD);
    free(err);
    nrrd_ = 0;
    if (!ignore_error) 
      throw errstr;
  }

  if (nrrd_.get_rep() && nrrd_->nrrd && nrrd_->nrrd->dim == 3) {
    width_ = nrrd_->nrrd->axis[1].size;
    height_ = nrrd_->nrrd->axis[2].size;
    rescale_to_power_of_2();
  } else if (!ignore_error) {
    throw "NrrdTextureObj::NrrdTextureObj(nrrd) nrrd not valid";
  }
  
  //  double d = drand48();
  //  if (d < 0.33) 
  //    color_[0] = color_[1] = color_[2] = color_[3] = 0.0;
  //  else if (d < 0.66) 
  //    color_[0] = color_[1] = color_[2] = color_[3] = 0.5;
  //  else
  color_[0] = color_[1] = color_[2] = color_[3] = 1.0;

}


NrrdTextureObj::NrrdTextureObj(NrrdDataHandle nrrd,
                               bool repeatx,
                               bool repeaty) :
  nrrd_(nrrd),
  filename_(),
  fromfile_(0),
  width_(-1),
  height_(-1),
  alpha_(1.0),
  dirty_(true),
  texture_id_(0),
  repeat_x_(repeatx),
  repeat_y_(repeaty)
{
  if (!nrrd_.get_rep() || !nrrd_->nrrd || nrrd_->nrrd->dim != 3) 
    throw "NrrdTextureObj::NrrdTextureObj(nrrd) nrrd not valid";

  width_ = nrrd_->nrrd->axis[1].size;
  height_ = nrrd_->nrrd->axis[2].size;
  color_[0] = color_[1] = color_[2] = color_[3] = 0.0;
  rescale_to_power_of_2();
}


NrrdTextureObj::~NrrdTextureObj()
{
}

void
NrrdTextureObj::set_alpha(double a)
{
  alpha_ = a;
  dirty_ = true;
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
NrrdTextureObj::rescale_to_power_of_2()
{
  if (!nrrd_.get_rep() || !nrrd_->nrrd) return;
  Nrrd *nin = nrrd_->nrrd;
  
  NrrdResampleInfo *info = nrrdResampleInfoNew();
  NrrdKernel *kern = nrrdKernelBox;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;

  for (int a = 0; a < nin->dim; a++) {
    info->samples[a] = nin->axis[a].size;
    info->kernel[a] = 0;
    //    if ((a == 1 && repeat_x_) || (a == 2 && repeat_y_)) {
    if (a) {
      
      info->samples[a] = Pow2(nin->axis[a].size);
      info->kernel[a] = kern;

      if (!(airExists(nin->axis[a].min) && airExists(nin->axis[a].max))) {
	nrrdAxisInfoMinMaxSet(nin, a, nin->axis[a].center ? 
			      nin->axis[a].center : nrrdDefCenter);
      }
      info->min[a] = nin->axis[a].min;
      if ((a == 1 && repeat_x_) || (a == 2 && repeat_y_))
	info->max[a] = nin->axis[a].max;
      else 
	info->max[a] = nin->axis[a].min + 
	  Pow2(int(ceil(nin->axis[a].max-nin->axis[a].min)));

    }      
    memcpy(info->parm[a], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  }    
  info->boundary = nrrdBoundaryBleed;
  info->type = nin->type;
  info->renormalize = AIR_TRUE;

  NrrdData *nrrd = scinew NrrdData;
  if (nrrdSpatialResample(nrrd->nrrd=nrrdNew(), nin, info)) {
    char *err = biffGetDone(NRRD);
    throw string("Trouble resampling: ") + err;
    free(err);
  }
  nrrdResampleInfoNix(info); 

  //  nrrdSave(("/tmp/out"+to_string(info->samples[1])+"x"+to_string(info->samples[2])+".png").c_str(),nrrd->nrrd,0);
  nrrd_ = nrrd;
}


bool
NrrdTextureObj::bind()
{
  if (!nrrd_.get_rep() || !nrrd_->nrrd) return false;
  const bool bound = glIsTexture(texture_id_);
  if (!bound) {
    glGenTextures(1, &texture_id_);
  }

  glBindTexture(GL_TEXTURE_2D, texture_id_);

  if (bound && !dirty_) return true;
  dirty_ = false;
  Nrrd nrrd = *nrrd_->nrrd;
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
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  //  const GLint filter_mode= /*GL_LINEAR */GL_NEAREST;
  //  const GLint filter_mode= GL_LINEAR;
  const GLint filter_mode= GL_NEAREST;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode);
  glTexParameteri(GL_TEXTURE_2D, GL_MAP_COLOR, 0);
  glPixelTransferf(GL_ALPHA_SCALE, alpha_);
  glTexImage2D(GL_TEXTURE_2D, 0, pixtype,
	       nrrd.axis[prim].size, nrrd.axis[prim+1].size, 
	       0, pixtype, type, nrrd.data);
  return true;
}

  
void
NrrdTextureObj::draw_quad(double x, double y, double w, double h) 
{
  if (bind()) {
    glEnable(GL_TEXTURE_2D);
    if (nrrd_->nrrd->axis[0].size == 1) {
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
      glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, color_);
    }  else {
      GLfloat black[] = { 0.0, 0.0, 0.0, 0.0 };
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);;
      glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, black);
    }

  } else {
    glDisable(GL_TEXTURE_2D);
  }
    

  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  glColor4fv(color_);
  //  glColor4d(1.0, 1.0, 1.0, 1.0);//color_);

  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0);
  glVertex3d(x, y, 0.0);
  
  double tx = (repeat_x_?w/width_:double(width_)/nrrd_->nrrd->axis[1].size);
  double ty = (repeat_y_?h/height_:double(height_)/nrrd_->nrrd->axis[2].size);
  //tx = 1.0;
  //ty = 1.0;
  glTexCoord2d(tx, 0.0);
  glVertex3d(x+w, y, 0.0);

  glTexCoord2d(tx, ty);
  glVertex3d(x+w, y+h, 0.0);

  glTexCoord2d(0.0, ty);
  glVertex3d(x, y+h, 0.0);

  glEnd();
  glDisable(GL_TEXTURE_2D);
}

}
