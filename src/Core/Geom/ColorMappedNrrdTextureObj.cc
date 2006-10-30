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
 *  ColorMappedNrrdTextureObj.cc
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
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>

#include <teem/air.h>
#include <limits.h>

#include <Core/Thread/MutexPool.h>

namespace SCIRun {


ColorMappedNrrdTextureObj::ColorMappedNrrdTextureObj(NrrdDataHandle &nrrdh,
                                                     ColorMapHandle &cmap) :
  lock("ColorMappedNrrdTextureObj"),
  ref_cnt(0),
  colormap_(cmap),
  nrrd_handle_(nrrdh),
  dirty_(false),
  texture_id_dirty_(4,true),
  texture_id_(4,0),
  xdiv_(2),
  ydiv_(2),
  clut_min_(0.0), 
  clut_max_(1.0),
  data_(0),
  label_(0),
  min_(0.,0.,0.),
  xdir_(1.,0.,0.),
  ydir_(0.,1.,0.)  
{
  ASSERT(nrrdh.get_rep());
  ASSERT(nrrdh->nrrd_);
  ASSERT(nrrdh->nrrd_->axis[0].size == 1);
  size_t dim = 4*nrrdh->nrrd_->axis[1].size * nrrdh->nrrd_->axis[2].size;
  data_ = new float[dim];
  ASSERT(data_);
}




ColorMappedNrrdTextureObj::~ColorMappedNrrdTextureObj()
{
  delete[] data_;
  for (unsigned int t = 0; t < texture_id_.size(); ++t) 
    if (glIsTexture(texture_id_[t]))
        glDeleteTextures(1,(const GLuint *)&texture_id_[t]);
}

void
ColorMappedNrrdTextureObj::set_clut_minmax(float min, float max)
{
  clut_min_ = min;
  clut_max_ = max;
  dirty_ = true;
}

void
ColorMappedNrrdTextureObj::set_colormap(ColorMapHandle &cmap)
{
  colormap_ = cmap;
  dirty_ = true;
}


void
ColorMappedNrrdTextureObj::set_label(unsigned int label)
{
  label_ = label;
  dirty_ = true;
}


template <class T>
void
ColorMappedNrrdTextureObj::apply_colormap_to_raw_data(float *dst, 
                                                      T *src, 
                                                      int row_width,
                                                      int region_start,
                                                      int region_width,
                                                      int region_height,
                                                      const float *rgba, 
                                                      int ncolors,
                                                      float scale, 
                                                      float bias)
{
  const int sizeof4floats = 4*sizeof(float);
  const int maxcolor = ncolors-1;
  for (int row = 0; row < region_height; ++row) {
    int start_pos = region_start + row*row_width;
    int end_pos = start_pos + region_width;
    for (int pos = start_pos; pos < end_pos; ++pos) {
      float val = float(src[pos])*scale+bias;
      int color = Round(Clamp(val, 0.0f, 1.0f)*maxcolor);
      memcpy(dst + pos * 4, rgba + color * 4, sizeof4floats);
    }
  }
}




template <class T>
void
ColorMappedNrrdTextureObj::apply_colormap_to_label_bit(float *dst, 
                                                       T *src, 
                                                       int row_width,
                                                       int region_start,
                                                       int region_width,
                                                       int region_height,
                                                       const float *rgba,
                                                       unsigned char bit)
{
  const int sizeof4floats = 4*sizeof(float);
  const unsigned int mask = 1 << bit;
  ++bit;
  for (int row = 0; row < region_height; ++row) {
    int start_pos = region_start + row*row_width;
    int end_pos = start_pos + region_width;
    for (int pos = start_pos; pos < end_pos; ++pos) {
      if (src[pos] & mask) {
        memcpy(dst + pos * 4, rgba + bit * 4, sizeof4floats);
      } else {
        memcpy(dst + pos * 4, rgba, sizeof4floats);
      }              
    }
  }
}



void
ColorMappedNrrdTextureObj::apply_colormap(int x1, int y1, int x2, int y2,
                                          int border)
{
  if (!colormap_.get_rep()) {
    return;
  }

  Nrrd *nrrd = nrrd_handle_->nrrd_;

  if (x1 > x2) SWAP(x1,x2);
  if (y1 > y2) SWAP(y1,y2);
  x1 = Clamp(x1-border,   0, nrrd->axis[1].size);
  x2 = Clamp(x2+border+1, 0, nrrd->axis[1].size);
  y1 = Clamp(y1-border,   0, nrrd->axis[2].size);
  y2 = Clamp(y2+border+1, 0, nrrd->axis[2].size);

  double clut_min = clut_min_;

  int ncolors = colormap_->resolution();
  const float *rgba = colormap_->get_rgba();

  const float range = (clut_max_ - clut_min);
  const float scale = (range > 0.00001) ? (1.0 / range) : 1.0;
  const float bias =  (range > 0.00001) ? -clut_min*scale : 0.0;

  const int row_width = nrrd->axis[1].size;
  const int region_start = row_width * y1 + x1;
  const int region_wid = x2 - x1;
  const int region_hei = y2 - y1;

  switch (nrrd->type) {
  case nrrdTypeChar: {
    apply_colormap_to_raw_data(data_, (char *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeUChar: {
    apply_colormap_to_raw_data(data_, 
                               (unsigned char *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeShort: {
    apply_colormap_to_raw_data(data_, (short *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeUShort: {
    apply_colormap_to_raw_data(data_, (unsigned short *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeInt: {
    apply_colormap_to_raw_data(data_, (int *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeUInt: {
    if (label_) {
      unsigned char bit = 0;
      while (!(label_ & (1 << bit))) ++bit;
      apply_colormap_to_label_bit(data_, 
                                  (unsigned int *)nrrd->data,
                                  row_width, region_start, 
                                  region_wid, region_hei,
                                  rgba, bit);
    } else {
      apply_colormap_to_raw_data(data_, (unsigned int *)nrrd->data,
                                 row_width, region_start, region_wid, 
                                 region_hei, rgba, ncolors, scale, bias);
    }
    
    
  } break;
  case nrrdTypeLLong: {
    apply_colormap_to_raw_data(data_, 
                               (signed long long *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeULLong: {
    apply_colormap_to_raw_data(data_, 
                               (unsigned long long *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeFloat: {
    apply_colormap_to_raw_data(data_, (float *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  case nrrdTypeDouble: {
    apply_colormap_to_raw_data(data_, (double *)nrrd->data,
                               row_width, region_start, region_wid, region_hei,
                               rgba, ncolors, scale, bias);
  } break;
  default: throw "Unsupported data type: " + 
             to_string(nrrd->type);
  }
  
  
  Point min(x1, y1, 0), max(x2, y2, 1);
  BBox bbox(min,max);
  for (int i = 0; i < 4; ++i) {
    Point min2(xdiv_[i%2].first,  ydiv_[i/2].first,  0);
    Point max2(xdiv_[i%2].second, ydiv_[i/2].second, 1);
    BBox bbox2(min2, max2);
    if (bbox.overlaps(bbox2)) {
      texture_id_dirty_[i] = true;
    }
  }

  dirty_ = false;
}

  
bool
ColorMappedNrrdTextureObj::bind(int x, int y)
{

  const int pos = y*2+x;
  const bool bound = glIsTexture(texture_id_[pos]);


  if (!bound) {
    glGenTextures(1, (GLuint *)&texture_id_[pos]);
    CHECK_OPENGL_ERROR();  
  }

  glBindTexture(GL_TEXTURE_2D, texture_id_[pos]);
  CHECK_OPENGL_ERROR();      

  if (bound && !dirty_ && !texture_id_dirty_[pos]) {
    return true;
  }



  Nrrd *nrrd = nrrd_handle_->nrrd_;

  if (dirty_) {
    apply_colormap(0, 0, nrrd->axis[1].size, nrrd->axis[2].size);
  }


  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, nrrd->axis[1].size);  
  glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, nrrd->axis[2].size);  
  CHECK_OPENGL_ERROR();  

  if (x == 0) {
    xdiv_[x].first = 0;
    xdiv_[x].second = LargestPowerOf2(nrrd->axis[1].size);
  } else if (x == 1) {
    int wid = Pow2(nrrd->axis[1].size - xdiv_[0].second);
    xdiv_[1].second = nrrd->axis[1].size;
    xdiv_[1].first = xdiv_[1].second - wid;
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, xdiv_[1].first);
  }
  CHECK_OPENGL_ERROR();  


  if (y == 0) {
    ydiv_[y].first = 0;
    ydiv_[y].second = LargestPowerOf2(nrrd->axis[2].size);
  } else if (y == 1) {
    int wid = Pow2(nrrd->axis[2].size - ydiv_[0].second);
    ydiv_[1].second = nrrd->axis[2].size;
    ydiv_[1].first = ydiv_[1].second - wid;
    glPixelStorei(GL_UNPACK_SKIP_ROWS, ydiv_[1].first);
  }
  CHECK_OPENGL_ERROR();  

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  CHECK_OPENGL_ERROR();  
  
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
               xdiv_[x].second - xdiv_[x].first, 
               ydiv_[y].second - ydiv_[y].first,
               0,GL_RGBA, GL_FLOAT, data_);
  CHECK_OPENGL_ERROR();  

  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0); 
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0); 

  CHECK_OPENGL_ERROR();  
  texture_id_dirty_[pos] = false;

  return true;
}


void
ColorMappedNrrdTextureObj::set_coords(const Point &min, 
                                      const Vector &xdir, 
                                      const Vector &ydir)
{
  min_ = min;
  xdir_ = xdir;
  ydir_ = ydir;
}


void
ColorMappedNrrdTextureObj::get_bounds(BBox &bbox) {
  bbox.extend(min_);
  bbox.extend(min_+xdir_);
  bbox.extend(min_+ydir_);
}



  
void
ColorMappedNrrdTextureObj::draw_quad() 
{
  glEnable(GL_TEXTURE_2D);
  glColor4d(opacity_, opacity_, opacity_, opacity_);
  unsigned int yoff = 0;
  for (int y = 0; y < 2; ++y) {
    unsigned int xoff = 0;
    for (int x = 0; x < 2; ++x) {
      if (!bind(x,y)) continue;

      float x1 = xoff / float(nrrd_handle_->nrrd_->axis[1].size);
      float y1 = yoff / float(nrrd_handle_->nrrd_->axis[2].size);
      float x2 = xdiv_[x].second / float(nrrd_handle_->nrrd_->axis[1].size);
      float y2 = ydiv_[y].second / float(nrrd_handle_->nrrd_->axis[2].size);

      float tx = ((xoff - xdiv_[x].first) / 
                  float(xdiv_[x].second - xdiv_[x].first));
      float ty = ((yoff - ydiv_[y].first) / 
                  float(ydiv_[y].second - ydiv_[y].first));
      xoff = xdiv_[x].second;

      glBegin(GL_QUADS);
      
      glTexCoord2d(tx, ty); 
      Point p = min_ + x1*xdir_ + y1*ydir_;
      glVertex3f(p.x(), p.y(), p.z());
      
      glTexCoord2d(1.0, ty);
      p = min_ + x2*xdir_ + y1*ydir_;
      glVertex3f(p.x(), p.y(), p.z());
      
      glTexCoord2d(1.0, 1.0);
      p = min_ + x2*xdir_ + y2*ydir_;
      glVertex3f(p.x(), p.y(), p.z());
      
      glTexCoord2d(tx, 1.0);
      p = min_ + x1*xdir_ + y2*ydir_;
      glVertex3f(p.x(), p.y(), p.z());
      glEnd();
      


      CHECK_OPENGL_ERROR(); 
    }
    yoff = ydiv_[y].second;
  }
  glDisable(GL_TEXTURE_2D);
}


}


