//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
//    File   : CM2Widget.cc
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:29 2004

#include <sci_gl.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Core/Datatypes/CM2Shader.h>
#include <Packages/Volume/Core/Datatypes/CM2Widget.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Util/Utils.h>
#include <Core/Math/MinMax.h>

#include <iostream>
#include <sstream>

using namespace std;

using namespace SCIRun;

namespace Volume {

CM2Widget::CM2Widget()
  : line_color_(0.75, 0.75, 0.75),
    line_alpha_(1.0),
    selected_color_(1.0, 0.0, 0.0),
    selected_alpha_(1.0),
    thin_line_width_(0.75),
    thick_line_width_(2.0),
    point_size_(7.0),
    color_(0.5, 0.5, 1.0),
    alpha_(0.7),
    selected_(0)
{}

CM2Widget::~CM2Widget()
{}

CM2Widget::CM2Widget(CM2Widget& copy)
  : line_color_(copy.line_color_),
    line_alpha_(copy.line_alpha_),
    selected_color_(copy.selected_color_),
    selected_alpha_(copy.selected_alpha_),
    thin_line_width_(copy.thin_line_width_),
    thick_line_width_(copy.thick_line_width_),
    point_size_(copy.point_size_),
    color_(copy.color_),
    alpha_(copy.alpha_),
    selected_(copy.selected_)
{}

void
CM2Widget::set_alpha(float a)
{
  alpha_ = Clamp((double)a, 0.0, 1.0);
}


TriangleCM2Widget::TriangleCM2Widget()
  : base_(0.5), top_x_(0.15), top_y_(0.5), width_(0.25), bottom_(0.5)
{
  color_.r(1.0);
  color_.g(1.0);
  color_.b(1.0);
  alpha_ = 1.0;
}

TriangleCM2Widget::TriangleCM2Widget(float base, float top_x, float top_y,
                                     float width, float bottom)
  : base_(base), top_x_(top_x), top_y_(top_y), width_(width), bottom_(bottom)
{}

TriangleCM2Widget::~TriangleCM2Widget()
{}

TriangleCM2Widget::TriangleCM2Widget(TriangleCM2Widget& copy)
  : CM2Widget(copy),
    base_(copy.base_),
    top_x_(copy.top_x_),
    top_y_(copy.top_y_),
    width_(copy.width_),
    bottom_(copy.bottom_),
    last_x_(copy.last_x_),
    last_y_(copy.last_y_),
    last_width_(copy.last_width_),
    pick_ix_(copy.pick_ix_),
    pick_iy_(copy.pick_iy_)
{}

CM2Widget*
TriangleCM2Widget::clone()
{
  return new TriangleCM2Widget(*this);
}

void
TriangleCM2Widget::rasterize(CM2ShaderFactory& factory, bool faux, Pbuffer* pbuffer)
{
  CM2BlendType blend = CM2_BLEND_RASTER;
  if(pbuffer) {
    if(pbuffer->need_shader())
      blend = CM2_BLEND_FRAGMENT_NV;
    else
      blend = CM2_BLEND_FRAGMENT_ATI;
  }
  FragmentProgramARB* shader = factory.shader(CM2_SHADER_TRIANGLE, faux, blend);

  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
  
    shader->bind();
    shader->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
    shader->setLocalParam(1, base_, base_+top_x_, top_y_, 0.0);
    shader->setLocalParam(2, width_, bottom_, 0.0, 0.0);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    shader->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
    if(pbuffer)
      shader->setLocalParam(4, 1.0/pbuffer->width(), 1.0/pbuffer->height(), 0.0, 0.0);
    
    glBegin(GL_TRIANGLES);
    {
      glVertex2f(base_, 0.0);
      glVertex2f(base_+top_x_+width_/2, top_y_);
      glVertex2f(base_+top_x_-width_/2, top_y_);
    }
    glEnd();
    shader->release();
  }
}

void
TriangleCM2Widget::rasterize(Array3<float>& array, bool faux)
{
  //std::cerr << tex->size(0) << " " << tex->size(1) << std::endl;
  if(array.dim3() != 4) return;
  int size_x = array.dim2();
  int size_y = array.dim1();
  float top_left = top_x_-std::abs(width_)/2;
  float top_right = top_x_+std::abs(width_)/2;
  int lb = (int)(bottom_*top_y_*size_y);
  int le = (int)(top_y_*size_y);
  int ilb = Clamp(lb, 0, size_y-1);
  int ile = Clamp(le, 0, size_y-1);
  //cerr << lb << " | " << le << endl;
  if(faux) {
    for(int i=ilb; i<=ile; i++) {
      float fb = (i/(float)le)*top_left + base_;
      float fe = (i/(float)le)*top_right + base_;
      float fm = (i/(float)le)*top_x_ + base_;
      int rb = (int)(fb*size_x);
      int re = (int)(fe*size_x);
      int rm = (int)(fm*size_x);
      int jrb = Clamp(rb, 0, size_x-1);
      int jre = Clamp(re, 0, size_x-1);
      int jrm = Clamp(rm, 0, size_x-1);
      float da = alpha_/(rm-rb);
      float dr = color_.r()/(rm-rb);
      float dg = color_.g()/(rm-rb);
      float db = color_.b()/(rm-rb);
      float a = alpha_-std::abs(rm-jrm+1)*da;
      float r = color_.r()-std::abs(rm-jrm+1)*dr;
      float g = color_.g()-std::abs(rm-jrm+1)*dg;
      float b = color_.b()-std::abs(rm-jrm+1)*db;
      for(int j=jrm-1; j>=jrb; j--, a-=da, r-=dr, b-=db, g-=dg) {
        array(i,j,0) = Clamp(array(i,j,0)*(1-a) + r, 0.0f, 1.0f);
        array(i,j,1) = Clamp(array(i,j,1)*(1-a) + g, 0.0f, 1.0f);
        array(i,j,2) = Clamp(array(i,j,2)*(1-a) + b, 0.0f, 1.0f);
        array(i,j,3) = Clamp(array(i,j,3)*(1-a) + a, 0.0f, 1.0f);
      }
      da = alpha_/(re-rm);
      dr = color_.r()/(re-rm);
      dg = color_.g()/(re-rm);
      db = color_.b()/(re-rm);
      a = alpha_-std::abs(rm-jrm)*da;
      r = color_.r()-std::abs(rm-jrm)*dr;
      g = color_.g()-std::abs(rm-jrm)*dg;
      b = color_.b()-std::abs(rm-jrm)*db;
      //cerr << mTop.x << " " << fm << " -> " << fe << std::endl;
      for (int j=jrm; j<=jre; j++, a-=da, r-=dr, b-=db, g-=dg)
      {
        array(i,j,0) = array(i,j,0)*(1-a) + r;
        array(i,j,1) = array(i,j,1)*(1-a) + g;
        array(i,j,2) = array(i,j,2)*(1-a) + b;
        array(i,j,3) = array(i,j,3)*(1-a) + a;
      }
    }
  } else {
    for(int i=ilb; i<=ile; i++) {
      float fb = (i/(float)le)*top_left + base_;
      float fe = (i/(float)le)*top_right + base_;
      float fm = (i/(float)le)*top_x_ + base_;
      int rb = (int)(fb*size_x);
      int re = (int)(fe*size_x);
      int rm = (int)(fm*size_x);
      int jrb = Clamp(rb, 0, size_x-1);
      int jre = Clamp(re, 0, size_x-1);
      int jrm = Clamp(rm, 0, size_x-1);
      float da = alpha_/(rm-rb);
      float a = alpha_-std::abs(rm-jrm+1)*da;
      for(int j=jrm-1; j>=jrb; j--, a-=da) {
        array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + (float)color_.r(), 0.0f, 1.0f);
        array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + (float)color_.g(), 0.0f, 1.0f);
        array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + (float)color_.b(), 0.0f, 1.0f);
        array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
      }
      da = alpha_/(re-rm);
      a = alpha_-std::abs(rm-jrm)*da;
      //cerr << mTop.x << " " << fm << " -> " << fe << std::endl;
      for (int j=jrm; j<=jre; j++, a-=da)
      {
        array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + (float)color_.r(), 0.0f, 1.0f);
        array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + (float)color_.g(), 0.0f, 1.0f);
        array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + (float)color_.b(), 0.0f, 1.0f);
        array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
      }
    }
  }
}

void
TriangleCM2Widget::draw()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_LINE_SMOOTH);
  glLineWidth(thin_line_width_);
  glBegin(GL_LINES);
  {
    selectcolor(1);
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_-width_/2, top_y_);
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_+width_/2, top_y_);
  }
  glEnd();

  const float b_x = bottom_*top_x_ + base_;
  const float b_y = bottom_*top_y_;
  const float w = bottom_*width_;
  glLineWidth(thick_line_width_);
  glBegin(GL_LINES);
  {
    selectcolor(4);
    glVertex2f(base_+top_x_-width_/2, top_y_);
    glVertex2f(base_+top_x_+width_/2, top_y_);
  }
  glEnd();
  glLineWidth(thin_line_width_);
  glBegin(GL_LINES);
  {
    selectcolor(1);
    glVertex2f(b_x-w/2, b_y);
    glVertex2f(b_x+w/2, b_y);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glEnable(GL_POINT_SMOOTH);
  glPointSize(point_size_);
  glBegin(GL_POINTS);
  {
    selectcolor(2);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    selectcolor(3);
    glVertex2f(b_x-w/2, b_y);
  }
  glEnd();
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}


int
TriangleCM2Widget::pick1 (int ix, int iy, int sw, int sh)
{
  const double x = ix / (double)sw;
  const double y = iy / (double)sh;
  const double xeps = point_size_ / sw * 0.5;
  const double yeps = point_size_ / sh * 0.5;
  const double yleps = thick_line_width_ / sh;

  const float b_x = bottom_*top_x_ + base_;
  const float b_y = bottom_*top_y_;
  const float w = bottom_*width_;

  const double cp1x = base_ + top_x_ + width_/2;
  const double cp1y = top_y_;
  if (fabs(x - cp1x) < xeps &&
      fabs(y - cp1y) < yeps)
  {
    return 2;
  }

  const double cp2x = b_x - w/2;
  const double cp2y = b_y;
  if (fabs(x - cp2x) < xeps &&
      fabs(y - cp2y) < yeps)
  {
    return 3;
  }

  const double cp1x2 = base_ + top_x_ - width_/2;
  if (fabs(y - top_y_) < yleps &&
      x > Min(cp1x, cp1x2) && x < Max(cp1x, cp1x2))
  {
    last_x_ = top_x_;
    last_y_ = top_y_;
    pick_ix_ = ix;
    pick_iy_ = iy;
    last_width_ = width_;
    return 4;
  }

  return 0;
}


int
TriangleCM2Widget::pick2 (int ix, int iy, int sw, int sh, int m)
{
  const double x = ix / (double)sw;
  const double y = iy / (double)sh;

  const double x1top = top_x_ + width_ * 0.5;
  const double x2top = top_x_ - width_ * 0.5;
  const double x1 = base_ + x1top * (y / top_y_);
  const double x2 = base_ + x2top * (y / top_y_);
  if (y < top_y_ &&
      x > Min(x1, x2) && x < Max(x1, x2))
  {
    last_x_ = base_;
    pick_ix_ = ix;
    pick_iy_ = iy; // modified only.
    last_hsv_ = HSVColor(color_);  // modified only
    return m?5:1;
  }
  
  return 0;
}


void
TriangleCM2Widget::move (int obj, int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  
  switch (selected_)
  {
  case 1:
    base_ = last_x_ + x - pick_ix_ / (double)w;
    break;

  case 2:
    width_ = (x - top_x_ - base_) * 2.0;
    break;

  case 3:
    bottom_ = Clamp(y / top_y_, 0.0, 1.0);
    break;

  case 4:
    top_x_ = last_x_ + x - pick_ix_ / (double)w;
    top_y_ = last_y_ + y - pick_iy_ / (double)h;
    width_ = last_width_ * top_y_ / last_y_;
    break;
    
  case 5:
    {
      // Hue controls on x axis
      const double hdelta = x - pick_ix_ / (double)w;
      double hue = last_hsv_[0] + hdelta * 360.0 * 2.0;
      while (hue < 0.0) hue += 360.0;
      while (hue > 360.0) hue -= 360;

      // Saturation controls on y axis
      const double sdelta = y - pick_iy_ / (double)h;
      double sat = Clamp(last_hsv_[1] - sdelta * 2.0, 0.0, 1.0);

      HSVColor hsv(hue, sat, last_hsv_.val());
      color_ = Color(hsv);
    }
    break;
  }
}


void
TriangleCM2Widget::release (int obj, int x, int y, int w, int h)
{
}


string
TriangleCM2Widget::tcl_pickle()
{
  ostringstream s;
  s << "t ";
  s << base_ << " ";
  s << top_x_ << " ";
  s << top_y_ << " ";
  s << width_ << " ";
  s << bottom_;
  return s.str();
}

void
TriangleCM2Widget::tcl_unpickle(const string &p)
{
  istringstream s(p);
  char c;
  s >> c;
  s >> base_;
  s >> top_x_;
  s >> top_y_;
  s >> width_;
  s >> bottom_;
}


RectangleCM2Widget::RectangleCM2Widget()
  : type_(CM2_RECTANGLE_1D), left_x_(0.25), left_y_(0.5), width_(0.25), height_(0.25),
    offset_(0.25)
{
  color_.r(1.0);
  color_.g(1.0);
  color_.b(0.7);
  alpha_ = 1.0;
}

RectangleCM2Widget::RectangleCM2Widget(CM2RectangleType type, float left_x, float left_y,
                                       float width, float height, float offset)
  : type_(type), left_x_(left_x), left_y_(left_y), width_(width), height_(height),
    offset_(offset)
{}

RectangleCM2Widget::~RectangleCM2Widget()
{}

RectangleCM2Widget::RectangleCM2Widget(RectangleCM2Widget& copy)
  : CM2Widget(copy),
    type_(copy.type_),
    left_x_(copy.left_x_),
    left_y_(copy.left_y_),
    width_(copy.width_),
    height_(copy.height_),
    offset_(copy.offset_),
    last_x_(copy.last_x_),
    last_y_(copy.last_y_),
    pick_ix_(copy.pick_ix_),
    pick_iy_(copy.pick_iy_)
{}

CM2Widget*
RectangleCM2Widget::clone()
{
  return new RectangleCM2Widget(*this);
}

void
RectangleCM2Widget::rasterize(CM2ShaderFactory& factory, bool faux, Pbuffer* pbuffer)
{
  CM2BlendType blend = CM2_BLEND_RASTER;
  if(pbuffer) {
    if(pbuffer->need_shader())
      blend = CM2_BLEND_FRAGMENT_NV;
    else
      blend = CM2_BLEND_FRAGMENT_ATI;
  }
  CM2ShaderType type = CM2_SHADER_RECTANGLE_1D;
  if(type_ == CM2_RECTANGLE_ELLIPSOID)
    type = CM2_SHADER_RECTANGLE_ELLIPSOID;
  FragmentProgramARB* shader = factory.shader(type, faux, blend);

  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
    shader->bind();
    shader->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
    shader->setLocalParam(1, left_x_, left_y_, width_, height_);
    if(offset_ < std::numeric_limits<float>::epsilon())
      shader->setLocalParam(2, offset_, 0.0, 1.0, 0.0);
    else if((1.0-offset_) < std::numeric_limits<float>::epsilon())
      shader->setLocalParam(2, offset_, 1.0, 0.0, 0.0);
    else
      shader->setLocalParam(2, offset_, 1/offset_, 1/(1-offset_), 0.0);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    shader->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
    if(pbuffer)
      shader->setLocalParam(4, 1.0/pbuffer->width(), 1.0/pbuffer->height(), 0.0, 0.0);
  
    glBegin(GL_QUADS);
    {
      glVertex2f(left_x_, left_y_);
      glVertex2f(left_x_+width_, left_y_);
      glVertex2f(left_x_+width_, left_y_+height_);
      glVertex2f(left_x_, left_y_+height_);
    }
    glEnd();
    shader->release();
  }
}

void
RectangleCM2Widget::rasterize(Array3<float>& array, bool faux)
{
  if(array.dim3() != 4) return;
  int size_x = array.dim2();
  int size_y = array.dim1();
  float left = left_x_;
  float right = left_x_+width_;
  float bottom = left_y_;
  float top = left_y_+height_;

  int lb = int(bottom*size_y);
  int le = int(top*size_y);
  int ilb = Clamp(lb, 0, size_y-1);
  int ile = Clamp(le, 0, size_y-1);
  //int la = int((mBall.y*mSize.y+bottom)*size.y);
  int rb = int(left*size_x);
  int re = int(right*size_x);
  int ra = int((offset_*width_+left)*size_x);
  int jrb = Clamp(rb, 0, size_x-1);
  int jre = Clamp(re, 0, size_x-1);
  int jra = Clamp(ra, 0, size_x-1);
  switch(type_) {
    case CM2_RECTANGLE_ELLIPSOID: {
      for(int i=ilb; i<=ile; i++) {
        for(int j=jrb; j<jre; j++) {
          float x = j/(float)size_x;
          float y = i/(float)size_y;
          x -= (left+right)/2;
          y -= (bottom+top)/2;
          x *= height_/width_;
          float w = 1-2*sqrt(x*x+y*y)/size_y;
          if (w < 0) w = 0;
          float a = alpha_*w;
          float r = faux ? color_.r()*w : color_.r();
          float g = faux ? color_.r()*w : color_.g();
          float b = faux ? color_.r()*w : color_.b();
          array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + r, 0.0f, 1.0f);
          array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + g, 0.0f, 1.0f);
          array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + b, 0.0f, 1.0f);
          array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
        }
      }
    } break;

    case CM2_RECTANGLE_1D: {
      if(faux) {
        float da = ra <= rb+1 ? 0.0 : alpha_/(ra-rb-1);
        float dr = ra <= rb+1 ? 0.0 : color_.r()/(ra-rb-1);
        float dg = ra <= rb+1 ? 0.0 : color_.g()/(ra-rb-1);
        float db = ra <= rb+1 ? 0.0 : color_.b()/(ra-rb-1);
        float a = ra <= rb+1 ? alpha_ : alpha_-std::abs(ra-jra)*da;
        float r = ra <= rb+1 ? color_.r() : color_.r()-std::abs(ra-jra)*dr;
        float g = ra <= rb+1 ? color_.g() : color_.g()-std::abs(ra-jra)*dg;
        float b = ra <= rb+1 ? color_.b() : color_.b()-std::abs(ra-jra)*db;
        for(int j=jra-1; j>=jrb; j--, a-=da, r-=dr, b-=db, g-=dg) {
          for(int i=ilb; i<=ile; i++) {
          
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + r, 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + g, 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + b, 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
          }
        }
        da = ra < re-1 ? alpha_/(re-ra-1) : 0.0;
        dr = ra < re-1 ? color_.r()/(re-ra-1) : 0.0;
        dg = ra < re-1 ? color_.g()/(re-ra-1) : 0.0;
        db = ra < re-1 ? color_.b()/(re-ra-1) : 0.0;
        a = alpha_-std::abs(ra-jra)*da;
        r = color_.r()-std::abs(ra-jra)*dr;
        g = color_.g()-std::abs(ra-jra)*dg;
        b = color_.b()-std::abs(ra-jra)*db;
        for(int j=jra; j<=jre; j++, a-=da, r-=dr, b-=db, g-=dg) {
          for(int i=ilb; i<=ile; i++) {
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + r, 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + g, 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + b, 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
          }
        }
      } else { // !faux
        float da = ra <= rb+1 ? 0.0 : alpha_/(ra-rb-1);
        float a = ra <= rb+1 ? alpha_ : alpha_-std::abs(ra-jra)*da;
        for(int j=jra-1; j>=jrb; j--, a-=da) {
          for(int i=ilb; i<=ile; i++) {
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + (float)color_.r(), 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + (float)color_.g(), 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + (float)color_.b(), 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
          }
        }
        da = ra < re-1 ? alpha_/(re-ra-1) : 0.0;
        a = alpha_-std::abs(ra-jra)*da;
        for(int j=jra; j<=jre; j++, a-=da) {
          for(int i=ilb; i<=ile; i++) {
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + (float)color_.r(), 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + (float)color_.g(), 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + (float)color_.b(), 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
          }
        }
      }
    } break;

  default:
    break;
  }
}


void
CM2Widget::selectcolor(int obj)
{
  if (selected_ == obj)
  {
    glColor4f(selected_color_.r(), selected_color_.g(),
	      selected_color_.b(), selected_alpha_);
  }
  else
  {
    glColor4f(line_color_.r(), line_color_.g(), line_color_.b(), line_alpha_);
  }
}


void
RectangleCM2Widget::draw()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  selectcolor(1);

  glEnable(GL_LINE_SMOOTH);
  glLineWidth(thin_line_width_);
  glBegin(GL_LINES);
  {
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_+height_);
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_, left_y_+height_);
    glVertex2f(left_x_, left_y_+height_);
    glVertex2f(left_x_+width_, left_y_+height_);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glEnable(GL_POINT_SMOOTH);
  glPointSize(point_size_);
  glBegin(GL_POINTS);
  {
    selectcolor(2);
    glVertex2f(left_x_, left_y_);
    selectcolor(3);
    glVertex2f(left_x_+width_, left_y_);
    selectcolor(4);
    glVertex2f(left_x_+width_, left_y_+height_);
    selectcolor(5);
    glVertex2f(left_x_, left_y_+height_);
    selectcolor(6);
    glVertex2f(left_x_+offset_*width_, left_y_+height_*0.5);
  }
  glEnd();
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}


int
RectangleCM2Widget::pick1 (int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  const double xeps = point_size_ / w * 0.5;
  const double yeps = point_size_ / h * 0.5;

  if (fabs(x - left_x_) < xeps &&
      fabs(y - left_y_) < yeps)
  {
    return 2;
  }

  if (fabs(x - left_x_ - width_) < xeps &&
      fabs(y - left_y_) < yeps)
  {
    return 3;
  }

  if (fabs(x - left_x_ - width_) < xeps &&
      fabs(y - left_y_ - height_) < yeps)
  {
    return 4;
  }

  if (fabs(x - left_x_) < xeps &&
      fabs(y - left_y_ - height_) < yeps)
  {
    return 5;
  }

  if (fabs(x - left_x_ - offset_ * width_) < xeps &&
      fabs(y - left_y_ - height_ * 0.5) < yeps)
  {
    return 6;
  }

  return 0;
}


int
RectangleCM2Widget::pick2 (int ix, int iy, int w, int h, int m)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;

  if (x > Min(left_x_, left_x_ + width_) &&
      x < Max(left_x_, left_x_ + width_) &&
      y > Min(left_y_, left_y_ + height_) &&
      y < Max(left_y_, left_y_ + height_))
  {
    last_x_ = left_x_;
    last_y_ = left_y_;
    pick_ix_ = ix;
    pick_iy_ = iy;
    last_hsv_ = HSVColor(color_);
    return m?7:1;
  }

  return 0;
}


void
RectangleCM2Widget::move (int obj, int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  
  switch (selected_)
  {
  case 1:
    left_x_ = last_x_ + x - pick_ix_ / (double)w;
    left_y_ = last_y_ + y - pick_iy_ / (double)h;
    break;
      
  case 2:
    width_ = width_ + left_x_ - x;
    left_x_ = x;
    height_ = height_ + left_y_ - y;
    left_y_ = y;
    break;

  case 3:
    width_ = x - left_x_;
    height_ = height_ + left_y_ - y;
    left_y_ = y;
    break;

  case 4:
    width_ = x - left_x_;
    height_ = y - left_y_;
    break;

  case 5:
    width_ = width_ + left_x_ - x;
    left_x_ = x;
    height_ = y - left_y_;
    break;

  case 6:
    offset_ = Clamp((x - left_x_) / width_, 0.0, 1.0);
    break;

  case 7:
    {
      // Hue controls on x axis
      const double hdelta = x - pick_ix_ / (double)w;
      double hue = last_hsv_[0] + hdelta * 360.0 * 2.0;
      while (hue < 0.0) hue += 360.0;
      while (hue > 360.0) hue -= 360;

      // Saturation controls on y axis
      const double sdelta = y - pick_iy_ / (double)h;
      double sat = Clamp(last_hsv_[1] - sdelta * 2.0, 0.0, 1.0);

      HSVColor hsv(hue, sat, last_hsv_.val());
      color_ = Color(hsv);
    }
    break;
  }
}


void
RectangleCM2Widget::release (int obj, int x, int y, int w, int h)
{
}


string
RectangleCM2Widget::tcl_pickle()
{
  ostringstream s;
  s << "r ";
  s << (int)type_ << " ";
  s << left_x_ << " ";
  s << left_y_ << " ";
  s << width_ << " ";
  s << height_ << " ";
  s << offset_;
  return s.str();
}

void
RectangleCM2Widget::tcl_unpickle(const string &p)
{
  istringstream s(p);
  char c;
  s >> c;
  int t;
  s >> t;
  type_ = (CM2RectangleType)t;
  s >> left_x_;
  s >> left_y_;
  s >> width_;
  s >> height_;
  s >> offset_;
}


} // End namespace Volume
