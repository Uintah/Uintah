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
#include <Packages/Volume/Core/Datatypes/CM2Widget.h>
#include <Packages/Volume/Core/Util/Utils.h>
#include <Core/Math/MinMax.h>

#include <iostream>
using namespace std;

using namespace SCIRun;

namespace Volume {

const string CM2ShaderString[CM2_LAST] =
  {
    // CM2_TRIANGLE
    "!!ARBfp1.0 \n"
    "PARAM color = program.local[0]; \n"
    "PARAM geom0 = program.local[1]; # {base, top_x, top_y, 0.0} \n"
    "PARAM geom1 = program.local[2]; # {width, bottom, 0.0, 0.0} \n"
    "PARAM sz = program.local[3]; # {1/sx, 1/sy, 0.0, 0.0} \n"
    "TEMP c, p, t;"
    "MUL p.xy, fragment.position.xyyy, sz.xyyy; \n"
    "MUL p.z, geom1.y, geom0.z; \n"
    "SUB p.z, p.y, p.z; \n"
    "KIL p.z; \n"
    "RCP t.z, geom0.z; \n"
    "MUL t.x, p.y, t.z; \n"
    "LRP c.x, t.x, geom0.y, geom0.x; \n"
    "MUL c.y, t.x, geom1.x; \n"
    "MUL c.y, c.y, 0.5; \n"
    "RCP c.y, c.y; \n"
    "SUB c.z, p.x, c.x; \n"
    "MUL c.z, c.y, c.z; \n"
    "ABS c.z, c.z; \n"
    "SUB c.z, 1.0, c.z; \n"
    "MUL c.w, color.w, c.z; \n"
    "MOV c.xyz, color.xyzz; \n"
    "MOV result.color, c; \n"
    "END",
    // CM2_RECTANGLE_1D
    "!!ARBfp1.0 \n"
    "PARAM color = program.local[0]; \n"
    "PARAM geom0 = program.local[1]; # {left_x, left_y, width, height} \n"
    "PARAM geom1 = program.local[2]; # {offset, 1/offset, 1/(1-offset), 0.0} \n"
    "PARAM sz = program.local[3]; # {1/sx, 1/sy, 0.0, 0.0} \n"
    "TEMP c, p, t; \n"
    "MUL p.xy, fragment.position.xyyy, sz.xyyy; \n"
    "SUB p.xy, p.xyyy, geom0.xyyy; \n"
    "RCP p.z, geom0.z; \n"
    "RCP p.w, geom0.w; \n"
    "MUL p.xy, p.xyyy, p.zwww; \n"
    "SUB t.x, p.x, geom1.x; \n"
    "MUL t.y, t.x, geom1.y; \n"
    "MUL t.z, t.x, geom1.z; \n"
    "CMP t.w, t.y, t.y, t.z; \n"
    "ABS t.w, t.w; \n"
    "SUB t.w, 1.0, t.w; \n"
    "MUL c.w, color.w, t.w; \n"
    "MOV c.xyz, color.xyzz; \n"
    "MOV result.color, c; \n"
    "END",
    // CM2_RECTANGLE_ELLIPSOID
    "!!ARBfp1.0 \n"
    "END"
  };

CM2ShaderFactory::CM2ShaderFactory()
{
  shader_ = new FragmentProgramARB*[CM2_LAST];
  for(int i=0; i<CM2_LAST; i++) {
    shader_[i] = new FragmentProgramARB(CM2ShaderString[i]);
  }
}

CM2ShaderFactory::~CM2ShaderFactory()
{
  for(int i=0; i<CM2_LAST; i++) {
    delete shader_[i];
  }
  delete [] shader_;
}

bool
CM2ShaderFactory::create()
{
  for(int i=0; i<CM2_LAST; i++) {
    if(shader_[i]->create()) return true;
  }
  return false;
}

void
CM2ShaderFactory::destroy()
{
  for(int i=0; i<CM2_LAST; i++) {
    shader_[i]->destroy();
  }
}

FragmentProgramARB*
CM2ShaderFactory::shader(int type)
{
  return type < CM2_LAST ? shader_[type] : 0;
}

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

TriangleCM2Widget::TriangleCM2Widget()
  : base_(0.5), top_x_(0.15), top_y_(0.5), width_(0.25), bottom_(0.5)
{}

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
TriangleCM2Widget::rasterize(CM2ShaderFactory& factory)
{
  FragmentProgramARB* shader = factory.shader(CM2_TRIANGLE);
  shader->bind();
  shader->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
  shader->setLocalParam(1, base_, base_+top_x_, top_y_, 0.0);
  shader->setLocalParam(2, width_, bottom_, 0.0, 0.0);

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  shader->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
    
  glBegin(GL_TRIANGLES);
  {
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    glVertex2f(base_+top_x_-width_/2, top_y_);
  }
  glEnd();
  shader->release();
}

void
TriangleCM2Widget::rasterize(Array3<float>& array)
{
  //std::cerr << tex->size(0) << " " << tex->size(1) << std::endl;
  if(array.dim3() != 4) return;
  int size_x = array.dim2();
  int size_y = array.dim1();
  float top_left = top_x_-std::abs(width_)/2;
  float top_right = top_x_+std::abs(width_)/2;
  int lb = (int)(bottom_*top_y_*size_y);
  int le = (int)(top_y_*size_y);
  int ilb = CLAMP(lb, 0, size_y-1);
  int ile = CLAMP(le, 0, size_y-1);
  //cerr << lb << " | " << le << endl;
  for(int i=ilb; i<=ile; i++) {
    float fb = (i/(float)le)*top_left + base_;
    float fe = (i/(float)le)*top_right + base_;
    float fm = (i/(float)le)*top_x_ + base_;
    int rb = (int)(fb*size_x);
    int re = (int)(fe*size_x);
    int rm = (int)(fm*size_x);
    int jrb = CLAMP(rb, 0, size_x-1);
    int jre = CLAMP(re, 0, size_x-1);
    int jrm = CLAMP(rm, 0, size_x-1);
    float da = alpha_/(rm-rb);
    float a = alpha_-std::abs(rm-jrm+1)*da;
    for(int j=jrm-1; j>=jrb; j--, a-=da) {
      array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
      array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
      array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
      array(i,j,3) = array(i,j,3)*(1-a) + a;
    }
    da = alpha_/(re-rm);
    a = alpha_-std::abs(rm-jrm)*da;
    //cerr << mTop.x << " " << fm << " -> " << fe << std::endl;
    for (int j=jrm; j<=jre; j++, a-=da)
    {
      array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
      array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
      array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
      array(i,j,3) = array(i,j,3)*(1-a) + a;
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
TriangleCM2Widget::pick2 (int ix, int iy, int sw, int sh)
{
  const double x = ix / (double)sw;
  const double y = iy / (double)sh;

  const double x1top = top_x_ + width_ * 0.5;
  const double x2top = top_x_ - width_ * 0.5;
  const double x1 = base_ + x1top * (y / top_y_);
  const double x2 = base_ + x2top * (y / top_y_);
  if (y < top_y_ && y > bottom_ * top_y_ &&
      x > Min(x1, x2) && x < Max(x1, x2))
  {
    last_x_ = base_;
    pick_ix_ = ix;
    return 1;
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
    bottom_ = CLAMP(y / top_y_, 0.0, 1.0);
    break;

  case 4:
    top_x_ = last_x_ + x - pick_ix_ / (double)w;
    top_y_ = last_y_ + y - pick_iy_ / (double)h;
    width_ = last_width_ * top_y_ / last_y_;
    break;
  }
}


void
TriangleCM2Widget::release (int obj, int x, int y, int w, int h)
{
  //move(obj, x, y, w, h);
}


RectangleCM2Widget::RectangleCM2Widget()
  : type_(CM2_RECTANGLE_1D), left_x_(0.25), left_y_(0.5), width_(0.25), height_(0.25),
    offset_(0.25)
{}

RectangleCM2Widget::RectangleCM2Widget(int type, float left_x, float left_y,
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
RectangleCM2Widget::rasterize(CM2ShaderFactory& factory)
{
  FragmentProgramARB* shader = factory.shader(type_);

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

void
RectangleCM2Widget::rasterize(Array3<float>& array)
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
  int ilb = CLAMP(lb, 0, size_y-1);
  int ile = CLAMP(le, 0, size_y-1);
  //int la = int((mBall.y*mSize.y+bottom)*size.y);
  int rb = int(left*size_x);
  int re = int(right*size_x);
  int ra = int((offset_*width_+left)*size_x);
  int jrb = CLAMP(rb, 0, size_x-1);
  int jre = CLAMP(re, 0, size_x-1);
  int jra = CLAMP(ra, 0, size_x-1);
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
          array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
          array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
          array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
          array(i,j,3) = array(i,j,3)*(1-a) + a;
        }
      }
    } break;

    case CM2_RECTANGLE_1D: {
      float da = ra <= rb+1 ? 0.0 : alpha_/(ra-rb-1);
      float a = ra <= rb+1 ? alpha_ : alpha_-std::abs(ra-jra)*da;
      for(int j=jra-1; j>=jrb; j--, a-=da) {
        for(int i=ilb; i<=ile; i++) {
          array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
          array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
          array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
          array(i,j,3) = array(i,j,3)*(1-a) + a;
        }
      }
      da = ra < re-1 ? alpha_/(re-ra-1) : 0.0;
      a = alpha_-std::abs(ra-jra)*da;
      for(int j=jra; j<=jre; j++, a-=da) {
        for(int i=ilb; i<=ile; i++) {
          array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
          array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
          array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
          array(i,j,3) = array(i,j,3)*(1-a) + a;
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
RectangleCM2Widget::pick2 (int ix, int iy, int w, int h)
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
    return 1;
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
    offset_ = CLAMP((x - left_x_) / width_, 0.0, 1.0);
    break;
  }
}


void
RectangleCM2Widget::release (int obj, int x, int y, int w, int h)
{
  //move(obj, x, y, w, h);
}


} // End namespace Volume
