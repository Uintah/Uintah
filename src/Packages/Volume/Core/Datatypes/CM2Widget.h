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
//    File   : CM2Widget.h
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:12 2004

#ifndef CM2Widget_h
#define CM2Widget_h

#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Color.h>

namespace Volume {

class CM2ShaderFactory;

class CM2Widget
{
public:
  CM2Widget();
  virtual ~CM2Widget();

  // appearance
  virtual void draw() = 0;
  virtual void rasterize(CM2ShaderFactory& factory) = 0;
  virtual void rasterize(SCIRun::Array3<float>& array) = 0;

  // behavior
  virtual int pick1 (int x, int y, int w, int h) = 0;
  virtual int pick2 (int x, int y, int w, int h) = 0;
  virtual void move (int obj, int x, int y, int w, int h) = 0;
  virtual void release (int obj, int x, int y, int w, int h) = 0;

  void select(int obj) { selected_ = obj; }
  void unselect_all() { selected_ = 0; }
  
protected:
  void selectcolor(int obj);

  SCIRun::Color line_color_;
  float line_alpha_;
  SCIRun::Color selected_color_;
  float selected_alpha_;
  float thin_line_width_;
  float thick_line_width_;
  float point_size_;
  SCIRun::Color color_;
  float alpha_;
  int selected_;
};

class FragmentProgramARB;

enum {
  CM2_TRIANGLE = 0,
  CM2_RECTANGLE_1D = 1,
  CM2_RECTANGLE_ELLIPSOID = 2,
  CM2_LAST = 3
};

class CM2ShaderFactory
{
public:
  CM2ShaderFactory();
  ~CM2ShaderFactory();
  
  bool create();
  void destroy();

  FragmentProgramARB* shader(int type);

protected:
  FragmentProgramARB** shader_;
};

class TriangleCM2Widget : public CM2Widget
{
public:
  TriangleCM2Widget();
  TriangleCM2Widget(float base, float top_x, float top_y,
                    float width, float bottom);
  ~TriangleCM2Widget();

  // appearance
  void draw();
  void rasterize(CM2ShaderFactory& factory);
  void rasterize(SCIRun::Array3<float>& array);
  
  // behavior
  virtual int pick1 (int x, int y, int w, int h);
  virtual int pick2 (int x, int y, int w, int h);
  virtual void move (int obj, int x, int y, int w, int h);
  virtual void release (int obj, int x, int y, int w, int h);
  
protected:
  float base_;
  float top_x_, top_y_;
  float width_;
  float bottom_;

  // Used by picking.
  float last_x_, last_y_, last_width_;
  int pick_ix_, pick_iy_;
};

class RectangleCM2Widget : public CM2Widget
{
public:
  RectangleCM2Widget();
  RectangleCM2Widget(int type, float left_x, float left_y,
                     float width, float height, float offset);
  ~RectangleCM2Widget();

  // appearance
  void draw();
  void rasterize(CM2ShaderFactory& factory);
  void rasterize(SCIRun::Array3<float>& array);
  
  // behavior
  virtual int pick1 (int x, int y, int w, int h);
  virtual int pick2 (int x, int y, int w, int h);
  virtual void move (int obj, int x, int y, int w, int h);
  virtual void release (int obj, int x, int y, int w, int h);
  
protected:
  int type_;
  float left_x_, left_y_;
  float width_, height_, offset_;

  // Used by picking.
  float last_x_, last_y_;
  int pick_ix_, pick_iy_;
};


} // End namespace Volume

#endif // CM2Widget_h
