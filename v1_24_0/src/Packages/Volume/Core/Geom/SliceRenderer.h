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
//    File   : SliceRenderer.h
//    Author : Milan Ikits
//    Date   : Wed Jul  7 23:36:05 2004

#ifndef SliceRenderer_h
#define SliceRenderer_h

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>

#include <Core/Containers/BinaryTree.h>

#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Geom/TextureRenderer.h>

namespace Volume {

using SCIRun::GeomObj;
using SCIRun::DrawInfoOpenGL;

class SliceRenderer : public TextureRenderer
{
public:
  SliceRenderer(TextureHandle tex, ColorMapHandle cmap1, Colormap2Handle cmap2,
                int tex_mem);
  SliceRenderer(const SliceRenderer&);
  ~SliceRenderer();

  inline void set_control_point(const Point& point) { control_point_ = point; }

  inline void set_x(bool b) { if(b) draw_view_ = false; draw_x_ = b; }
  inline void set_y(bool b) { if(b) draw_view_ = false; draw_y_ = b; }
  inline void set_z(bool b) { if(b) draw_view_ = false; draw_z_ = b; }
  inline void set_view(bool b) {
    if(b) {
      draw_x_=false; draw_y_=false; draw_z_=false;
    }
    draw_view_ = b;
  }

  bool draw_x() const { return draw_x_; }
  bool draw_y() const { return draw_y_; }
  bool draw_z() const { return draw_z_; }
  bool draw_view() const { return draw_view_; }
  bool draw_phi_0() const { return draw_phi0_; }
  bool draw_phi_1() const { return draw_phi1_; }
  double phi0() const { return phi0_; }
  double phi1() const { return phi1_; }
  bool draw_cyl() const { return draw_cyl_; }

  void set_cylindrical(bool cyl_active, bool draw_phi0, double phi0, 
		       bool draw_phi1, double phi1) {
    draw_cyl_ = cyl_active;
    draw_phi0_ = draw_phi0;
    phi0_ = phi0;
    draw_phi1_ = draw_phi1;
    phi1_ = phi1;
  }

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void draw();
  virtual void draw_wireframe();
#endif

  virtual GeomObj* clone();

protected:
  Point control_point_;
  bool draw_x_;
  bool draw_y_;
  bool draw_z_;
  bool draw_view_;
  bool draw_phi0_;
  double phi0_;
  bool draw_phi1_;
  double phi1_;
  bool draw_cyl_;
};

} // end namespace SCIRun

#endif // SliceRenderer_h
