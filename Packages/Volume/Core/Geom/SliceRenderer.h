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
class FragmentProgramARB;

class SliceRenderer : public TextureRenderer
{
public:
  SliceRenderer();
  SliceRenderer(TextureHandle tex, ColorMapHandle map, Colormap2Handle cmap2);
  SliceRenderer(const SliceRenderer&);
  ~SliceRenderer();

  virtual void BuildTransferFunction();
  virtual void BuildTransferFunction2();

  void SetControlPoint( const Point& point){ control_point_ = point; }

  void SetX(bool b){ if(b){draw_view_ = false;} drawX_ = b; }
  void SetY(bool b){ if(b){draw_view_ = false;} drawY_ = b; }
  void SetZ(bool b){ if(b){draw_view_ = false;} drawZ_ = b; }
  void SetView(bool b){ if(b){drawX_=false; drawY_=false; drawZ_=false;}
                        draw_view_ = b; }

  bool drawX() const { return drawX_; }
  bool drawY() const { return drawY_; }
  bool drawZ() const { return drawZ_; }
  bool drawView() const { return draw_view_; }

  bool draw_phi_0() const { return draw_phi0_; }
  bool draw_phi_1() const { return draw_phi1_; }
  double phi0() const { return phi0_; }
  double phi1() const { return phi1_; }
  bool draw_cyl() const { return draw_cyl_; }

  void set_cylindrical(bool cyl_active, bool draw_phi0, double phi0, 
		       bool draw_phi1, double phi1) 
  {
    draw_cyl_ = cyl_active;
    draw_phi0_ = draw_phi0;
    phi0_ = phi0;
    draw_phi1_ = draw_phi1;
    phi1_ = phi1;
  }

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void draw();
  virtual void drawWireFrame();
  virtual void load_colormap();
protected:
  void draw(Brick& b, Polygon* poly);
#endif
  
public:
  virtual GeomObj* clone();
  
protected:
  Point                 control_point_;

  bool                  drawX_;
  bool                  drawY_;
  bool                  drawZ_;
  bool                  draw_view_;
  bool                  draw_phi0_;
  double                phi0_;
  bool                  draw_phi1_;
  double                phi1_;
  bool                  draw_cyl_;
  unsigned char     transfer_function_[1024];

  FragmentProgramARB* VolShader1;
  FragmentProgramARB* VolShader4;
  FragmentProgramARB* FogVolShader1;
  FragmentProgramARB* FogVolShader4;
};

} // end namespace SCIRun

#endif // SliceRenderer_h

