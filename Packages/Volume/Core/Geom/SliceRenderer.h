/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef SLICERENDERER_H
#define SLICERENDERER_H

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>

// #if defined( HAVE_GLEW )
// #include <GL/glew.h>
// #else
// #include <GL/gl.h>
// #include <sci_glu.h>
// #endif

#include <Core/Containers/BinaryTree.h>

#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Geom/TextureRenderer.h>

namespace Volume {

using SCIRun::GeomObj;
using SCIRun::DrawInfoOpenGL;

#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
class FragmentProgramARB;
#endif
class SliceRenderer : public TextureRenderer
{
public:

  SliceRenderer();
  SliceRenderer(TextureHandle tex, ColorMapHandle map);
  SliceRenderer(const SliceRenderer&);
  ~SliceRenderer();

  virtual void BuildTransferFunction();

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
  virtual void setup();
  virtual void cleanup();
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

#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  FragmentProgramARB *VolShader;
#endif

};

} // End namespace SCIRun


#endif
