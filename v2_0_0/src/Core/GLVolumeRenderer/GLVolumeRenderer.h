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

#ifndef GLVOLUMERENDERER_H
#define GLVOLUMERENDERER_H

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>
#include <GL/gl.h>
#include <sci_glu.h>

#include <Core/Containers/Octree.h>

#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/GLTexRenState.h>
#include <Core/GLVolumeRenderer/GLVolRenState.h>
#include <Core/GLVolumeRenderer/FullRes.h>
#include <Core/GLVolumeRenderer/ROI.h>
#include <Core/GLVolumeRenderer/LOS.h>
#include <Core/GLVolumeRenderer/GLPlanes.h>
#include <Core/GLVolumeRenderer/TexPlanes.h>
#include <Core/GLVolumeRenderer/GLMIP.h>
#include <Core/GLVolumeRenderer/GLAttenuate.h>
#include <Core/GLVolumeRenderer/GLOverOp.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>

namespace SCIRun {



class GLVolumeRenderer : public GeomObj
{
public:

  GLVolumeRenderer();
  GLVolumeRenderer(GLTexture3DHandle tex, ColorMapHandle map);
  GLVolumeRenderer(const GLVolumeRenderer&);
  ~GLVolumeRenderer();

  void SetNSlices(int s) { slices_ = s;}
  void SetSliceAlpha( double as){ slice_alpha_ = as;}
  void BuildTransferFunctions();


  void SetVol( GLTexture3DHandle tex ){ 
    mutex_.lock(); tex_ = tex; state_->NewBricks(); mutex_.unlock();}
  void SetColorMap( ColorMapHandle cmap){
    mutex_.lock(); cmap_ = cmap; BuildTransferFunctions();
    cmap_has_changed_ = true; mutex_.unlock(); }
  void SetControlPoint( const Point& point){ control_point_ = point; }

  void Reload() { state_->Reload(); }


  void DrawFullRes(){ state_ = state(fr_, 0);}
  void DrawLOS(){ state_ = state(los_, 0);}
  void DrawROI(){ state_ = state(roi_, 0);}
  void DrawPlanes(){ state_ = state(tp_, 1);}

  void SetX(bool b){ if(b){drawView_ = false;} drawX_ = b; }
  void SetY(bool b){ if(b){drawView_ = false;} drawY_ = b; }
  void SetZ(bool b){ if(b){drawView_ = false;} drawZ_ = b; }
  void SetView(bool b){ if(b){drawX_=false; drawY_=false; drawZ_=false;}
                        drawView_ = b; }

  enum tex_ren_mode_e {
    TRS_GLOverOp,
    TRS_GLMIP,
    TRS_GLAttenuate,
    TRS_GLPlanes
  };

  void set_tex_ren_state(tex_ren_mode_e mode) {
    switch (mode) {
    case TRS_GLOverOp:
      tex_ren_state_ = state(oo_, 0);
      break;
    case TRS_GLMIP:
      tex_ren_state_ = state(mip_, 0);
      break;
    case TRS_GLAttenuate:
      tex_ren_state_ = state(atten_, 0);
      break;
    case TRS_GLPlanes:
      tex_ren_state_ = state(planes_, 1);
      break;
    default:
      ASSERTFAIL("invalid tex_ren_state");
    }
  }

  void SetInterp( bool i) { interp_ = i; }



#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb){ tex_->get_bounds( bb ); }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);

  void setup();
  void preDraw(){ tex_ren_state_->preDraw(); }
  void draw(){ state_->draw(); }
  void postDraw(){ tex_ren_state_->postDraw(); }
  void cleanup();

  void drawWireFrame(){ glColor4f(0.8,0.8,0.8,1.0); state_->drawWireFrame(); }
  void set_cylindrical(bool cyl_active, bool draw_phi0, double phi0, 
		       bool draw_phi1, double phi1) {
    draw_cyl_ = cyl_active;
    draw_phi0_ = draw_phi0;
    phi0_ = phi0;
    draw_phi1_ = draw_phi1;
    phi1_ = phi1;
  }
  
  //! accessors.
  GLTexture3DHandle get_tex3d_handle() const { return tex_; }
  const Point &control_point() const {return control_point_;}
  GLTexture3DHandle tex() const { return tex_; }
  DrawInfoOpenGL* di() const { return di_; }
  const unsigned char *transfer_functions(int i) const 
  { return &transfer_functions_[i][0]; } 
  bool interp() const { return interp_; }
  int slices() const { return slices_; }
  double slice_alpha() const { return slice_alpha_; }
  bool drawX() const { return drawX_; }
  bool drawY() const { return drawY_; }
  bool drawZ() const { return drawZ_; }
  bool drawView() const { return drawView_; }

  bool draw_phi_0() const { return draw_phi0_; }
  bool draw_phi_1() const { return draw_phi1_; }
  double phi0() const { return phi0_; }
  double phi1() const { return phi1_; }
  bool draw_cyl() const { return draw_cyl_; }

private:
  int                   slices_;
  GLTexture3DHandle     tex_;

  //! GLVolRenStates in lieu of static variables in the state object
  //! this allows multiple volume renders to work..
  GLVolRenState        *state_;

  ROI                  *roi_;
  FullRes              *fr_;
  LOS                  *los_; 
  TexPlanes            *tp_;
 
  GLTexRenState        *tex_ren_state_;
 
  // GLTexRenStates done for the same reasons as above.
  GLOverOp             *oo_;
  GLMIP                *mip_;
  GLAttenuate          *atten_;
  GLPlanes             *planes_;
  Mutex                 mutex_;
  ColorMapHandle        cmap_;
  Point                 control_point_;
  
  double                slice_alpha_;

  bool                  cmap_has_changed_;
  bool                  drawX_;
  bool                  drawY_;
  bool                  drawZ_;
  bool                  drawView_;
  bool                  draw_phi0_;
  double                phi0_;
  bool                  draw_phi1_;
  double                phi1_;
  bool                  draw_cyl_;
  DrawInfoOpenGL       *di_;
 
  //! Sets the state function without having to write a bunch of code
  template <class T> T* state( T*& st, int l);
  
  bool                  interp_;
  int                   lighting_;
  static double         swap_matrix_[16];
  static int            r_count_;
  unsigned char         transfer_functions_[8][1024];
};

template <class T> 
T* GLVolumeRenderer::state( T*& st, int l)
{ 
  if(st == 0) 
    st = scinew T(this);
  lighting_ = l;
  return st;
}
 
} // End namespace SCIRun


#endif
