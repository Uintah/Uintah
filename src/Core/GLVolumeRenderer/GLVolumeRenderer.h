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
#include <Core/Datatypes/ColorMap.h>
#include <Core/Geom/GeomObj.h>
#include <GL/gl.h>
#include <GL/glu.h>

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
  friend class GLVolRenState;
  friend class GLTexRenState;
  friend class GLAttenuate;
  friend class GLOverOp;
  friend class GLMIP;
  friend class GLPlanes;
  friend class FullRes;
  friend class ROI;
  friend class LOS;
  friend class TexPlanes;
private:
  GLVolRenState* _state;
  // GLVolRenStates in lieu of static variables in the state object
  // this allows multiple volume renders to work..

  ROI* _roi;
  FullRes* _fr;
  LOS* _los; 
  TexPlanes* _tp;
 
  GLTexRenState* _gl_state;
 
  // GLTexRenStates done for the same reasons as above.
  GLOverOp* _oo;
  GLMIP* _mip;
  GLAttenuate* _atten;
  GLPlanes* _planes;
public:

  GLVolumeRenderer(int id);

  GLVolumeRenderer(int id, GLTexture3DHandle tex,
		   ColorMapHandle map);

  void SetNSlices(int s) { slices = s;}
  void SetSliceAlpha( double as){ slice_alpha = as;}
  void BuildTransferFunctions();


  void SetVol( GLTexture3DHandle tex ){ 
    mutex.lock(); this->tex = tex; _state->NewBricks(); mutex.unlock();}
  void SetColorMap( ColorMapHandle map){
    mutex.lock(); this->cmap = map; BuildTransferFunctions();
    cmapHasChanged = true; mutex.unlock(); }
  void SetControlPoint( const Point& point){ controlPoint = point; }

  void Reload() { _state->Reload(); }

/*   void DrawFullRes(){ _state = FullRes::Instance(this);} */
/*   void DrawLOS(){ _state = LOS::Instance(this);} */
/*   void DrawROI(){ _state = ROI::Instance(this);} */
/*   void DrawPlanes(){ _state = TexPlanes::Instance(this); } */
  void DrawFullRes(){ _state = state(_fr, 0);}
  void DrawLOS(){ _state = state(_los, 0);}
  void DrawROI(){ _state = state(_roi, 0);}
  void DrawPlanes(){ _state = state(_tp, 1);}

  void SetX(bool b){ if(b){drawView = false;} drawX = b; }
  void SetY(bool b){ if(b){drawView = false;} drawY = b; }
  void SetZ(bool b){ if(b){drawView = false;} drawZ = b; }
  void SetView(bool b){ if(b){drawX=false; drawY=false; drawZ=false;}
                        drawView = b; }
/*   void GLOverOp(){ _gl_state = GLOverOp::Instance( this ); } */
/*   void GLMIP(){ _gl_state = GLMIP::Instance( this ); } */
/*   void GLAttenuate(){ _gl_state = GLAttenuate::Instance( this ); } */
/*   void GLPlanes(){ _gl_state = GLPlanes::Instance(this);} */
  void _GLOverOp(){ _gl_state = state(_oo, 0);}
  void _GLMIP(){ _gl_state = state(_mip, 0); }
  void _GLAttenuate(){ _gl_state = state(_atten, 0 ); }
  void _GLPlanes(){ _gl_state = state(_planes, 1);}

  void SetInterp( bool i) { _interp = i; }

  GLVolumeRenderer(const GLVolumeRenderer&);
  ~GLVolumeRenderer();


#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb){ tex->get_bounds( bb ); }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);

  void setup();
  void preDraw(){ _gl_state->preDraw(); }
  void draw(){ _state->draw(); }
  void postDraw(){ _gl_state->postDraw(); }
  void cleanup();

  void drawWireFrame(){ glColor4f(0.8,0.8,0.8,1.0); _state->drawWireFrame(); }

  GLTexture3DHandle get_tex3d_handle() const { return tex; }
protected:
  int slices;
  GLTexture3DHandle tex;

private:

  Mutex mutex;
  ColorMapHandle cmap;
  Point controlPoint;
  
  double slice_alpha;

  bool cmapHasChanged;
  bool drawX, drawY, drawZ, drawView;
  DrawInfoOpenGL* di_;
 

  // Sets the state function without having to write a bunch of code
  template <class T> T* state( T* st, int l);


  
  bool _interp;
  int _lighting;
  static double swapMatrix[16];
  static int rCount;
  unsigned char TransferFunctions[8][1024];
};

template <class T> 
T* GLVolumeRenderer::state( T* st, int l)
{ 
  if(st == 0) 
    st = scinew T(this);
  _lighting = l;
  return st;
}
 
} // End namespace SCIRun


#endif
