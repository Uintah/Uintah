#ifndef GLVOLUMERENDERER_H
#define GLVOLUMERENDERER_H

#include <SCICore/Thread/Mutex.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <SCICore/Geom/GeomObj.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "SliceTable.h"
#include "Brick.h"
#include "Octree.h"

#include "GLTexRenState.h"
#include "GLVolRenState.h"
#include "FullRes.h"
#include "ROI.h"
#include "LOS.h"
#include "GLPlanes.h"
#include "TexPlanes.h"
#include "GLMIP.h"
#include "GLAttenuate.h"
#include "GLOverOp.h"
#include "GLTexture3D.h"

namespace SCICore {
namespace GeomSpace  {


using namespace SCICore::Geometry;
using namespace SCICore::Datatypes;
using namespace Kurt::Datatypes;
using SCICore::Thread::Mutex;

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
  TexPlanes* _tp;
  ROI* _roi;
  FullRes* _fr;
  LOS* _los;
  
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

  void SetNSlices(int s) { slices = s; }
  void SetSliceAlpha( double as){ slice_alpha = as;}

  void SetVol( GLTexture3DHandle tex ){ 
    mutex.lock(); this->tex = tex; mutex.unlock();}
  void SetColorMap( ColorMapHandle map){
    mutex.lock(); this->cmap = map; cmapHasChanged = true;
    mutex.unlock(); }
  void SetControlPoint( const Point& point){ controlPoint = point; }

  void Reload() { _state->Reload(); }

/*   void DrawFullRes(){ _state = FullRes::Instance(this);} */
/*   void DrawLOS(){ _state = LOS::Instance(this);} */
/*   void DrawROI(){ _state = ROI::Instance(this);} */
/*   void DrawPlanes(){ _state = TexPlanes::Instance(this); } */
  void DrawFullRes(){ _state = state(_fr);}
  void DrawLOS(){ _state = state(_los);}
  void DrawROI(){ _state = state(_roi);}
  void DrawPlanes(){ _state = state(_tp);}

  void SetX(bool b){ if(b){drawView = false;} drawX = b; }
  void SetY(bool b){ if(b){drawView = false;} drawY = b; }
  void SetZ(bool b){ if(b){drawView = false;} drawZ = b; }
  void SetView(bool b){ if(b){drawX=false; drawY=false; drawZ=false;}
                        drawView = b; }
/*   void GLOverOp(){ _gl_state = GLOverOp::Instance( this ); } */
/*   void GLMIP(){ _gl_state = GLMIP::Instance( this ); } */
/*   void GLAttenuate(){ _gl_state = GLAttenuate::Instance( this ); } */
/*   void GLPlanes(){ _gl_state = GLPlanes::Instance(this);} */
  void GLOverOp(){ _gl_state = state(_oo);}
  void GLMIP(){ _gl_state = state(_mip); }
  void GLAttenuate(){ _gl_state = state(_atten ); }
  void GLPlanes(){ _gl_state = state(_planes);}

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
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);

  void setup();
  void preDraw(){ _gl_state->preDraw(); }
  void draw(){ _state->draw(); }
  void postDraw(){ _gl_state->postDraw(); }
  void cleanup();

  void drawWireFrame(){ glColor4f(0.8,0.8,0.8,1.0); _state->drawWireFrame(); }

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
  
   
 

  // Sets the state function without having to write a bunch of code
  template <class T>
    T* state( T* st){ if(st == 0) st = new T(this); return st;}


  bool _interp;
  
  static double swapMatrix[16];
  
};


 
}  // namespace GeomSpace
} // namespace SCICore


#endif
