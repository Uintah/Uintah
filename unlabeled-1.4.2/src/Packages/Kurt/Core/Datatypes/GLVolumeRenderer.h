#ifndef GLVOLUMERENDERER_H
#define GLVOLUMERENDERER_H

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
public:

  GLVolumeRenderer(int id);

  GLVolumeRenderer(int id, GLTexture3DHandle tex,
		   ColorMapHandle map);

  void SetNSlices(int s) { slices = s; }
  void SetSliceAlpha( double as){ slice_alpha = as;}

  void SetVol( GLTexture3DHandle tex ){ this->tex = tex.get_rep(); }
  void SetColorMap( ColorMapHandle map){this->cmap = map->raw1d;
                                   cmapHasChanged = true;}
  void SetControlPoint( const Point& point){ controlPoint = point; }

  void Reload() { _state->Reload(); }

  void DrawFullRes(){ _state = FullRes::Instance(this);}
  void DrawLOS(){ _state = LOS::Instance(this);}
  void DrawROI(){ _state = ROI::Instance(this);}
  void DrawPlanes(){ _state = TexPlanes::Instance(this); }
  void SetX(bool b){ if(b){drawView = false;} drawX = b; }
  void SetY(bool b){ if(b){drawView = false;} drawY = b; }
  void SetZ(bool b){ if(b){drawView = false;} drawZ = b; }
  void SetView(bool b){ if(b){drawX=false; drawY=false; drawZ=false;}
                        drawView = b; }
  void GLOverOp(){ _gl_state = GLOverOp::Instance( this ); }
  void GLMIP(){ _gl_state = GLMIP::Instance( this ); }
  void GLAttenuate(){ _gl_state = GLAttenuate::Instance( this ); }
  void GLPlanes(){ _gl_state = GLPlanes::Instance(this);}

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
private:

  const GLTexture3D *tex;
  unsigned char* cmap;
  Point controlPoint;
  
  double slice_alpha;

  bool cmapHasChanged;
  bool drawX, drawY, drawZ, drawView;
  
  GLVolRenState* _state;
  GLTexRenState* _gl_state;
  static double swapMatrix[16];
  
};
}  // namespace GeomSpace
} // namespace SCICore


#endif
