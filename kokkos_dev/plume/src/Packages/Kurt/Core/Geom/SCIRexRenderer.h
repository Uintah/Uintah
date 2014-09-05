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

#ifndef SCIREXRENDERER_H
#define SCIREXRENDERER_H

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Persistent/Persistent.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <ostream>
#include <string>

namespace SCIRun {
class Barrier;
// class GLVolumeRenderer;
}

namespace Kurt {
using SCIRun::Mutex;
using SCIRun::Point;
using SCIRun::Ray;
using SCIRun::Vector;
using SCIRun::BBox;
using SCIRun::Transform;
using SCIRun::ColorMapHandle;
using SCIRun::GLTexture3DHandle;
using SCIRun::GeomObj;
using SCIRun::Material;
using SCIRun::GeomSave;
using SCIRun::FieldHandle;
using SCIRun::DrawInfoOpenGL;
using SCIRun::Piostream;
using SCIRun::PersistentTypeID;
using SCIRun::Barrier;
using SCIRun::GLTexture3D;
using SCIRun::GLVolumeRenderer;

using std::vector;

class OGLXVisual;
class SCIRexWindow;
class SCIRexCompositer;
struct SCIRexRenderData;
class SCIRexRenderer : public GeomObj
{
public:
  enum renderStyle { OVEROP, MIP, ATTENUATE };
  
  SCIRexRenderer();
  
  SCIRexRenderer(vector<char*>& displays, int compositers,
		 FieldHandle tex, ColorMapHandle map,
		 bool isfixed, double min, double max,
		 GLTexture3DHandle texH = 0);
  
  void SetNSlices(int s) { //slices_ = s; cmapHasChanged_ = true;}
    slices_ = s;
    vector<GLVolumeRenderer *>::iterator it  = renderers.begin();
    for(; it != renderers.end(); it++)
      (*it)->SetNSlices(s);
  }
  void SetSliceAlpha( double as){ //slice_alpha_ = as; cmapHasChanged_ = true; }    
    slice_alpha_ = as;
    vector<GLVolumeRenderer *>::iterator it  = renderers.begin();
    for(; it != renderers.end(); it++)
      (*it)->SetSliceAlpha(as);
  }

    
  void SetInterp(bool b){
    interp_ = b;
    vector<GLVolumeRenderer *>::iterator it  = renderers.begin();
    for(; it != renderers.end(); it++)
      (*it)->SetInterp(b);
  }

  void getRange( double& min, double& max) { min = min_; max = max_;}
  void SetRange(const double& min, const double& max){ min_ = min; max_ = max;}
  
  void over_op(){ SetRenderStyle(OVEROP); rs_ = MIP; }
  void mip(){ SetRenderStyle(MIP); rs_ = MIP; }
  void attenuate(){ SetRenderStyle(ATTENUATE); rs_ = ATTENUATE; }
  void RangeIsFixed(bool b){ is_fixed_ = b;}

  int getBrickSize(){ return brick_size_; }
  void SetBrickSize( int bs ) { brick_size_ = bs; }
  
  void Build();

  void UpdateCompositers( int n );
  // if the SetVol and or SetBrickSize SetRange are called, Build must 
  // be called before any other operations.
  void SetVol( FieldHandle tex ){ this->tex_ = tex;}
  
  void SetColorMap( ColorMapHandle map){
     mutex_.lock();
    this->cmap_ = map;
    cmapHasChanged_ = true;
     mutex_.unlock();
    vector<GLVolumeRenderer *>::iterator it  = renderers.begin();
    for(; it != renderers.end(); it++)
      (*it)->SetColorMap(map);
  }
  
  void Reload() { NOT_FINISHED("SCIRexRenderer::Reload");}
  void DumpFrames(bool dump);
  void UseDepth(bool use_depth);
  SCIRexRenderer(const SCIRexRenderer&);
  ~SCIRexRenderer();
  
  void SetRenderStyle( renderStyle rs );
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const std::string& format, GeomSave*);
  
  void setup();
  virtual void preDraw();
  virtual void draw();
  virtual void postDraw();
  void cleanup();
  
  void drawWireFrame(){ glColor4f(0.8,0.8,0.8,1.0);
  }
  
  void update_compositer_data();
protected:
  renderStyle  rs_;
  
  FieldHandle tex_;
  int lighting_;
  int slices_;
 
private:
  
  Mutex mutex_;
  Mutex win_mutex_;
  ColorMapHandle cmap_;

  double  min_, max_;
  bool is_fixed_;
  bool interp_;
   
  int brick_size_;

  double slice_alpha_;
  DrawInfoOpenGL* di_;

  Transform field_transform;
  
  bool windows_init_; 
  bool compositers_init_;
  bool cmapHasChanged_;

  unsigned char transfer_functions_[8][1024];
  
  vector<SCIRexWindow *> windows;
  vector<SCIRexCompositer *> compositers;
  vector<GLTexture3DHandle> textures;
  vector<GLVolumeRenderer *> renderers;
  SCIRexRenderData *render_data_;

  void make_render_windows(FieldHandle tex_, 
		     double& min_, double& max_,
		     bool is_fixed_, int brick_size_,
		     vector<char *>& displays);


};

} // End namespace Kurt


#endif
