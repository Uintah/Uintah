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

#ifndef VOLUMERENDERER_H
#define VOLUMERENDERER_H

#include <Packages/Kurt/Core/Geom/BrickGrid.h>
#include <Packages/Kurt/Core/Geom/GridVolRen.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/ColorMap.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Persistent/Persistent.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <Core/Datatypes/Brick.h>

namespace Kurt {

using SCIRun::Mutex;
using SCIRun::Point;
using SCIRun::Ray;
using SCIRun::Vector;
using SCIRun::BBox;
using SCIRun::Transform;
using SCIRun::ColorMapHandle;
using SCIRun::GeomObj;
using SCIRun::Material;
using SCIRun::GeomSave;
using SCIRun::FieldHandle;
using SCIRun::DrawInfoOpenGL;
using SCIRun::Piostream;
using SCIRun::PersistentTypeID;


class VolumeRenderer : public GeomObj
{
public:
  enum renderStyle { OVEROP, MIP, ATTENUATE };

  VolumeRenderer(int id);

  VolumeRenderer(int id, FieldHandle tex,
		   ColorMapHandle map);

  void SetNSlices(int s) { slices_ = s; cmapHasChanged = true;}
  void SetSliceAlpha( double as){ slice_alpha = as; cmapHasChanged = true; }
  void BuildTransferFunction();
  
  void over_op(){ rs_ = OVEROP; }
  void mip(){ rs_ = MIP; }
  void attenuate(){ rs_ = ATTENUATE; }


  void SetVol( FieldHandle tex ){ 
    mutex.lock(); this->tex_ = tex; buildBrickGrid(); mutex.unlock();}
  void SetColorMap( ColorMapHandle map){
    mutex.lock(); this->cmap = map;   cmapHasChanged = true;  mutex.unlock();}
  void Reload() { gvr_.Reload(); }

  void SetInterp( bool i) { gvr_.SetInterp( i ); }

  VolumeRenderer(const VolumeRenderer&);
  ~VolumeRenderer();


#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb){ bg_->get_bounds( bb ); }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);

  void setup();
  void preDraw();
  void draw(){ gvr_.draw(*(bg_.get_rep()), slices_); }
  void postDraw();
  void cleanup();

  void drawWireFrame(){ glColor4f(0.8,0.8,0.8,1.0);
  gvr_.drawWireFrame(*(bg_.get_rep())); }
  void buildBrickGrid();

protected:
  renderStyle  rs_;

  int slices_;
  FieldHandle tex_;
  BrickGridHandle bg_;
  GridVolRen gvr_;
private:

  Mutex mutex;
  ColorMapHandle cmap;
  
  double slice_alpha;

  DrawInfoOpenGL* di_;
 
  
  int lighting_;

  bool cmapHasChanged;
  unsigned char TransferFunction[1024];
};

} // End namespace Kurt


#endif
