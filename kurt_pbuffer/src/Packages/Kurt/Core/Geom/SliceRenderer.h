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

#include <Packages/Kurt/Core/Geom/VolumeRenderer.h>

#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geometry/Point.h>
#include <iostream>

namespace Kurt {
using std::cerr;
using SCIRun::ColorMapHandle;
using SCIRun::FieldHandle;
using SCIRun::Point;



class SliceRenderer : public VolumeRenderer
{
public:

  SliceRenderer();

  SliceRenderer(GridVolRen* gvr, FieldHandle tex,
		 ColorMapHandle map, 
		 bool fixed, double min, double max);
  SliceRenderer(const SliceRenderer&);
  ~SliceRenderer();

  void SetControlPoint( const Point& point){ gvr_->SetControlPoint( point); }
  void SetX(bool b){gvr_->SetX(b); }
  void SetY(bool b){gvr_->SetY(b); }
  void SetZ(bool b){gvr_->SetZ(b); }
  void SetView(bool b){ gvr_->SetView(b); }

  virtual void preDraw(){ 
    cerr<<"in SliceRenderer preDraw\n";
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0);
  }

  virtual void draw(){ gvr_->draw(*(bg_.get_rep()), slices_); }
  virtual void postDraw(){ glDisable(GL_ALPHA_TEST); }


protected:

};

} // End namespace Kurt


#endif
