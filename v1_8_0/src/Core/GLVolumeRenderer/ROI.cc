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

#include <Core/GLVolumeRenderer/ROI.h>
#include <Core/Geometry/Ray.h>
#include <Core/GLVolumeRenderer/ROIIterator.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/SliceTable.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>

namespace SCIRun {


ROI::ROI(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}

void
ROI::setAlpha( const Brick& b)
{
  double alphaScale;
  if(b.level() == volren->tex()->depth())
    alphaScale = 1.0/pow(2.0, b.level());
  else
    alphaScale = 1.0/pow(2.0, b.level()*2);
  glColor4f(1,1,1, volren->slice_alpha() * alphaScale);
}

void
ROI::draw()
{
  Ray viewRay;
  Brick* brick;
  if( newbricks_ ){
    glDeleteTextures( textureNames.size(), &(textureNames[0]));
    textureNames.clear();
    newbricks_ = false;
  }

  computeView(viewRay);
  
  ROIIterator it( volren->tex().get_rep(), viewRay,  volren->control_point());

  BBox box;
  if( volren->tex()->has_slice_bounds() )
    volren->tex()->get_slice_bounds(box);
  else 
    volren->tex()->get_bounds(box);
  SliceTable st(box.min(),
		box.max(), 
		viewRay,
		volren->slices(),
                volren->tex()->depth());

  
  vector<Polygon* > polys;
  vector<Polygon* >::iterator pit;
  double tmin, tmax, dt;
  double ts[8];
  int i;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    for(pit = polys.begin(); pit != polys.end(); pit++) { delete *pit; }
    polys.clear();
    Brick& b = *brick;
    for( i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
    sortParameters(ts, 8);

    st.getParameters( b, tmin, tmax, dt);

    b.ComputePolys( viewRay,  tmin, tmax, dt, ts, polys);
    
    loadColorMap( b );
    loadTexture( b );
    makeTextureMatrix( b );
    enableTexCoords();
    //    setAlpha( b );
    drawPolys( polys );
    disableTexCoords();
  }
}

void 
ROI::drawWireFrame()
{
  Ray viewRay;
  computeView(viewRay);
  
  ROIIterator it( volren->tex().get_rep(), viewRay,  volren->control_point());

  Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // End namespace SCIRun
