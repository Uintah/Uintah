#include "FullRes.h"
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include "FullResIterator.h"
#include "Brick.h"
#include "Polygon.h"
#include "SliceTable.h"
#include "GLVolumeRenderer.h"
#include "VolumeUtils.h"
#include <iostream>

namespace Kurt {

using namespace SCIRun;
using namespace Uintah;

FullRes::FullRes(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
FullRes::draw()
{
  //AuditAllocator(default_allocator);
  static Ray* prev_view = 0;
  Ray viewRay;
  Brick* brick;
  computeView(viewRay);

  SliceTable st(volren->tex->min(),
		volren->tex->max(), 
		viewRay,
		volren->slices,
		volren->tex->depth());
  
  vector<Polygon* > polys;
  Point vertex;
  double tmin, tmax, dt;
  double ts[8];
  int i;
  
  if( prev_view != 0 &&
      (prev_view->origin() != viewRay.origin() ||
       prev_view->direction() != viewRay.direction())){

    LevelIterator it(volren->tex.get_rep(), viewRay, volren->controlPoint, 1);
    for( brick = it.Start(); !it.isDone(); brick = it.Next()){
      polys.clear();
      Brick& b = *brick;
      for( i = 0; i < 8; i++)
	ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
      sortParameters(ts, 8);
      
      st.getParameters( b, tmin, tmax, dt);
      
      b.ComputePolys( viewRay,  tmin, tmax, dt, ts, polys);
      
      drawBrick( b , polys);
      *prev_view = viewRay;
    }
  } else {
    FullResIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

    for( brick = it.Start(); !it.isDone(); brick = it.Next()){
      polys.clear();
      Brick& b = *brick;
      for( i = 0; i < 8; i++)
	ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
      sortParameters(ts, 8);
      
      st.getParameters( b, tmin, tmax, dt);
      
      b.ComputePolys( viewRay,  tmin, tmax, dt, ts, polys);
      
      drawBrick( b , polys);
      if( prev_view == 0 )
	prev_view = scinew Ray( viewRay );
    }
  }
  //AuditAllocator(default_allocator);
}

void FullRes::drawBrick(Brick& b, const vector<Polygon *>& polys)
{
    loadColorMap( b );
    loadTexture( b );
    makeTextureMatrix( b );
    enableTexCoords();
    enableBlend();
    setAlpha( b );
    drawPolys( polys );
    disableBlend();
    disableTexCoords();
}
void
FullRes::setAlpha( const Brick&  )
{
  glColor4f(1,1,1,1.0);
}

void 
FullRes::drawWireFrame()
{
  Ray viewRay;
  computeView( viewRay );
  static Ray* prev_view = 0;
  const Brick* brick;

  
  if( prev_view != 0 &&
      (prev_view->origin() != viewRay.origin() ||
       prev_view->direction() != viewRay.direction())){

    LevelIterator it(volren->tex.get_rep(), viewRay, volren->controlPoint, 1);
    for( brick = it.Start(); !it.isDone(); brick = it.Next()){
      GLVolRenState::drawWireFrame( *brick );
      *prev_view = viewRay;
    }
  } else {
    FullResIterator it( volren->tex.get_rep(), viewRay,
			volren->controlPoint);
    
    const Brick* brick;
    for( brick = it.Start(); !it.isDone(); brick = it.Next()){
      GLVolRenState::drawWireFrame( *brick );
     if( prev_view == 0 )
	prev_view = scinew Ray( viewRay );
    }
  }
}

} // End namespace Kurt


