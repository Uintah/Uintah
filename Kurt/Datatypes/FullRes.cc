#include "FullRes.h"
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Malloc/Allocator.h>
#include "FullResIterator.h"
#include "Brick.h"
#include "Polygon.h"
#include "SliceTable.h"
#include "GLVolumeRenderer.h"
#include "VolumeUtils.h"
#include <iostream>
namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Ray;
using Kurt::Datatypes::SliceTable;

FullRes::FullRes(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
FullRes::draw()
{
  //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
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
  int i,j, k;
  
  if( prev_view != 0 &&
      (prev_view->origin() != viewRay.origin() ||
       prev_view->direction() != viewRay.direction())){

    
    const GLTexture3D* tex =  volren->tex.get_rep();
    const Octree< Brick* >* node = tex->bontree;
    Brick *brick = (*node)();
    Brick& b = *brick;

    
    polys.clear();
    for( i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
    sortParameters(ts, 8);

    st.getParameters( b, tmin, tmax, dt);

    b.ComputePolys( viewRay,  tmin, tmax, dt, ts, polys);
    
    drawBrick( b, polys );
    *prev_view = viewRay;
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
  //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
}

void FullRes::drawBrick(Brick& b, const vector<Polygon *>& polys)
{
    loadColorMap( b );
    loadTexture( b );
    makeTextureMatrix( b );
    enableTexCoords();
    enableBlend();
    // setAlpha( b );
    drawPolys( polys );
    disableBlend();
    disableTexCoords();
}
void
FullRes::setAlpha( const Brick& b )
{
  //double sliceRatio = pow(2.0, volren->tex->depth() - b.level() - 1); 
  //double alpha = 1.0 - pow((1.0 - volren->slice_alpha), sliceRatio);
  glColor4f(1,1,1,1.0);
}

void 
FullRes::drawWireFrame()
{
  Ray viewRay;
  computeView( viewRay );
  
  FullResIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

  const Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // end namespace Datatypes
} // end namespace Kurt

