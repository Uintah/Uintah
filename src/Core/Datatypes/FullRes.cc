#include <SCICore/Datatypes/FullRes.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Datatypes/FullResIterator.h>
#include <SCICore/Datatypes/Brick.h>
#include <SCICore/Datatypes/Polygon.h>
#include <SCICore/Datatypes/SliceTable.h>
#include <SCICore/Datatypes/GLVolumeRenderer.h>
#include <SCICore/Datatypes/VolumeUtils.h>
#include <iostream>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Ray;
using SCICore::Datatypes::SliceTable;

FullRes::FullRes(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
FullRes::draw()
{
  //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
  Ray viewRay;
  Brick* brick;
  computeView(viewRay);
  
  FullResIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

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
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
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
    enableBlend();
    //setAlpha( b );
    drawPolys( polys );
    disableBlend();
    disableTexCoords();
    
  }
  //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
}

void
FullRes::setAlpha( const Brick& b )
{
  double alphaScale = 1.0/pow(2.0, b.level());
  glColor4f(1,1,1, volren->slice_alpha*alphaScale);
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
} // end namespace SCICore
