#include "FullRes.h"
#include <SCICore/Geometry/Ray.h>
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

GLVolRenState* FullRes::_instance = 0;

FullRes::FullRes(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
FullRes::draw()
{
  Ray viewRay;
  Brick* brick;
  computeView(viewRay);
  
  FullResIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

  SliceTable st(volren->tex->min(),
		volren->tex->max(), 
		viewRay,
		volren->slices);
  
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
    
    loadTexture( b );
    makeTextureMatrix( b );
    enableTexCoords();
    setAlpha( b );
    drawPolys( polys );
    disableTexCoords();

  }
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
GLVolRenState* 
FullRes::Instance(const GLVolumeRenderer* glvr)
{
  // Not a true Singleton class, but this does make sure that 
  // there is only one instance per volume renderer.
  if( _instance == 0 ){
    _instance = new FullRes( glvr );
  }
  
  return _instance;
}



} // end namespace Datatypes
} // end namespace Kurt

