#include "LOS.h"
#include <Core/Geometry/Ray.h>
#include "LOSIterator.h"
#include "Brick.h"
#include "Polygon.h"
#include "SliceTable.h"
#include "GLVolumeRenderer.h"
#include "VolumeUtils.h"

namespace Kurt {

using namespace SCIRun;

LOS::LOS(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
LOS::draw()
{
  Ray viewRay;
  Brick* brick;
  computeView(viewRay);
  
  LOSIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

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
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    polys.clear();
    Brick& b = *brick;
    for( i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
    sortParameters(ts, 8);

    st.getParameters( b, tmin, tmax, dt);

    b.ComputePolys( viewRay,  tmin, tmax, dt, ts, polys);
    
    drawBrick( b, polys );
  }
}

void LOS::drawBrick( Brick& b, const vector<Polygon *>& polys)
{
    loadTexture( b );
    makeTextureMatrix( b );
    enableTexCoords();
    setAlpha( b );
    drawPolys( polys );
    disableTexCoords();
}

void
LOS::setAlpha( const Brick& )
{
  glColor4f(1,1,1, volren->scale_alpha);
}

void 
LOS::drawWireFrame()
{
  Ray viewRay;

  computeView(viewRay);
  
  LOSIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

  Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // End namespace Kurt
