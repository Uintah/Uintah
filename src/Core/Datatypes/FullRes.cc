#include <Core/Datatypes/FullRes.h>
#include <Core/Geometry/Ray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FullResIterator.h>
#include <Core/Datatypes/Brick.h>
#include <Core/Datatypes/Polygon.h>
#include <Core/Datatypes/SliceTable.h>
#include <Core/Datatypes/GLVolumeRenderer.h>
#include <Core/Datatypes/VolumeUtils.h>
#include <iostream>

namespace SCIRun {


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

} // End namespace SCIRun
