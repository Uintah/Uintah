#include <SCICore/Datatypes/LOS.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Datatypes/LOSIterator.h>
#include <SCICore/Datatypes/Brick.h>
#include <SCICore/Datatypes/Polygon.h>
#include <SCICore/Datatypes/SliceTable.h>
#include <SCICore/Datatypes/GLVolumeRenderer.h>
#include <SCICore/Datatypes/VolumeUtils.h>

namespace SCICore {
namespace GeomSpace  {

using SCICore::Geometry::Ray;


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
    //setAlpha( b );
    drawPolys( polys );
    disableTexCoords();
  }
}

void
LOS::setAlpha( const Brick& b )
{
  double alphaScale = 1.0/pow(2.0, b.level());
  glColor4f(1,1,1, volren->slice_alpha*alphaScale);
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

}  // namespace GeomSpace
} // namespace SCICore

