#include <Core/Datatypes/ROI.h>
#include <Core/Geometry/Ray.h>
#include <Core/Datatypes/ROIIterator.h>
#include <Core/Datatypes/Brick.h>
#include <Core/Datatypes/Polygon.h>
#include <Core/Datatypes/SliceTable.h>
#include <Core/Datatypes/GLVolumeRenderer.h>
#include <Core/Datatypes/VolumeUtils.h>

namespace SCIRun {


ROI::ROI(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}

void
ROI::setAlpha( const Brick& b)
{
  double alphaScale;
  if(b.level() == volren->tex->depth())
    alphaScale = 1.0/pow(2.0, b.level());
  else
    alphaScale = 1.0/pow(2.0, b.level()*2);
  glColor4f(1,1,1, volren->slice_alpha*alphaScale);
}

void
ROI::draw()
{
  Ray viewRay;
  Brick* brick;
  computeView(viewRay);

  
  ROIIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

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
ROI::drawWireFrame()
{
  Ray viewRay;
  computeView(viewRay);
  
  ROIIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

  Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // End namespace SCIRun
