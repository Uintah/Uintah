#include "ROI.h"
#include <SCICore/Geometry/Ray.h>
#include "ROIIterator.h"
#include "Brick.h"
#include "Polygon.h"
#include "SliceTable.h"
#include "GLVolumeRenderer.h"
#include "VolumeUtils.h"

namespace SCICore {
namespace GeomSpace  {

using SCICore::Geometry::Ray;

ROI::ROI(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}

void
ROI::setAlpha( const Brick& b)
{
  if(b.level() == volren->tex->depth()){
    double sliceRatio = pow(2.0, volren->tex->depth() - b.level() - 1); 
    double alpha = 1.0 - pow((1.0 - volren->slice_alpha), sliceRatio);
    glColor4f(1,1,1, alpha);
  } else {
    double alphaScale = pow(2.0, b.level()*2);
    glColor4f(1,1,1, volren->slice_alpha*alphaScale);
  }

//   We want to lower the alpha of surrounding bricks so that the 
//    region of influence can be seen.
//   double alphaScale;
//   if(b.level() == volren->tex->depth())
//     alphaScale = 1.0/pow(2.0, b.level());
//   else
//     alphaScale = 1.0/pow(2.0, b.level()*2);
//   glColor4f(1,1,1, volren->slice_alpha*alphaScale);
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
    
    drawBrick( b, polys);
  }
}

void ROI::drawBrick( Brick& b, const vector<Polygon *>& polys)
{
    loadTexture( b );
    makeTextureMatrix( b );
    enableTexCoords();
    //setAlpha( b );
    drawPolys( polys );
    disableTexCoords();
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

}  // namespace GeomSpace
} // namespace SCICore
