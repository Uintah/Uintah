/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <sci_gl.h>
#include <sci_glu.h>

#include <Core/GLVolumeRenderer/FullRes.h>
#include <Core/Geometry/Ray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GLVolumeRenderer/FullResIterator.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/SliceTable.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <iostream>

namespace SCIRun {

FullRes::FullRes(const GLVolumeRenderer* glvr ) :

  GLVolRenState( glvr )
{
}


void
FullRes::draw()
{
  CHECK_OPENGL_ERROR("FullRes::draw")
  Ray viewRay;
  Brick* brick;

  if( newbricks_ ){
    glDeleteTextures( textureNames.size(), &(textureNames[0]));
    textureNames.clear();
    newbricks_ = false;
  }

  computeView(viewRay);
  
  FullResIterator it( volren->tex().get_rep(), viewRay,  volren->control_point());

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
    enableBlend();
    makeTextureMatrix( b );
    enableTexCoords();

#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    if( !VolShader->created() ){
      VolShader->create();
    }
    VolShader->bind();
#endif
    drawPolys( polys );
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
     VolShader->release();
#endif
     disableTexCoords();
     disableBlend();    
  }
  reload_ = false;
}

void
FullRes::setAlpha( const Brick& b )
{
  double alphaScale = 1.0/pow(2.0, b.level());
  glColor4f(1,1,1, volren->slice_alpha() * alphaScale);
}

void 
FullRes::drawWireFrame()
{
  Ray viewRay;
  computeView( viewRay );
  
  FullResIterator it(volren->tex().get_rep(), viewRay, 
		     volren->control_point());

  const Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // End namespace SCIRun
