/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/
#include <sci_defs.h>

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

#if defined(HAVE_GLEW)
#include <GL/glew.h>
#else
#include <GL/gl.h>
#include <sci_glu.h>
#endif

//  GLenum errCode;
//  const GLubyte *errString;

//  void glCheckForError(const char* message)
//  {
//    if((errCode = glGetError()) != GL_NO_ERROR){
//      errString = gluErrorString(errCode);
//      cerr<<"OpenGL Error: "<<message<<" "<<(const char*)errString<<endl;
//    }
//  }

FullRes::FullRes(const GLVolumeRenderer* glvr ) :

  GLVolRenState( glvr )
{
}


void
FullRes::draw()
{
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
