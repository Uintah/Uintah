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

#include <Packages/Kurt/Core/Geom/VolumeRenderer.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Datatypes/Field.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Color.h>
#include <Core/Containers/Array1.h>
#include <GL/gl.h>
#include <iostream>
using std::cerr;

namespace Kurt {

using SCIRun::Color;
using SCIRun::Array1;
using SCIRun::FieldHandle;




VolumeRenderer::VolumeRenderer(int id) 
  : GeomObj( id ),
    rs_(VolumeRenderer::OVEROP),
    slices_(0),
    tex_(0),
    bg_(0),
    mutex("VolumeRenderer Mutex"),
    cmap(0),
    slice_alpha(1.0),
    cmapHasChanged(true),
    di_(0),
    lighting_(0)
{
  NOT_FINISHED("VolumeRenderer::VolumeRenderer(int id, const Texture3D* tex, ColorMap* cmap)");
}


VolumeRenderer::VolumeRenderer(int id, 
			       FieldHandle tex,
			       ColorMapHandle map)
  : GeomObj( id ),
    rs_(VolumeRenderer::OVEROP),
    slices_(0),
    tex_(tex),
    bg_(0),
    mutex("VolumeRenderer Mutex"),
    cmap(map),
    slice_alpha(1.0),
    cmapHasChanged(true),
    di_(0),
    lighting_(0)
{
  mutex.lock();
  buildBrickGrid();
  mutex.unlock();
}

VolumeRenderer::VolumeRenderer(const VolumeRenderer& copy)
  : GeomObj( copy.id ),
    rs_(copy.rs_),
    slices_(copy.slices_),
    tex_(copy.tex_),
    bg_(copy.bg_),
    mutex("VolumeRenderer Mutex"),
    cmap(copy.cmap),
    slice_alpha(copy.slice_alpha),
    cmapHasChanged(copy.cmapHasChanged),
    di_(copy.di_),
    lighting_(copy.lighting_)
{
  mutex.lock();
  buildBrickGrid();
  mutex.unlock();
} 

VolumeRenderer::~VolumeRenderer()
{
}

GeomObj* 
VolumeRenderer::clone()
{
  return scinew VolumeRenderer( *this );
}

void
VolumeRenderer::buildBrickGrid()
{
  bg_ = new BrickGrid( tex_, 64 );
  bg_->init();
}


#ifdef SCI_OPENGL
void 
VolumeRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
    //AuditAllocator(default_allocator);
  if( !pre_draw(di, mat, lighting_) ) return;
  mutex.lock();
  di_ = di;
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    //AuditAllocator(default_allocator);
    setup();
    preDraw();
    draw();
    postDraw();
    cleanup();
  }
  di_ = 0;
  mutex.unlock();

}
#endif


void
VolumeRenderer::setup()
{


#ifdef __sgi
  glEnable(GL_TEXTURE_3D_EXT);
#endif
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

  if( cmap.get_rep() ) {
#ifdef __sgi
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    
    if( cmapHasChanged ) {
      BuildTransferFunction();
      gvr_.SetColorMap(TransferFunction);
      cmapHasChanged = false;
    }
#endif
  }
  glColor4f(1,1,1,1); // set to all white for modulation
}

void
VolumeRenderer::preDraw()
{
  switch (rs_) {
  case OVEROP:
    glEnable(GL_BLEND);
#ifdef __sgi
    glBlendEquation(GL_FUNC_ADD_EXT);
#endif
    glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    break;
  case MIP:
    glEnable(GL_BLEND);
    glBlendEquation(GL_MAX_EXT);
    glBlendFunc(GL_ONE, GL_ONE);
    break;
  case ATTENUATE:
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD_EXT);
    glBlendFunc(GL_CONSTANT_ALPHA_EXT, GL_ONE);
    glBlendColor(1.f, 1.f, 1.f, 1.f/slices_);
    break;
  default:
    cerr<<"Shouldn't be here!\n";
    ASSERT(0);
  }
}

void
VolumeRenderer::postDraw()
{
  switch (rs_) {
  case OVEROP:
  case MIP:
  case ATTENUATE:
    glDisable(GL_BLEND);
    break;
  default:
    cerr<<"Shouldn't be here!\n";
    ASSERT(0);
  }
}

void
VolumeRenderer::cleanup()
{


#ifdef __sgi
  if( cmap.get_rep() )
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
  glDisable(GL_TEXTURE_3D_EXT);
#endif
  // glEnable(GL_DEPTH_TEST);  
}  


#define VOLUMERENDERER_VERSION 1

void VolumeRenderer::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("VolumeRenderer::io");
}


bool
VolumeRenderer::saveobj(std::ostream&, const string&, GeomSave*)
{
   NOT_FINISHED("VolumeRenderer::saveobj");
    return false;
}
inline Color FindColor(const Array1<Color>& c,const Array1<float>& s,float t)
{
  int j=0;

  if (t<=s[0])
    return c[0];
  if (t>= s[s.size()-1])
    return c[c.size()-1];

  // t is within the interval...

  while((j < c.size()) && (t > s[j])) {
    j++;
  }

  double slop = (s[j] - t)/(s[j]-s[j-1]);

  return c[j-1]*slop + c[j]*(1.0-slop);
  
}

inline double FindAlpha(const Array1<float>& c,const Array1<float>& s,float t)
{
  int j=0;

  if (t<=s[0])
    return c[0];
  if (t>= s[s.size()-1])
    return c[c.size()-1];

  // t is within the interval...

  while((j < c.size()) && (t > s[j])) {
    j++;
  }

  float slop = (s[j] - t)/(s[j]-s[j-1]);

  return c[j-1]*slop + c[j]*(1.0-slop);
}

void
VolumeRenderer::BuildTransferFunction( )
{
  const int tSize = 256;
  int defaultSamples = 512;
  //  double L = (tex->max() - tex->min()).length();
  //  double dt = L/defaultSamples;
  float mul = 1.0/(tSize - 1);

  double bp = tan( 1.570796327*(0.5 - slice_alpha*0.49999));
  double sliceRatio = defaultSamples/(double(slices_));
  double alpha, alpha1, alpha2;
  for( int j = 0; j < tSize; j++ )
  {
    Color c = FindColor(cmap->rawRampColor,
			cmap->rawRampColorT, j*mul);
    alpha = FindAlpha(cmap->rawRampAlpha,
		      cmap->rawRampAlphaT, j*mul);
    
    
    alpha1 = pow(alpha, bp);
    alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);

    TransferFunction[4*j + 0] = (unsigned char)(c.r()*255);
    TransferFunction[4*j + 1] = (unsigned char)(c.g()*255);
    TransferFunction[4*j + 2] = (unsigned char)(c.b()*255);
    TransferFunction[4*j + 3] = (unsigned char)(alpha2*255);
  }
}
} // End namespace Uintah
