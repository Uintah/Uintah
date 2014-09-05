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
#include <Core/Datatypes/Color.h>
#include <Core/Containers/Array1.h>
#include <GL/gl.h>
#include <iostream>
using std::cerr;

namespace Kurt {

using SCIRun::Color;
using SCIRun::Array1;
using SCIRun::FieldHandle;




VolumeRenderer::VolumeRenderer() 
  : rs_(VolumeRenderer::OVEROP),
    slices_(0),
    tex_(0),
    bg_(0),
    gvr_(0),
    lighting_(0),
    mutex("VolumeRenderer Mutex"),
    cmap(0),
    slice_alpha(1.0),
    di_(0),
    cmapHasChanged(true)
{
  NOT_FINISHED("VolumeRenderer::VolumeRenderer(int id, const Texture3D* tex, ColorMap* cmap)");
}


VolumeRenderer::VolumeRenderer(GridVolRen* gvr,
			       FieldHandle tex,
			       ColorMapHandle map,
			       bool fixed,
			       double min, double max)
  :
    rs_(VolumeRenderer::OVEROP),
    slices_(0),
    tex_(tex),
    bg_(0),
    gvr_(gvr),
    lighting_(0),
    mutex("VolumeRenderer Mutex"),
    cmap(map),
    brick_size_(128),
    slice_alpha(1.0),
    is_fixed_(fixed),
    min_val_(min),
    max_val_(max),
    di_(0),
    cmapHasChanged(true)
{
  mutex.lock();
  buildBrickGrid();
  mutex.unlock();
}

VolumeRenderer::VolumeRenderer(const VolumeRenderer& copy)
  : 
    rs_(copy.rs_),
    slices_(copy.slices_),
    tex_(copy.tex_),
    bg_(copy.bg_),
    gvr_(copy.gvr_),
    lighting_(copy.lighting_),
    mutex("VolumeRenderer Mutex"),
    cmap(copy.cmap),
    brick_size_(copy.brick_size_),
    slice_alpha(copy.slice_alpha),
    is_fixed_(copy.is_fixed_),
    min_val_(copy.min_val_),
    max_val_(copy.max_val_),
    di_(copy.di_),
    cmapHasChanged(copy.cmapHasChanged)
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

  bg_ = scinew BrickGrid( tex_, brick_size_,  is_fixed_, min_val_, max_val_);
  bg_->init();

}

void
VolumeRenderer::GetRange(double& min, double& max)
{
  if( bg_.get_rep() != 0){
    pair<double, double> range;
    bg_->get_range( range );
    min = range.first; max = range.second;
  } else {
    min = min_val_; max = max_val_;
  }
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


  glEnable(GL_TEXTURE_3D_EXT);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

  if( cmap.get_rep() ) {
#ifdef GL_TEXTURE_COLOR_TABLE_SGI
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
    glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);   
#endif
    if( cmapHasChanged ) {
      BuildTransferFunction();
      gvr_->SetColorMap(TransferFunction);
      cmapHasChanged = false;
    }
  }
  glColor4f(1,1,1,1); // set to all white for modulation
}

void
VolumeRenderer::preDraw()
{
  switch (rs_) {
  case OVEROP:
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD_EXT);
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


  if( cmap.get_rep() )
#ifdef GL_TEXTURE_COLOR_TABLE_SGI
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
  glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
#endif
  glDisable(GL_TEXTURE_3D_EXT);
  // glEnable(GL_DEPTH_TEST);  
}  


#define VOLUMERENDERER_VERSION 1

void VolumeRenderer::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("VolumeRenderer::io");
}


bool
VolumeRenderer::saveobj(std::ostream&, const std::string&, GeomSave*)
{
   NOT_FINISHED("VolumeRenderer::saveobj");
    return false;
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
    Color c = cmap->getColor(j*mul);
    alpha = cmap->getAlpha(j*mul);
    
    alpha1 = pow(alpha, bp);
    alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);

    TransferFunction[4*j + 0] = (unsigned char)(c.r()*255);
    TransferFunction[4*j + 1] = (unsigned char)(c.g()*255);
    TransferFunction[4*j + 2] = (unsigned char)(c.b()*255);
    TransferFunction[4*j + 3] = (unsigned char)(alpha2*255);
  }
}
} // End namespace Uintah
