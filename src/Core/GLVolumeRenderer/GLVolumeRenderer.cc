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

#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>

#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>

#include <iostream>

namespace SCIRun {

using std::cerr;


double GLVolumeRenderer::swap_matrix_[16] = { 0,0,1,0,
					      0,1,0,0,
					      1,0,0,0,
					      0,0,0,1};

int GLVolumeRenderer::r_count_ = 0;

GLVolumeRenderer::GLVolumeRenderer() : 
  GeomObj(),
  slices_(0),
  tex_(0),
  state_(state(fr_, 0)),
  roi_(0),
  fr_(0),
  los_(0),
  tp_(0),
  tex_ren_state_(state(oo_, 0)),
  oo_(0),
  mip_(0),
  atten_(0),
  planes_(0),
  mutex_("GLVolumeRenderer Mutex"),
  cmap_(0),
  control_point_(Point(0,0,0)),
  slice_alpha_(1.0),
  cmap_has_changed_(true),
  drawX_(false),
  drawY_(false),
  drawZ_(false),
  drawView_(false),
  interp_(true),
  lighting_(0),
  alphamult_(true)
{
  r_count_++;
  NOT_FINISHED("GLVolumeRenderer::GLVolumeRenderer(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLVolumeRenderer::GLVolumeRenderer(GLTexture3DHandle tex,
				   ColorMapHandle map) : 
  GeomObj(),
  slices_(0),
  tex_(tex),
  state_(state(fr_, 0)),
  roi_(0),
  fr_(0),
  los_(0),
  tp_(0),
  tex_ren_state_(state(oo_, 0)),
  oo_(0),
  mip_(0),
  atten_(0),
  planes_(0),
  mutex_("GLVolumeRenderer Mutex"),
  cmap_(map),
  control_point_(Point(0,0,0)),
  slice_alpha_(1.0),
  cmap_has_changed_(true),
  drawX_(false),
  drawY_(false),
  drawZ_(false),
  drawView_(false),
  di_(0),
  interp_(true),
  lighting_(0),
  alphamult_(true)
{
  r_count_++;
}

GLVolumeRenderer::GLVolumeRenderer(const GLVolumeRenderer& copy) : 
  GeomObj( copy ),
  slices_(copy.slices_),
  tex_(copy.tex_),
  state_(copy.state_),
  roi_(copy.roi_),
  fr_(copy.fr_),
  los_(copy.los_),
  tp_(copy.tp_),
  tex_ren_state_(copy.tex_ren_state_),
  oo_(copy.oo_),
  mip_(copy.mip_),
  atten_(copy.atten_),
  planes_(copy.planes_),
  mutex_("GLVolumeRenderer Mutex"),
  cmap_(copy.cmap_),
  control_point_(copy.control_point_),
  slice_alpha_(copy.slice_alpha_),
  cmap_has_changed_(copy.cmap_has_changed_),
  drawX_(copy.drawX_),
  drawY_(copy.drawY_),
  drawZ_(copy.drawZ_),
  drawView_(copy.drawView_),
  di_(copy.di_),
  interp_(copy.interp_),
  lighting_(copy.lighting_),
  alphamult_(copy.alphamult_)
{
  r_count_++;
} 

GLVolumeRenderer::~GLVolumeRenderer()
{

  //delete cmap_;
  //delete state_;
  //delete tex_ren_state_;
  delete tp_;
  delete roi_;
  delete fr_;
  delete los_;
  delete oo_;
  delete mip_;
  delete atten_;
  delete planes_;
  r_count_--;
}

GeomObj* 
GLVolumeRenderer::clone()
{
  return scinew GLVolumeRenderer( *this );
}


#ifdef SCI_OPENGL
void 
GLVolumeRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  CHECK_OPENGL_ERROR("GLVolumeRenderer::draw begin")
  //AuditAllocator(default_allocator);
  if( !pre_draw(di, mat, lighting_) ) return;
  mutex_.lock();
  di_ = di;
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    //AuditAllocator(default_allocator);
    setup();
    //AuditAllocator(default_allocator);
    preDraw();

    // calls state_->draw()
    BBox bb;
    tex_->get_bounds(bb);
    state_->set_bounding_box(bb);
    draw();
    //AuditAllocator(default_allocator);
    postDraw();
    //AuditAllocator(default_allocator);
    cleanup();
    //AuditAllocator(default_allocator);
  }
  di_ = 0;
  mutex_.unlock();
  CHECK_OPENGL_ERROR("GLVolumeRenderer::draw end")
}
#endif


void
GLVolumeRenderer::setup()
{
  CHECK_OPENGL_ERROR("GLVolumeRenderer::setup begin")
#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  glActiveTextureARB(GL_TEXTURE0_ARB);
#endif
  glEnable(GL_TEXTURE_3D);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glEnable(GL_TEXTURE_1D);
    glActiveTextureARB(GL_TEXTURE0_ARB);
#elif defined(GL_TEXTURE_COLOR_TABLE_SGI) && defined(__sgi)
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
    glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
#endif
  if( cmap_.get_rep() ) {
    if( cmap_has_changed_ || r_count_ != 1) {
      BuildTransferFunctions();
      cmap_has_changed_ = false;
    }
  }
  glColor4f(1,1,1,1); // set to all white for modulation
  glDepthMask(GL_FALSE);
  CHECK_OPENGL_ERROR("GLVolumeRenderer::setup end")
}

void
GLVolumeRenderer::cleanup()
{

  glDepthMask(GL_TRUE);
  if( cmap_.get_rep() )
#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glDisable(GL_TEXTURE_1D);
  glActiveTextureARB(GL_TEXTURE0_ARB);
#elif defined( GL_TEXTURE_COLOR_TABLE_SGI) && defined(__sgi)
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
  glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
#endif
  glDisable(GL_TEXTURE_3D);
  // glEnable(GL_DEPTH_TEST);  
}  


#define GLVOLUMERENDERER_VERSION 1

void GLVolumeRenderer::io(Piostream&)
{
  // Nothing for now...
  NOT_FINISHED("GLVolumeRenderer::io");
}


bool
GLVolumeRenderer::saveobj(std::ostream&, const string&, GeomSave*)
{
  NOT_FINISHED("GLVolumeRenderer::saveobj");
  return false;
}


void
GLVolumeRenderer::BuildTransferFunctions( )
{
  const int tSize = 256;
  int defaultSamples = 512;
  //  double L = (tex_->max() - tex_->min()).length();
  //  double dt = L/defaultSamples;
  float mul = 1.0/(tSize - 1);
  if( tex_->depth() > 8 ) {
    cerr<<"Error: Texture too deep\n";
    return;
  }

  double bp;

  if(alphamult_){
    bp = tan( 1.570796327*(0.5 - slice_alpha_*0.49999));
  } else {
    bp = tan( 1.570796327 * 0.5 );
  }
  for (int i = 0; i < tex_->depth() + 1; i++)
  {
    double sliceRatio =
      defaultSamples/(double(slices_)/pow(2.0, tex_->depth() - i));
    for ( int j = 0; j < tSize; j++ )
    {
      const Color c = cmap_->getColor(j*mul);
      const double alpha = cmap_->getAlpha(j*mul);

      const double alpha1 = pow(alpha, bp);
      const double alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);
  

      //	  if( j == 128 ) cerr <<" alpha = "<< alpha<<std::endl;
      //	  if( j == 128 ) cerr <<" alpha1 = "<< alpha1<<std::endl;
      //	  if( j == 128 ) cerr <<" c.r() = "<< c.r()<<std::endl;
      //	  if( j == 128 ) cerr <<" c.g() = "<< c.g()<<std::endl;
      //	  if( j == 128 ) cerr <<" c.b() = "<< c.b()<<std::endl;
      //	  if( j == 128 ) cerr <<" alpha2 = "<< alpha2<<std::endl;

      transfer_functions_[i][4*j + 0] = (unsigned char)(c.r()*alpha2*255);
      transfer_functions_[i][4*j + 1] = (unsigned char)(c.g()*alpha2*255);
      transfer_functions_[i][4*j + 2] = (unsigned char)(c.b()*alpha2*255);
      if( alphamult_ ) {
	transfer_functions_[i][4*j + 3] = (unsigned char)(alpha2*255);
      } else {
	transfer_functions_[i][4*j + 3] = (unsigned char)(alpha*255);
      }
    }
  }
}
} // End namespace SCIRun
