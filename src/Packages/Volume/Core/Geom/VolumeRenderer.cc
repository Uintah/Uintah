#include <sci_defs.h>

#if defined(HAVE_GLEW)
#include <GL/glew.h>
#else
#include <GL/gl.h>
#endif

#include <Core/Geom/GeomOpenGL.h>

#include <Packages/Volume/Core/Geom/VolumeRenderer.h>
#include <Packages/Volume/Core/Geom/FragmentProgramARB.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Util/SliceTable.h>

//ATTRIB fc = fragment.color; \n\
//MUL_SAT result.color, c, fc;
static const char* ShaderString =
"!!ARBfp1.0 \n\
TEMP c0, c; \n\
ATTRIB tf = fragment.texcoord[0]; \n\
TEX c0, tf, texture[0], 3D; \n\
TEX c, c0, texture[1], 1D; \n\
MOV_SAT result.color, c; \n\
END";

//fogParam = {density, start, end, 1/(end-start) 
//fogCoord = z
static const char* FogShaderString =
"!!ARBfp1.0 \n\
TEMP c0, c, fogFactor, finalColor; \n\
PARAM fogColor = state.fog.color; \n\
PARAM fogParam = state.fog.params; \n\
ATTRIB fogCoord = fragment.fogcoord; \n\
SUB c, fogParam.z, fogCoord.x; \n\
MUL_SAT fogFactor.x, c, fogParam.w; \n\
ATTRIB tf = fragment.texcoord[0]; \n\
TEX c0, tf, texture[0], 3D; \n\
TEX finalColor, c0, texture[1], 1D; \n\
LRP result.color.rgb, fogFactor.x, finalColor, fogColor; \n\
END";


using namespace Volume;
using SCIRun::DrawInfoOpenGL;

VolumeRenderer::VolumeRenderer() :
  slices_(64),
  slice_alpha_(0.5),
  mode_( OVEROP )
{
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = new FragmentProgramARB( ShaderString, false );
  FogVolShader = new FragmentProgramARB( FogShaderString, false );
#endif
}

VolumeRenderer::VolumeRenderer(TextureHandle tex, ColorMapHandle map):
  TextureRenderer(tex, map),
  slices_(64),
  slice_alpha_(0.5),
  mode_(OVEROP)
{
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = new FragmentProgramARB( ShaderString, false );
  FogVolShader = new FragmentProgramARB( FogShaderString, false );
#endif
}

VolumeRenderer::VolumeRenderer( const VolumeRenderer& copy):
  TextureRenderer(copy.tex_, copy.cmap_),
  slices_(copy.slices_),
  slice_alpha_(copy.slice_alpha_),
  mode_(copy.mode_)
{
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = copy.VolShader;
  FogVolShader = new FragmentProgramARB( FogShaderString, false );
#endif
}

GeomObj*
VolumeRenderer::clone()
{
  return scinew VolumeRenderer(*this);
}

#ifdef SCI_OPENGL
void
VolumeRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
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
    draw();
    //AuditAllocator(default_allocator);
    cleanup();
    //AuditAllocator(default_allocator);
  }
  di_ = 0;
  mutex_.unlock();
}


void
VolumeRenderer::setup()
{
  // First set up the Textures.
#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  glActiveTextureARB(GL_TEXTURE0_ARB);
#endif
  glEnable(GL_TEXTURE_3D);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glEnable(GL_TEXTURE_1D);
    glActiveTextureARB(GL_TEXTURE0_ARB);
#elif defined(GL_TEXTURE_COLOR_TABLE_SGI)
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
    glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
#endif
  if( cmap_.get_rep() ) {
    if( cmap_has_changed_ || r_count_ != 1) {
      BuildTransferFunction();
      // cmap_has_changed_ = false;
    }
  }
  glColor4f(1,1,1,1);  //set to all white for modulation
  glDepthMask(GL_FALSE);
  
  // now set up the Blending
  switch ( mode_ ) {
  case OVEROP:
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    break;
  case MIP:
    glBlendEquation(GL_MAX);
    glBlendFunc(GL_ONE, GL_ONE);
    break;
  case ATTENUATE:
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc( GL_ONE, GL_ONE_MINUS_CONSTANT_ALPHA);
    glBlendColor(1.0,1.0,1.0,1.0/slices_);
    break;
  }
}

void
VolumeRenderer::cleanup()
{

  glDepthMask(GL_TRUE);
  if( cmap_.get_rep() ){
#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glDisable(GL_TEXTURE_1D);
    glActiveTextureARB(GL_TEXTURE0_ARB);
#elif defined( GL_TEXTURE_COLOR_TABLE_SGI)
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
    glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
#endif
  }
  glDisable(GL_TEXTURE_3D);
  // glEnable(GL_DEPTH_TEST);  
}

void
VolumeRenderer::draw()
{

  Ray viewRay;
  compute_view( viewRay );

  vector<Brick*> bricks;
  tex_->get_sorted_bricks( bricks, viewRay );

  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();

  BBox brickbounds;
  tex_->get_bounds( brickbounds );

  SliceTable st( brickbounds.min(), brickbounds.max(), viewRay, slices_);
  
  vector<Polygon* > polys;
  vector<Polygon* >::iterator pit;
  double tmin, tmax, dt;
  double ts[8];
  Brick *brick;
  
  for( ; it != it_end; it++ ){
    for( pit = polys.begin(); pit != polys.end(); pit++){ delete *pit; }
    polys.clear();
    Brick& b = *(*it);
    for( int i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
    sortParameters(ts, 8);

    st.getParameters( b, tmin, tmax, dt);

    b.ComputePolys( viewRay,  tmin, tmax, dt, ts, polys);

    load_colormap();
    load_texture( b );
    glEnable(GL_BLEND);
//     make_texture_matrix( b );
//     enable_tex_coords();

#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    GLboolean fog;
    glGetBooleanv(GL_FOG, &fog);
    if( fog ){
      if( !FogVolShader->created() ){
	FogVolShader->create();
      }
      FogVolShader->bind();
    } else {
      if( !VolShader->created() ){
	VolShader->create();
      }
      VolShader->bind();
    }
#endif

    drawPolys( polys );

#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    if( fog )
      FogVolShader->release();
    else
      VolShader->release();
#endif

//      disable_tex_coords();
     glDisable(GL_BLEND);
  }
  reload_ = false;
}

void
VolumeRenderer::drawWireFrame()
{
  Ray viewRay;
  compute_view( viewRay );

  double mvmat[16];
  TextureHandle tex = tex_;
  Transform field_trans = tex_->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  field_trans.get_trans(mvmat);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  glEnable(GL_DEPTH_TEST);


  vector<Brick*> bricks;
  tex_->get_sorted_bricks( bricks, viewRay );
  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();
  for(; it != it_end; ++it){
    Brick& brick = *(*it);
    glColor4f(0.8,0.8,0.8,1.0);

    glBegin(GL_LINES);
    for(int i = 0; i < 4; i++){
      glVertex3d(brick[i].x(), brick[i].y(), brick[i].z());
      glVertex3d(brick[i+4].x(), brick[i+4].y(), brick[i+4].z());
    }
    glEnd();

    glBegin(GL_LINE_LOOP);
    glVertex3d(brick[0].x(), brick[0].y(), brick[0].z());
    glVertex3d(brick[1].x(), brick[1].y(), brick[1].z());
    glVertex3d(brick[3].x(), brick[3].y(), brick[3].z());
    glVertex3d(brick[2].x(), brick[2].y(), brick[2].z());
    glEnd();

    glBegin(GL_LINE_LOOP);
    glVertex3d(brick[4].x(), brick[4].y(), brick[4].z());
    glVertex3d(brick[5].x(), brick[5].y(), brick[5].z());
    glVertex3d(brick[7].x(), brick[7].y(), brick[7].z());
    glVertex3d(brick[6].x(), brick[6].y(), brick[6].z());
    glEnd();

  }
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void 
VolumeRenderer::load_colormap()
{
  const unsigned char *arr = transfer_function_;

#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture)  && defined(__APPLE__)

  glActiveTextureARB(GL_TEXTURE1_ARB);
  {
    glEnable(GL_TEXTURE_1D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if( cmap_texture_ == 0 || cmap_has_changed_ ){
      glDeleteTextures(1, &cmap_texture_);
      glGenTextures(1, &cmap_texture_);
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);      
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage1D(GL_TEXTURE_1D, 0,
		   GL_RGBA,
		   256, 0,
		   GL_RGBA, GL_UNSIGNED_BYTE,
		   arr);
      cmap_has_changed_ = false;
    } else {
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
    }
    glActiveTexture(GL_TEXTURE0_ARB);
  }
#elif defined( GL_TEXTURE_COLOR_TABLE_SGI )
  //  cerr<<"GL_TEXTURE_COLOR_TABLE_SGI defined\n";
  glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
               GL_RGBA,
               256, // try larger sizes?
               GL_RGBA,  // need an alpha value...
               GL_UNSIGNED_BYTE, // try shorts...
               arr);
#elif defined( GL_SHARED_TEXTURE_PALETTE_EXT )
  //  cerr<<"GL_SHARED_TEXTURE_PALETTE_EXT  defined \n";

#ifndef HAVE_CHROMIUM
    ASSERT(glColorTableEXT != NULL );
  glColorTableEXT(GL_SHARED_TEXTURE_PALETTE_EXT,
		  GL_RGBA,
		  256, // try larger sizes?
		  GL_RGBA,  // need an alpha value...
		  GL_UNSIGNED_BYTE, // try shorts...
		  arr);
#endif
  //   glCheckForError("After glColorTableEXT");
#endif
}

#endif
void
VolumeRenderer::BuildTransferFunction()
{
  const int tSize = 256;
  int defaultSamples = 512;
  float mul = 1.0/(tSize - 1);
  double bp = 0;
  if( mode_ != MIP) {
    bp = tan( 1.570796327 * (0.5 - slice_alpha_*0.49999));
  } else {
    bp = tan( 1.570796327 * 0.5 );
  }
  double sliceRatio =  defaultSamples/(double(slices_));
  for ( int j = 0; j < tSize; j++ )
  {
    const Color c = cmap_->getColor(j*mul);
    const double alpha = cmap_->getAlpha(j*mul);
    
    const double alpha1 = pow(alpha, bp);
    const double alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);
    transfer_function_[4*j + 0] = (unsigned char)(c.r()*alpha2*255);
    transfer_function_[4*j + 1] = (unsigned char)(c.g()*alpha2*255);
    transfer_function_[4*j + 2] = (unsigned char)(c.b()*alpha2*255);
    transfer_function_[4*j + 3] = (unsigned char)(alpha2*255);
  }
}




