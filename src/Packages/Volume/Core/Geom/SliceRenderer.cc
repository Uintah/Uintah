#include <sci_defs.h>

#if defined(HAVE_GLEW)
#include <GL/glew.h>
#else
#include <GL/gl.h>
#endif

#include <Core/Geom/GeomOpenGL.h>

#include <Packages/Volume/Core/Geom/SliceRenderer.h>
#include <Packages/Volume/Core/Geom/FragmentProgramARB.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Util/SliceTable.h>

static const char* ShaderString =
"!!ARBfp1.0 \n\
TEMP c0, c; \n\
ATTRIB fc = fragment.color; \n\
ATTRIB tf = fragment.texcoord[0]; \n\
TEX c0, tf, texture[0], 3D; \n\
TEX c, c0, texture[1], 1D; \n\
MUL c, c, fc; \n\
MOV_SAT result.color, c; \n\
END";

using namespace Volume;
using SCIRun::DrawInfoOpenGL;

SliceRenderer::SliceRenderer():
  control_point_( Point(0,0,0) ),
  drawX_(false),
  drawY_(false),
  drawZ_(false),
  draw_view_(false),
  draw_phi0_(false),
  phi0_(0),
  draw_phi1_(false),
  phi1_(0),
  draw_cyl_(false)
{
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = new FragmentProgramARB( ShaderString, false );
#endif
  lighting_ = 1;
}

SliceRenderer::SliceRenderer(TextureHandle tex, ColorMapHandle map):
  TextureRenderer(tex, map),
  control_point_( Point(0,0,0) ),
  drawX_(false),
  drawY_(false),
  drawZ_(false),
  draw_view_(false),
  draw_phi0_(false),
  phi0_(0),
  draw_phi1_(false),
  phi1_(0),
  draw_cyl_(false)
{
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = new FragmentProgramARB( ShaderString, false );
#endif
  lighting_ = 1;
}

SliceRenderer::SliceRenderer( const SliceRenderer& copy ) :
  TextureRenderer(copy.tex_, copy.cmap_),
  control_point_( copy.control_point_),
  drawX_(copy.drawX_),
  drawY_(copy.drawY_),
  drawZ_(copy.drawX_),
  draw_view_(copy.draw_view_),
  draw_phi0_(copy.phi0_),
  phi0_(copy.phi0_),
  draw_phi1_(copy.draw_phi1_),
  phi1_(copy.phi1_),
  draw_cyl_(copy.draw_cyl_)
{
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = copy.VolShader;
#endif
  lighting_ = 1;
}

SliceRenderer::~SliceRenderer()
{}

GeomObj*
SliceRenderer::clone()
{
  return scinew SliceRenderer(*this);
}

#ifdef SCI_OPENGL
void
SliceRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
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
SliceRenderer::setup()
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

  glDepthMask(GL_TRUE);
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);
}

void
SliceRenderer::cleanup()
{
  glDisable(GL_ALPHA_TEST);
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
SliceRenderer::draw()
{

  Ray viewRay;
  compute_view( viewRay );

  vector<Brick*> bricks;
  tex_->get_sorted_bricks( bricks, viewRay );

  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();

  BBox brickbounds;
  tex_->get_bounds( brickbounds );

  Polygon*  poly;
  BBox box;
  double t;
  for( ; it != it_end; it++ ){
    Brick& b = *(*it);

    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(control_point_);
    bool draw_z = false;

    if (draw_cyl_) {
      const double to_rad = M_PI / 180.0;
      BBox bb;
      tex_->get_bounds( bb );
      Point cyl_mid = bb.min() + bb.diagonal() * 0.5;
      if(draw_phi0_) {
	Vector phi(1.,0,0);
	
	Transform rot;
	rot.pre_rotate(phi0_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
      
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
      if(draw_phi1_) {
	Vector phi(1.,0,0);
	
	Transform rot;
	rot.pre_rotate(phi1_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
      
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
      if(drawZ_){
	draw_z = true;
      }

    } else {

      if(draw_view_){
	t = intersectParam(-viewRay.direction(), control_point_, viewRay);
	b.ComputePoly(viewRay, t, poly);
	draw(b, poly);
      } else {
      
	if(drawX_){
	  Point o(b[0].x(), mid.y(), mid.z());
	  Vector v(c.x() - o.x(), 0,0);
	  if(c.x() > b[0].x() && c.x() < b[7].x() ){
	    if( viewPt.x() > c.x() ){
	      o.x(b[7].x());
	      v.x(c.x() - o.x());
	    } 
	    Ray r(o,v);
	    t = intersectParam(-r.direction(), control_point_, r);
	    b.ComputePoly( r, t, poly);
	    draw( b, poly );
	  }
	}
	if(drawY_){
	  Point o(mid.x(), b[0].y(), mid.z());
	  Vector v(0, c.y() - o.y(), 0);
	  if(c.y() > b[0].y() && c.y() < b[7].y() ){
	    if( viewPt.y() > c.y() ){
	      o.y(b[7].y());
	      v.y(c.y() - o.y());
	    } 
	    Ray r(o,v);
	    t = intersectParam(-r.direction(), control_point_, r);
	    b.ComputePoly( r, t, poly);
	    draw( b, poly );
	  }
	}
	if(drawZ_){
	  draw_z = true;
	}
      }
    }
    
    if (draw_z) {
      Point o(mid.x(), mid.y(), b[0].z());
      Vector v(0, 0, c.z() - o.z());
      if(c.z() > b[0].z() && c.z() < b[7].z() ){
	if( viewPt.z() > c.z() ){
	  o.z(b[7].z());
	  v.z(c.z() - o.z());
	} 
	Ray r(o,v);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );  
      }
    }
  }
}
  

void
SliceRenderer::draw(Brick& b, Polygon* poly)
{
  vector<Polygon *> polys;
  polys.push_back( poly );

  load_colormap();
  load_texture( b );
//   make_texture_matrix( b );
//   enable_tex_coords();
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
//   disable_tex_coords();
  reload_ = false;
}

void 
SliceRenderer::drawWireFrame()
{
  Ray viewRay;
  compute_view( viewRay );
  
}

void 
SliceRenderer::load_colormap()
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

#endif // #if defined(SCI_OPENGL)
// void
// SliceRenderer::BuildTransferFunction()
// {
//   const int tSize = 256;
//   float mul = 1.0/(tSize - 1);
//   for ( int j = 0; j < tSize; j++ )
//   {
//     const Color c = cmap_->getColor(j*mul);
//     const double alpha = cmap_->getAlpha(j*mul);
    
//     transfer_function_[4*j + 0] = (unsigned char)(c.r());
//     transfer_function_[4*j + 1] = (unsigned char)(c.g());
//     transfer_function_[4*j + 2] = (unsigned char)(c.b());
//     transfer_function_[4*j + 3] = (unsigned char)(alpha);
//   }
// }

void
SliceRenderer::BuildTransferFunction()
{
  const int tSize = 256;
  float mul = 1.0/(tSize - 1);
  double bp = tan( 1.570796327 * 0.5 );
  
  double sliceRatio = 512.0;
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

