#include "GLVolumeRenderer.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <GL/gl.h>
#include <iostream>

namespace SCICore {
namespace GeomSpace  {

using std::cerr;

using namespace SCICore::Geometry;
using namespace SCICore::Datatypes;
using Kurt::Datatypes::Octree;
using Kurt::Datatypes::Brick;
using Kurt::Datatypes::SliceTable;


double GLVolumeRenderer::swapMatrix[16] = { 0,0,1,0,
					    0,1,0,0,
					    1,0,0,0,
					    0,0,0,1};

GLVolumeRenderer::GLVolumeRenderer(int id) 
 : GeomObj( id ), 
  tex(0),  cmap(0),
  controlPoint(Point(0,0,0)), slices(0),
  cmapHasChanged(true),
  slice_alpha(1.0), _state(FullRes::Instance(this)),
  _gl_state(GLOverOp::Instance( this )),
  drawX(false),drawY(false),drawZ(false),drawView(false)
{
  NOT_FINISHED("GLVolumeRenderer::GLVolumeRenderer(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLVolumeRenderer::GLVolumeRenderer(int id, 
				   GLTexture3DHandle tex,
				   ColorMapHandle map)
 : GeomObj( id ), 
  tex(tex.get_rep()),  cmap(map->raw1d),
  controlPoint(Point(0,0,0)), slices(0),
  cmapHasChanged(true),
  slice_alpha(1.0), _state(FullRes::Instance(this)),
  _gl_state(GLOverOp::Instance( this )),
  drawX(false),drawY(false),drawZ(false),drawView(false)
{
  
}

GLVolumeRenderer::GLVolumeRenderer(const GLVolumeRenderer& copy)
 : GeomObj( copy.id ),
  tex(copy.tex), cmap(copy.cmap),
  controlPoint(copy.controlPoint), slices(copy.slices),
  cmapHasChanged(copy.cmapHasChanged),
  slice_alpha(copy.slice_alpha), _state(copy._state),
  _gl_state(copy._gl_state),
  drawX(copy.drawX),drawY(copy.drawY),
  drawZ(copy.drawZ),drawView(copy.drawView)
{
} 

GLVolumeRenderer::~GLVolumeRenderer()
{

  delete cmap;
  delete _state;
  delete _gl_state;
}

GeomObj* 
GLVolumeRenderer::clone()
{
  return new GLVolumeRenderer( *this );
}

#ifdef SCI_OPENGL
void 
GLVolumeRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  if( !pre_draw(di, mat, 0) ) return;
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    setup();
    preDraw();
    draw();
    postDraw();
    cleanup();
  }
}
#endif


void
GLVolumeRenderer::setup()
{



  glEnable(GL_TEXTURE_3D_EXT);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

  if( cmap ) {
#ifdef __sgi
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    if( cmapHasChanged ) {
      glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI,
		      GL_RGBA,
		      256, // try larger sizes?
		      GL_RGBA,  // need an alpha value...
		      GL_UNSIGNED_BYTE, // try shorts...
		      cmap);
#endif
      cmapHasChanged = false;
    }
  }
  glColor4f(1,1,1,1); // set to all white for modulation

//   glMatrixMode(GL_TEXTURE);
//   glPushMatrix();
//   glRotated( 90, 0,1,0);
//   glMatrixMode(GL_MODELVIEW);

}


void
GLVolumeRenderer::cleanup()
{

//   glMatrixMode(GL_TEXTURE);
//   glPopMatrix();
//   glMatrixMode(GL_MODELVIEW);

#ifdef __sgi
  if( cmap )
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#endif
  glDisable(GL_TEXTURE_3D_EXT);
  // glEnable(GL_DEPTH_TEST);  
}  


#define GLVOLUMERENDERER_VERSION 1

void GLVolumeRenderer::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("GLVolumeRenderer::io");
}


bool
GLVolumeRenderer::saveobj(std::ostream&, const clString& format, GeomSave*)
{
   NOT_FINISHED("GLVolumeRenderer::saveobj");
    return false;
}

  
} // namespace SCICore
} // namespace GeomSpace
