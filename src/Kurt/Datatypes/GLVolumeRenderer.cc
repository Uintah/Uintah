#include "GLVolumeRenderer.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Malloc/Allocator.h>
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
  //using SCICore::Thread::Mutex;


double GLVolumeRenderer::swapMatrix[16] = { 0,0,1,0,
					    0,1,0,0,
					    1,0,0,0,
					    0,0,0,1};

GLVolumeRenderer::GLVolumeRenderer(int id) 
  : GeomObj( id ), mutex("GLVolumeRenderer Mutex"),
  tex(0), cmap(0),
  controlPoint(Point(0,0,0)), slices(0),
  cmapHasChanged(true),
  slice_alpha(1.0),
  drawX(false),drawY(false),drawZ(false),drawView(false),
  _interp(true), _lighting(0), _tp(0), _roi(0), _fr(0), _los(0),
  _oo(0), _mip(0), _atten(0), _planes(0),
  _state(state(_fr, 0)),  _gl_state(state(_oo, 0))
{

  NOT_FINISHED("GLVolumeRenderer::GLVolumeRenderer(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLVolumeRenderer::GLVolumeRenderer(int id, 
				   GLTexture3DHandle tex,
				   ColorMapHandle map)
 : GeomObj( id ),  mutex("GLVolumeRenderer Mutex"),
  tex(tex), cmap(map),
  controlPoint(Point(0,0,0)), slices(0),
  cmapHasChanged(true),
  slice_alpha(1.0), 
  drawX(false),drawY(false),drawZ(false),drawView(false),
  _interp(true),  _lighting(0), _tp(0), _roi(0), _fr(0), _los(0),
  _oo(0), _mip(0), _atten(0), _planes(0),
  _state(state(_fr, 0)),  _gl_state(state(_oo, 0))
{
}

GLVolumeRenderer::GLVolumeRenderer(const GLVolumeRenderer& copy)
  : GeomObj( copy.id ), mutex("GLVolumeRenderer Mutex"),
  tex(copy.tex),  cmap(copy.cmap),
  controlPoint(copy.controlPoint), slices(copy.slices),
  cmapHasChanged(copy.cmapHasChanged),
  slice_alpha(copy.slice_alpha), _state(copy._state),
  _gl_state(copy._gl_state),
  drawX(copy.drawX),drawY(copy.drawY),
    drawZ(copy.drawZ),drawView(copy.drawView),  _lighting(copy._lighting),
   _interp(copy._interp), _tp(copy._tp), _roi(copy._roi), _fr(copy._fr),
   _los(copy._los), _oo(copy._oo), _mip(copy._mip), _atten(copy._atten),
   _planes(copy._planes)
{
} 

GLVolumeRenderer::~GLVolumeRenderer()
{

  //delete cmap;
  //delete _state;
  //delete _gl_state;
  delete _tp;
  delete _roi;
  delete _fr;
  delete _los;
  delete _oo;
  delete _mip;
  delete _atten;
  delete _planes;
  
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
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
  if( !pre_draw(di, mat, _lighting) ) return;
  mutex.lock();
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
    setup();
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
    preDraw();
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
    draw();
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
    postDraw();
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
    cleanup();
    //SCICore::Malloc::AuditAllocator(SCICore::Malloc::default_allocator);
  }
  mutex.unlock();

}
#endif


void
GLVolumeRenderer::setup()
{



  glEnable(GL_TEXTURE_3D_EXT);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

  if( cmap.get_rep() ) {
#ifdef __sgi
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    
    if( cmapHasChanged ) {
      glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
		      GL_RGBA,
		      256, // try larger sizes?
		      GL_RGBA,  // need an alpha value...
		      GL_UNSIGNED_BYTE, // try shorts...
		      (cmap->raw1d));
      

//       if( cmap->IsScaled()){
// 	unsigned char map[256];
// 	double t_min, t_max, c_min, c_max;
// 	int min, max;
// 	tex->get_minmax(t_min, t_max);
// 	cmap->getMin(c_min); cmap->getMax(c_max);
// 	min = (c_min - t_min)*255/(t_max - t_min);
// 	max = (c_max - t_min)*255/(t_max - t_min);
// 	int i,j;
// 	for(i = 0; i < min; i++)
// 	  map[i] = (cmap->raw1d)[0];
// 	for(i = max; i < 256; i++)
// 	  map[i] = (cmap->raw1d)[255];
// 	min = ((min > 0) ? min : 0);
// 	max = ((max < 256) ? max : 255);
// 	for(i = min, j = 0;  i < max ; i++, j = (i-min)*255.0/(max - min))
// 	  map[i] = (cmap->raw1d)[j];
//       }


#endif
      cmapHasChanged = true;
    }
  }
  glColor4f(1,1,1,1); // set to all white for modulation

}


void
GLVolumeRenderer::cleanup()
{


#ifdef __sgi
  if( cmap.get_rep() )
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
