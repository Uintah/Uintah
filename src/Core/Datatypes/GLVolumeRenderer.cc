#include <SCICore/Datatypes/GLVolumeRenderer.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geom/Color.h>
#include <GL/gl.h>
#include <iostream>

namespace SCICore {
namespace GeomSpace  {

using std::cerr;

using namespace SCICore::Geometry;
using namespace SCICore::Datatypes;
using SCICore::Datatypes::Octree;
using SCICore::Datatypes::Brick;

double GLVolumeRenderer::swapMatrix[16] = { 0,0,1,0,
					    0,1,0,0,
					    1,0,0,0,
					    0,0,0,1};

int GLVolumeRenderer::rCount = 0;

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
  rCount++;
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
  rCount++;
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
  rCount++;
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
  rCount--;
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
    
    if( cmapHasChanged || rCount != 1) {
      BuildTransferFunctions();
      cmapHasChanged = false;
    }
#endif
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
GLVolumeRenderer::saveobj(std::ostream&, const clString&, GeomSave*)
{
   NOT_FINISHED("GLVolumeRenderer::saveobj");
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
GLVolumeRenderer::BuildTransferFunctions( )
{
  const int tSize = 256;
  int defaultSamples = 512;
  double L = (tex->max() - tex->min()).length();
  //  double dt = L/defaultSamples;
  float mul = 1.0/(tSize - 1);
  if( tex->depth() > 8 ) {
    cerr<<"Error: Texture too deep\n";
    return;
  }

  double bp = tan( 1.570796327*(0.5 - slice_alpha*0.49999));
  for(int i = 0; i < tex->depth() + 1; i++){
      double sliceRatio = defaultSamples/(double(slices)/
					  pow(2.0, tex->depth() - i));

      double alpha, alpha1, alpha2;
      for( int j = 0; j < tSize; j++ )
	{
	  Color c = FindColor(cmap->rawRampColor,
				    cmap->rawRampColorT, j*mul);
	  alpha = FindAlpha(cmap->rawRampAlpha,
			    cmap->rawRampAlphaT, j*mul);


	  alpha1 = pow(alpha, bp);

	  if( j == 128 ) cerr <<" alpha = "<< alpha<<std::endl;
	  if( j == 128 ) cerr <<" alpha1 = "<< alpha1<<std::endl;

	  alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);
	  if( j == 128 ) cerr <<" alpha2 = "<< alpha2<<std::endl;
	  TransferFunctions[i][4*j + 0] = (c.r()*255);
	  TransferFunctions[i][4*j + 1] = (c.g()*255);
	  TransferFunctions[i][4*j + 2] = (c.b()*255);
	  TransferFunctions[i][4*j + 3] = (alpha2*255);
	}
  }
}
} // namespace SCICore
} // namespace GeomSpace
