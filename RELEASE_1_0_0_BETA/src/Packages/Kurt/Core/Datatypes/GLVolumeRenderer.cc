#include "GLVolumeRenderer.h"
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Color.h>
#include <GL/gl.h>
#include <iostream>

namespace Kurt {

using std::cerr;
using namespace SCIRun;

//using Mutex;

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
    //AuditAllocator(default_allocator);
  if( !pre_draw(di, mat, _lighting) ) return;
  mutex.lock();
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    //AuditAllocator(default_allocator);
    setup();
    //AuditAllocator(default_allocator);
    preDraw();
    //AuditAllocator(default_allocator);
    draw();
    //AuditAllocator(default_allocator);
    postDraw();
    //AuditAllocator(default_allocator);
    cleanup();
    //AuditAllocator(default_allocator);
  }
  mutex.unlock();

}
#endif


void
GLVolumeRenderer::setup()
{
  glEnable(GL_TEXTURE_3D);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

  if( cmap.get_rep() ) {
#ifdef __sgi
    //cerr << "Using Lookup!\n";
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    
    if( cmapHasChanged || rCount != 1) {
      BuildTransferFunctions();
//       glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
// 		      GL_RGBA,
// 		      256, // try larger sizes?
// 		      GL_RGBA,  // need an alpha value...
// 		      GL_UNSIGNE_BYTE, // try shorts...
// 		      (cmap->raw1d));
      

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
      cmapHasChanged = false;
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
//  bp = tan(1.570796327*(0.5 - bcglutB*0.49999));
//   cp = tan(1.570796327*(0.5 + bcglutC*0.49999));
//   g = bcglutG*0.99999 + 0.000005;
//   for (i=0; i<=NRRDIMGLUTSIZE-1; i++) {
//     v =  (float)i/(NRRDIMGLUTSIZE-1);
//     if (v < g) {
//       tv = v/g;
//       tv = pow(tv, cp);
//       tv = tv*g;
//     }
//     else {
//       tv = (v-g)/(1.0-g);
//       tv = 1-pow(1-tv, cp);
//       tv = tv*(1.0-g) + g;
//     }
//     tv = pow(tv, bp);
//     tv = NRRD_CLAMP(0.0, tv, 1.0);
//     NRRD_INDEX(0.0, tv, 1.0, 256, ans);
//     bcglutUC[i] = ans;


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

	  //	  if( j == 128 ) cerr <<" alpha = "<< alpha<<std::endl;
	  //if( j == 128 ) cerr <<" alpha1 = "<< alpha1<<std::endl;

	  alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);
	  // if( j == 128 ) cerr <<" alpha2 = "<< alpha2<<std::endl;
	  TransferFunctions[i][4*j + 0] = (c.r()*255);
	  TransferFunctions[i][4*j + 1] = (c.g()*255);
	  TransferFunctions[i][4*j + 2] = (c.b()*255);
	  TransferFunctions[i][4*j + 3] = (alpha2*255);
	}
  }
}

} // End namespace Kurt
