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

#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Color.h>
#include <GL/gl.h>
#include <iostream>

namespace SCIRun {

using std::cerr;


double GLVolumeRenderer::swapMatrix[16] = { 0,0,1,0,
					    0,1,0,0,
					    1,0,0,0,
					    0,0,0,1};

int GLVolumeRenderer::rCount = 0;

GLVolumeRenderer::GLVolumeRenderer(int id) 
  : GeomObj( id ),
    _state(state(_fr, 0)),
    _roi(0),
    _fr(0),
    _los(0),
    _tp(0),
    _gl_state(state(_oo, 0)),
    _oo(0),
    _mip(0),
    _atten(0),
    _planes(0),
    slices(0),
    tex(0),
    mutex("GLVolumeRenderer Mutex"),
    cmap(0),
    controlPoint(Point(0,0,0)),
    slice_alpha(1.0),
    cmapHasChanged(true),
    drawX(false),
    drawY(false),
    drawZ(false),
    drawView(false),
    _interp(true),
    _lighting(0)
{
  rCount++;
  NOT_FINISHED("GLVolumeRenderer::GLVolumeRenderer(int id, const Texture3D* tex, ColorMap* cmap)");
}


GLVolumeRenderer::GLVolumeRenderer(int id, 
				   GLTexture3DHandle tex,
				   ColorMapHandle map)
  : GeomObj( id ),
    _state(state(_fr, 0)),
    _roi(0),
    _fr(0),
    _los(0),
    _tp(0),
    _gl_state(state(_oo, 0)),
    _oo(0),
    _mip(0),
    _atten(0),
    _planes(0),
    slices(0),
    tex(tex),
    mutex("GLVolumeRenderer Mutex"),
    cmap(map),
    controlPoint(Point(0,0,0)),
    slice_alpha(1.0),
    cmapHasChanged(true),
    drawX(false),
    drawY(false),
    drawZ(false),
    drawView(false),
    di_(0),
    _interp(true),
    _lighting(0)
{
  rCount++;
}

GLVolumeRenderer::GLVolumeRenderer(const GLVolumeRenderer& copy)
  : GeomObj( copy.id ),
    _state(copy._state),
    _roi(copy._roi),
    _fr(copy._fr),
    _los(copy._los),
    _tp(copy._tp),
    _gl_state(copy._gl_state),
    _oo(copy._oo),
    _mip(copy._mip),
    _atten(copy._atten),
    _planes(copy._planes),
    slices(copy.slices),
    tex(copy.tex),
    mutex("GLVolumeRenderer Mutex"),
    cmap(copy.cmap),
    controlPoint(copy.controlPoint),
    slice_alpha(copy.slice_alpha),
    cmapHasChanged(copy.cmapHasChanged),
    drawX(copy.drawX),
    drawY(copy.drawY),
    drawZ(copy.drawZ),
    drawView(copy.drawView),
    di_(copy.di_),
    _interp(copy._interp),
    _lighting(copy._lighting)
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
  di_ = di;
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
  di_ = 0;
  mutex.unlock();

}
#endif


void
GLVolumeRenderer::setup()
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
    if( cmapHasChanged || rCount != 1) {
      BuildTransferFunctions();
      cmapHasChanged = false;
    }
  }
  glColor4f(1,1,1,1); // set to all white for modulation
}


void
GLVolumeRenderer::cleanup()
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
  //  double L = (tex->max() - tex->min()).length();
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
	  alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);

//	  if( j == 128 ) cerr <<" alpha = "<< alpha<<std::endl;
//	  if( j == 128 ) cerr <<" alpha1 = "<< alpha1<<std::endl;
//	  if( j == 128 ) cerr <<" c.r() = "<< c.r()<<std::endl;
//	  if( j == 128 ) cerr <<" c.g() = "<< c.g()<<std::endl;
//	  if( j == 128 ) cerr <<" c.b() = "<< c.b()<<std::endl;
//	  if( j == 128 ) cerr <<" alpha2 = "<< alpha2<<std::endl;

	  TransferFunctions[i][4*j + 0] = (unsigned char)(c.r()*255);
	  TransferFunctions[i][4*j + 1] = (unsigned char)(c.g()*255);
	  TransferFunctions[i][4*j + 2] = (unsigned char)(c.b()*255);
	  TransferFunctions[i][4*j + 3] = (unsigned char)(alpha2*255);
	}
  }
}
} // End namespace SCIRun
