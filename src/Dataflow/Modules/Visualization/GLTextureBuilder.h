#ifndef GLTEXTUREBUILDER_H
#define GLTEXTUREBUILDER_H
/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GLTexture3D.h>
#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <GL/glx.h>


namespace SCIRun {
  class GLTexture3D;
}

namespace SCIRun {



class GLTextureBuilder : public Module {

public:
  GLTextureBuilder( const clString& id);

  virtual ~GLTextureBuilder();

  virtual void execute();

private:
  ScalarFieldIPort *inscalarfield;

  GLTexture3DOPort* otexture;
   
  Point Smin,Smax;
  Vector ddv;
  
  ScalarFieldHandle sfrg;
  GLTexture3DHandle tex;

  TCLint max_brick_dim;
  TCLdouble min, max;
  TCLint isFixed;
//  bool MakeContext(Display *dpy, GLXContext& cx);
  // void DestroyContext(Display *dpy, GLXContext& cx);

};

} // End namespace SCIRun

#endif
