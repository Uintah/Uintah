#ifndef GLTEXTUREBUILDER_H
#define GLTEXTUREBUILDER_H
/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <Kurt/Datatypes/GLTexture3DPort.h>
#include <Kurt/Datatypes/GLTexture3D.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <GL/glx.h>



namespace SCICore{
  namespace Datatypes{
   class ScalarFieldRGuchar;
  }
}

namespace Kurt {
  namespace Datatypes {
  class GLTexture3D;
  }
namespace Modules {

using namespace SCICore::TclInterface;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Datatypes::ScalarFieldRGuchar;

using PSECore::Datatypes::GLTexture3DOPort;
using PSECore::Datatypes::ScalarFieldIPort;
using PSECore::Datatypes::ScalarFieldHandle;
using PSECore::Dataflow::Module;
using Kurt::Datatypes::GLTexture3D;
using Kurt::Datatypes::GLTexture3DHandle;

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

} // namespace Modules
} // namespace Uintah

#endif
