
#include <PSECore/Datatypes/GLTexture3DPort.h>
#include <PSECommon/share/share.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {

using namespace PSECore::Datatypes;


extern "C" {
PSECommonSHARE IPort* make_GLTexture3DIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<GLTexture3DHandle>(module,name);
}
PSECommonSHARE OPort* make_GLTexture3DOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<GLTexture3DHandle>(module,name);
}
}

template<> clString SimpleIPort<GLTexture3DHandle>::port_type("GLTexture3D");
template<> clString SimpleIPort<GLTexture3DHandle>::port_color("gray40");


} // End namespace Datatypes
} // End namespace PSECore

