
#include "GLTexture3DPort.h"

namespace PSECore {
namespace Datatypes {

using namespace Kurt::Datatypes;


template<> clString SimpleIPort<GLTexture3DHandle>::port_type("GLTexture3D");
template<> clString SimpleIPort<GLTexture3DHandle>::port_color("gray40");


} // End namespace Datatypes
} // End namespace PSECore

