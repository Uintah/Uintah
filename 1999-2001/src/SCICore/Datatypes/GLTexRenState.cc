#include <SCICore/Datatypes/GLTexRenState.h>


namespace SCICore {
namespace GeomSpace  {

GLTexRenState::GLTexRenState(const GLVolumeRenderer* glvr)
    : volren( glvr )
{
  // Base Class, holds pointer to TexureRenderer and common
  // computation.
}

} // end namespace Datatypes
} // end namespace SCICore

