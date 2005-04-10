#include "GLTexRenState.h"


namespace Kurt {
namespace GeomSpace  {

GLTexRenState::GLTexRenState(const GLVolumeRenderer* glvr)
    : volren( glvr )
{
  // Base Class, holds pointer to TexureRenderer and common
  // computation.
}

} // end namespace GeomSpace
} // end namespace Kurt

