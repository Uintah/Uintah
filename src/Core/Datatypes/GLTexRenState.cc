#include <Core/Datatypes/GLTexRenState.h>


namespace SCIRun {

GLTexRenState::GLTexRenState(const GLVolumeRenderer* glvr)
    : volren( glvr )
{
  // Base Class, holds pointer to TexureRenderer and common
  // computation.
}

} // End namespace SCIRun

