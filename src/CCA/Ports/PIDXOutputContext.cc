#include "PIDXOutputContext.h"
#if HAVE_PIDX

namespace Uintah {
 
  PIDXOutputContext::PIDXOutputContext(std::string filename, 
                                       unsigned int timeStep,
                                       int globalExtents[5], MPI_Comm comm)
    : filename(filename), timestep(timeStep), comm(comm)
{
  int gextent[5];
  int returnV;

  for (int i = 0; i < 5; i++)
    gextent[i] = globalExtents[i];

	    
// std::cout<<"Test C : Calling PIDX\n";
// PIDX_create((char*)filename, comm, 15, 256, 3, gextent);
// std::cout<<"Return Value\n" << returnV;
}
PIDXOutputContext::~PIDXOutputContext() {}

} // end namespace Uintah

#endif // HAVE_PIDX
