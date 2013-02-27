#include "PIDXOutputContext.h"
#if HAVE_PIDX

namespace Uintah {
 
PIDXOutputContext::PIDXOutputContext(const char* filename, MPI_Comm comm)
  : filename(filename), comm(comm)
{
  int *gextent;
  int returnV;
  gextent = (int*)malloc(3 * sizeof(int));
	    
  gextent[0] = 512;
  gextent[1] = 512;
  gextent[2] = 512;
	    
// std::cout<<"Test C : Calling PIDX\n";
// PIDX_create((char*)filename, comm, 15, 256, 3, gextent);
// std::cout<<"Return Value\n" << returnV;
}
PIDXOutputContext::~PIDXOutputContext() {}

} // end namespace Uintah

#endif // HAVE_PIDX
