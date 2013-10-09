#include "PIDXOutputContext.h"
#if HAVE_PIDX

namespace Uintah {
 
  PIDXOutputContext::PIDXOutputContext(std::string filename, 
                                       unsigned int timeStep,
                                       int globalExtents[5], MPI_Comm comm)
    : filename(filename), timestep(timeStep), comm(comm)
{
  const int *gextent;
  int returnV;

  //  for (int i = 0; i < 5; i++)
  gextent = globalExtents;


  PIDX_time_step_define(0, 2, "time%04d/");	    
  // std::cout << "timestep = " << timestep << " filename = " << filename << std::endl;
  // std::cout << "gextent[0] = " << gextent[0] << " gextent[1] = " << gextent[1] <<
  //   "gextent[2] = " << gextent[2] << " gextent[3] = " << gextent[3] <<
  //   "gextent[4] = " << gextent[4] << std::endl;
  idx_ptr = PIDX_create(comm, (timestep-1), filename.c_str(), 15, 128, 3, 1,gextent,1);
// std::cout<<"Test C : Calling PIDX\n";
// PIDX_create((char*)filename, comm, 15, 256, 3, gextent);
// std::cout<<"Return Value\n" << returnV;
}
PIDXOutputContext::~PIDXOutputContext() {}

} // end namespace Uintah

#endif // HAVE_PIDX
