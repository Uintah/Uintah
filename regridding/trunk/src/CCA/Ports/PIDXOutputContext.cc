#include "PIDXOutputContext.h"
#if HAVE_PIDX

namespace Uintah {
    /*
      PIDXOutputContext::PIDXOutputContext(std::string filename, 
					  unsigned int timeStep,
					  int globalExtents[5], MPI_Comm comm, int amr_levels, int** refinement_ratio)
	: filename(filename), timestep(timeStep), comm(comm), amr_levels(amr_levels), refinement_ratio(refinement_ratio)
    {
      const int *gextent;
      int returnV;
      gextent = globalExtents;
      PIDX_AMR_time_step_define(0, 2, "time%04d/");
      idx_ptr = PIDX_AMR_create(comm, (timestep-1), filename.c_str(), 15, 128, 3, 1,gextent,1,1, amr_levels, refinement_ratio);

    }
    */
    PIDXOutputContext::PIDXOutputContext() {}

    PIDXOutputContext::~PIDXOutputContext() {}

    void PIDXOutputContext::initialize(std::string filename, unsigned int timeStep, int globalExtents[5] ,MPI_Comm comm, int amr_levels, int** refinement_ratio)
    {
	this->filename = filename;
	this->timestep = timeStep;
	this->comm = comm; 
	this->amr_levels = amr_levels;
	this->refinement_ratio = refinement_ratio;
      
	const int *gextent;
	gextent = globalExtents;
      
        PIDX_AMR_time_step_define(0, 2, "time%04d/");
        //	idx_ptr = PIDX_AMR_create(comm, (timestep-1), filename.c_str(), 15, 256, 3, 1,gextent,1,1, amr_levels, refinement_ratio);
    }


} // end namespace Uintah

#endif // HAVE_PIDX
