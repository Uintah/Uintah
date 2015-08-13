#include "PIDXOutputContext.h"
#if HAVE_PIDX

namespace Uintah 
{

  PIDXOutputContext::PIDXOutputContext() 
  {}

  PIDXOutputContext::~PIDXOutputContext() 
  {
    PIDX_close_access(this->access);
  }



  
  void PIDXOutputContext::initialize(std::string filename, unsigned int timeStep, int globalExtents[3] ,MPI_Comm comm)
  {


    this->filename = filename;
    this->timestep = timeStep;
    this->comm = comm; 


    PIDX_point global_bounding_box;
    PIDX_create_access(&(this->access));
    PIDX_set_mpi_access(this->access, this->comm);

    
    PIDX_set_point_5D(global_bounding_box, globalExtents[0], globalExtents[1], globalExtents[2], 1, 1);


    PIDX_file_create(filename.c_str(), PIDX_MODE_CREATE, access, &(this->file));


    PIDX_set_dims(this->file, global_bounding_box);
    PIDX_set_current_time_step(this->file, timeStep);
    PIDX_set_block_size(this->file, 14);
    PIDX_set_block_count(this->file, 256);

  }


} // end namespace Uintah

#endif // HAVE_PIDX
