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

    int64_t restructured_box_size[5] = {64, 64, 64, 1, 1};
    PIDX_set_restructuring_box(file, restructured_box_size);

    //PIDX_set_resolution(this->file, 0, 2);
    PIDX_set_dims(this->file, global_bounding_box);
    PIDX_set_current_time_step(this->file, timeStep);
    PIDX_set_block_size(this->file, 16);
    PIDX_set_block_count(this->file, 128);
    
    PIDX_set_compression_type(this->file, PIDX_CHUNKING_ZFP);
    PIDX_set_lossy_compression_bit_rate(this->file, 8);

  }


} // end namespace Uintah

#endif // HAVE_PIDX
