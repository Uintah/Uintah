#ifndef include_sci_mpi_h 
#define include_sci_mpi_h 

#if defined (HAVE_MPI) || defined (HAVE_MPICH)

#include <mpi.h>

#else //!HAVE_MPI


#define MPI_SUCCESS 0
#define MPI_COMM_WORLD 0

#define MPI_Init(argc,argv) MPI_SUCCESS
#define MPI_Comm_size(comm, size) (*(size)=1, MPI_SUCCESS)
#define MPI_Comm_rank(comm, rank) (*(rank)=1, MPI_SUCCESS)
#define MPI_Finalize( ) MPI_SUCCESS


#endif //HAVE_MPI || HAVE_MPICH


#endif //include_sci_mpi_h 
