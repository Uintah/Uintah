/* REFERENCED */
static char *id="@(#) $Id$";

#include "Parallel.h"
#include <iostream>
using std::cerr;
#include <mpi.h>

namespace Uintah {
namespace Parallel {

static bool usingMPI;
#if 0
static MPI_Comm worldComm;
#endif
static int worldSize;
static int worldRank;

#if 0
static
void
MPI_Error(char* what, int errorcode)
{
    // Simple error handling for now...
    char string[1000];
    int resultlen=1000;
    //MPI_Error_string(errorcode, string, &resultlen);
    cerr << "MPI Error in " << what << ": " << string << '\n';
    exit(1);
}
#endif

void
Parallel::initializeManager(int argc, char* argv[])
{
    if(getenv("MPI_ENVIRONMENT")){
	usingMPI=true;
#if 0
	int status;
	if((status=MPI_Init(&argc, &argv)) != MPI_SUCCESS)
	    MPI_Error("MPI_Init", status);
	worldComm=MPI_COMM_WORLD;
	if((status=MPI_Comm_size(worldComm, &worldSize)) != MPI_SUCCESS)
	    MPI_Error("MPI_Comm_size", status);
	if((status=MPI_Comm_rank(worldComm, &worldRank)) != MPI_SUCCESS)
	    MPI_Error("MPI_Comm_rank", status);
#endif
    } else {
	usingMPI=false;
#if 0
	worldComm=-1;
#endif
	worldRank=0;
	worldSize=1;
    }
    cerr << "Parallel: processor " << worldRank << " of " << worldSize;
    if(usingMPI)
	cerr << " (using MPI)";
    cerr << '\n';
}

void
Parallel::finalizeManager()
{
    if(usingMPI){
#if 0
	int status;
	if((status=MPI_Finalize()) != MPI_SUCCESS)
	    MPI_Error("MPI_Finalize", status);
#endif
    }
}

} // end namespace Parallel
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:38  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
