/* REFERENCED */
static char *id="@(#) $Id$";

#include "Parallel.h"

#include <iostream>
#include <mpi.h>
#include <stdlib.h>

using std::cerr;

namespace Uintah {
namespace Parallel {

static bool usingMPI;
static MPI_Comm worldComm;

static int worldSize;
static int worldRank;

static
void
MpiError(char* what, int errorcode)
{
    // Simple error handling for now...
    char string[1000];
    int resultlen=1000;
    //MPI_Error_string(errorcode, string, &resultlen);
    cerr << "MPI Error in " << what << ": " << string << '\n';
    exit(1);
}

int
Parallel::getSize() 
{
  return worldSize;
}

int
Parallel::getRank()
{
  return worldRank;
}

void
Parallel::initializeManager(int argc, char* argv[])
{
    if(getenv("MPI_ENVIRONMENT")){

	usingMPI=true;

	int status;
	if((status=MPI_Init(&argc, &argv)) != MPI_SUCCESS)
	    MpiError("MPI_Init", status);
	worldComm=MPI_COMM_WORLD;
	if((status=MPI_Comm_size(worldComm, &worldSize)) != MPI_SUCCESS)
	    MpiError("MPI_Comm_size", status);
	if((status=MPI_Comm_rank(worldComm, &worldRank)) != MPI_SUCCESS)
	    MpiError("MPI_Comm_rank", status);
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
	int status;
	if((status=MPI_Finalize()) != MPI_SUCCESS)
	    MpiError("MPI_Finalize", status);
    }
}

} // end namespace Parallel
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2000/04/19 20:58:56  dav
// adding MPI support
//
// Revision 1.3  2000/03/17 09:30:21  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  2000/03/16 22:08:38  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
