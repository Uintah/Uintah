
// $Id$

#include <Uintah/Parallel/Parallel.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream>
#include <mpi.h>
#include <stdlib.h>

using namespace Uintah;
using std::cerr;
using namespace SCICore::Exceptions;
using std::string;

static bool usingMPI;
static MPI_Comm worldComm;
static ProcessorGroup* rootContext = 0;

static
void
MpiError(char* what, int errorcode)
{
   // Simple error handling for now...
   char string[1000];
   int resultlen=1000;
   MPI_Error_string(errorcode, string, &resultlen);
   cerr << "MPI Error in " << what << ": " << string << '\n';
   exit(1);
}

bool
Parallel::usingMPI()
{
   return ::usingMPI;
}

void
Parallel::initializeManager(int& argc, char**& argv)
{
   ::usingMPI=false;
   // Look for SGI MPI
   if(getenv("MPI_ENVIRONMENT")){
      ::usingMPI=true;
   } else {
      // Look for mpich
      for(int i=0;i<argc;i++){
	 string s = argv[i];
	 if(s.substr(0,3) == "-p4")
	    ::usingMPI=true;
      }
   }
   if(::usingMPI){	
      int status;
      if((status=MPI_Init(&argc, &argv)) != MPI_SUCCESS)
	 MpiError("MPI_Init", status);
      worldComm=MPI_COMM_WORLD;
      int worldSize;
      if((status=MPI_Comm_size(worldComm, &worldSize)) != MPI_SUCCESS)
	 MpiError("MPI_Comm_size", status);
      int worldRank;
      if((status=MPI_Comm_rank(worldComm, &worldRank)) != MPI_SUCCESS)
	 MpiError("MPI_Comm_rank", status);
      rootContext = scinew ProcessorGroup(0, worldComm, true,
					  worldRank,worldSize);
   } else {
      rootContext = scinew ProcessorGroup(0,0, false, 0, 1);
      worldComm=-1;
   }

   ProcessorGroup* world = getRootProcessorGroup();
   cerr << "Parallel: processor " << world->myrank() << " of " << world->size();
   if(::usingMPI)
      cerr << " (using MPI)";
   cerr << '\n';
}

void
Parallel::finalizeManager(Circumstances circumstances)
{
   if(::usingMPI){
      if(circumstances == Abort){
	 MPI_Abort(worldComm, 1);
      } else {
	 int status;
	 if((status=MPI_Finalize()) != MPI_SUCCESS)
	    MpiError("MPI_Finalize", status);
      }
   }
}

ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if(!rootContext)
      throw InternalError("Parallel not initialized");

   return rootContext;
}

//
// $Log$
// Revision 1.10  2000/09/25 18:44:59  sparker
// Added using statement for std::string
//
// Revision 1.9  2000/09/25 18:13:51  sparker
// Correctly handle mpich
//
// Revision 1.8  2000/07/27 22:39:54  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.7  2000/06/17 07:06:48  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.6  2000/04/26 06:49:15  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/25 00:41:23  dav
// more changes to fix compilations
//
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
