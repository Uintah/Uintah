
// $Id$

#include <Uintah/Parallel/Parallel.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream>
#include <mpi.h>

using namespace SCICore::Exceptions;
using namespace Uintah;

using std::cerr;
using std::cout;
using std::string;

static bool            allowThreads;
static bool            usingMPI = false;
static int             maxThreads = 1;
static MPI_Comm        worldComm = -1;
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

int
Parallel::getMaxThreads()
{
   return ::maxThreads;
}

void
Parallel::noThreading()
{
  ::allowThreads = false;
  ::maxThreads = 1;
}

void
Parallel::initializeManager(int& argc, char**& argv)
{
   if( char * max = getenv( "PSE_MAX_THREADS" ) ){
      ::maxThreads = atoi( max );
      ::allowThreads = true;
      cerr << "PSE_MAX_THREADS set to " << ::maxThreads << "\n";

      if( ::maxThreads <= 0 || ::maxThreads > 16 ){
         // Empirical evidence points to 16 being the most threads
         // that we should use... (this isn't conclusive evidence)
         cerr << "PSE_MAX_THREADS is out of range 1..16\n";
         throw InternalError( "PSE_MAX_THREADS is out of range 1..16\n" );
      }
   }
  
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
   }

   ProcessorGroup* world = getRootProcessorGroup();
   cerr << "Parallel: processor " << world->myrank() + 1 
	<< " of " << world->size();
   if(::usingMPI)
      cerr << " (using MPI)";
   cerr << '\n';
}

void
Parallel::finalizeManager(Circumstances circumstances)
{
   if(::usingMPI){
      if(circumstances == Abort){
	 cerr.flush();
	 cout.flush();
	 sleep(1);
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
// Revision 1.14  2000/09/29 19:52:56  dav
// Added cerr and cout flushes and a sleep before the abort... In the past
// print statements have been lost because of the abort.  Hopefully this
// will allow all output to be displayed before the program actually dies.
//
// Revision 1.13  2000/09/28 22:21:34  dav
// Added code that allows the MPIScheduler to run correctly even if
// PSE_MAX_THREADS is set.  This was messing up the assigning of resources.
//
// Revision 1.12  2000/09/26 21:44:34  dav
// removed PSE_MPI_DEBUG_LEVEL
//
// Revision 1.11  2000/09/26 21:42:34  dav
// added getMaxThreads
//
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
