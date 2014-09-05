

#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <mpi.h>

using namespace SCIRun;
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
Parallel::setMaxThreads( int maxNumThreads )
{
   ::maxThreads = maxNumThreads;
   ::allowThreads = true;
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
      const char* oldtag = AllocatorSetDefaultTagMalloc("MPI initialization");
      if((status=MPI_Init(&argc, &argv)) != MPI_SUCCESS)
	 MpiError("MPI_Init", status);
      worldComm=MPI_COMM_WORLD;
      int worldSize;
      if((status=MPI_Comm_size(worldComm, &worldSize)) != MPI_SUCCESS)
	 MpiError("MPI_Comm_size", status);
      int worldRank;
      if((status=MPI_Comm_rank(worldComm, &worldRank)) != MPI_SUCCESS)
	 MpiError("MPI_Comm_rank", status);
      AllocatorSetDefaultTagMalloc(oldtag);
      rootContext = scinew ProcessorGroup(0, worldComm, true,
					  worldRank,worldSize);
   } else {
      rootContext = scinew ProcessorGroup(0,0, false, 0, 1);
   }

   ProcessorGroup* world = getRootProcessorGroup();
   //   cerr << "Parallel: processor " << world->myrank() + 1 
   //<< " of " << world->size();
   if(world->myrank() == 0){
      cerr << "Parallel: " << world->size() << " processors";
      if(::usingMPI)
	 cerr << " (using MPI)";
      cerr << '\n';
   }
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
   if(rootContext){
     delete rootContext;
     rootContext=0;
   }
}

ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if(!rootContext)
      throw InternalError("Parallel not initialized");

   return rootContext;
}

