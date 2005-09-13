#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class ProcessorGroup;

/**************************************

CLASS
   Parallel
   
   Short description...

GENERAL INFORMATION

   Parallel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Parallel

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class Parallel {
   public:
      enum Circumstances {
	 NormalShutdown,
	 Abort
      };

      // Determines if MPI is being used.  MUST BE CALLED BEFORE
      // initializeManager()!  Also must be called before any one
      // calls "Uintah::Parallel::usingMPI()".  argc/argv are only
      // passed in so that they can be parsed to see if we are using
      // mpich. (mpich mpirun adds some flags to the args.)
      static void determineIfRunningUnderMPI( int argc, char** argv );

      //////////

      // Initializes MPI if necessary.  "scheduler" is used to tell
      // MPI to initialize the thread safety MPI libs (ie: MPI lib
      // thread safety is requested if (scheduler ==
      // "MixedScheduler")) If MPI thread safety is not needed, then
      // in theory MPI uses a faster library.
      static void initializeManager( int& argc, char**& argv, 
				     const std::string & scheduler );
      
      //////////
      // Insert Documentation Here:
      static void finalizeManager(Circumstances cirumstances = NormalShutdown);

      //////////
      // Insert Documenatation here:
      static ProcessorGroup* getRootProcessorGroup();

      //////////
      // Returns the MPI Rank of this process.  If this is not running
      // under MPI, than 0 is returned.  Rank value is set after call to
      // initializeManager();
      static int getMPIRank();
      
      //////////
      // Returns true if this process is using MPI
      static bool usingMPI();
      
      //////////
      // Ignore mpi probe, and force this to use MPI
      static void forceMPI();

      //////////
      // Ignore mpi probe, and force this to not use MPI
      static void forceNoMPI();

      //////////
      // Tells Parallel that Threads are not to be used.
      static void noThreading();

      //////////
      // Returns the number of threads that a processing element is
      // allowed to use to compute its tasks.  
      static int getMaxThreads();

      static void setMaxThreads( int maxNumThreads );
      
   private:
      Parallel();
      Parallel(const Parallel&);
      ~Parallel();
      Parallel& operator=(const Parallel&);
   };
} // End namespace Uintah



#endif
