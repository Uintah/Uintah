#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

#include <string>

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

      //////////
      // Determines if MPI is being used, and if so, initializes MPI.
      // "scheduler" is used to determine if MPI is initialized with
      // thread safety on (ie: on if scheduler == "MixedScheduler").
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
