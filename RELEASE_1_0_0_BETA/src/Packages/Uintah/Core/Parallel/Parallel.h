#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

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
      // Insert Documentation Here:
      static void initializeManager(int& argc, char**& argv);
      
      //////////
      // Insert Documentation Here:
      static void finalizeManager(Circumstances cirumstances = NormalShutdown);

      //////////
      // Insert Documenatation here:
      static ProcessorGroup* getRootProcessorGroup();
      
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
