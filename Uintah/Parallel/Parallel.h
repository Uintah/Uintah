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

} // end namespace Uintah

//
// $Log$
// Revision 1.10  2000/09/29 20:43:42  dav
// Added setMaxThreads()
//
// Revision 1.9  2000/09/28 22:21:34  dav
// Added code that allows the MPIScheduler to run correctly even if
// PSE_MAX_THREADS is set.  This was messing up the assigning of resources.
//
// Revision 1.8  2000/09/26 21:42:25  dav
// added getMaxThreads
//
// Revision 1.7  2000/09/25 18:13:51  sparker
// Correctly handle mpich
//
// Revision 1.6  2000/07/27 22:39:54  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.5  2000/06/17 07:06:48  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.4  2000/04/26 06:49:15  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/04/19 20:58:56  dav
// adding MPI support
//
// Revision 1.2  2000/03/16 22:08:39  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
