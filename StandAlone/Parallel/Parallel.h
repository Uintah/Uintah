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
      static void initializeManager(int argc, char** argv);
      
      //////////
      // Insert Documentation Here:
      static void finalizeManager(Circumstances cirumstances = NormalShutdown);

      //////////
      // Insert Documenatation here:
      static ProcessorGroup* getRootProcessorGroup();
      
      //////////
      // Returns true if this process is using MPI
      static bool usingMPI();
      
   private:
      Parallel();
      Parallel(const Parallel&);
      ~Parallel();
      Parallel& operator=(const Parallel&);
      
   };

} // end namespace Uintah

//
// $Log$
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
