/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

#include <sgi_stl_warnings_off.h>
#include   <string>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/Parallel/uintahshare.h>

// macro used by components to eliminate excess spew on 
// large parallel runs
#define proc0cout if(Parallel::getMPIRank()==0) cout

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

   class UINTAHSHARE Parallel {
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

      // check to see whether initializeManager has been called
      static bool isInitialized();
      
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

      static int getMPISize();
      
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
