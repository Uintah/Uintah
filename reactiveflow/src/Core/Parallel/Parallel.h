/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CORE_PARALLEL_PARALLEL_H
#define CORE_PARALLEL_PARALLEL_H

#include <Core/Thread/Thread.h>

// Macros used to eliminate excess spew on large parallel runs...
//
//   Note, make sure that MPI_Init (or MPI_Init_thread) is called
//   before using isProc0_macro.
//
#define isProc0_macro ( Uintah::Parallel::getMPIRank() == 0 &&           \
			( ( Uintah::Parallel::getNumThreads() > 1 &&	\
			    SCIRun::Thread::self()->myid() == 0 ) ||	\
			  ( Uintah::Parallel::getNumThreads() <= 1 ) ) )

#define proc0cout if( isProc0_macro ) std::cout
#define proc0cerr if( isProc0_macro ) std::cerr

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
      // Determines if MPI is being used.  MUST BE CALLED BEFORE
      // initializeManager()!  Also must be called before any one
      // calls "Uintah::Parallel::usingMPI()".  argc/argv are only
      // passed in so that they can be parsed to see if we are using
      // mpich. (mpich mpirun adds some flags to the args.)
      static void determineIfRunningUnderMPI( int argc, char** argv );

      //////////
      // Initializes MPI if necessary. 
      static void initializeManager( int& argc, char**& arg );

      //////////
      // check to see whether initializeManager has been called
      static bool isInitialized();
      
      //////////
      // Insert Documentation Here:
      static void finalizeManager( Circumstances cirumstances = NormalShutdown );

      //////////
      // Insert Documentation here:
      static ProcessorGroup* getRootProcessorGroup();

      //////////
      // Returns the MPI Rank of this process.  If this is not running
      // under MPI, than 0 is returned.  Rank value is set after call to
      // initializeManager();
      static int getMPIRank();

      //////////
      // Returns the size of MPI_Comm
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
      // Tells Parallel that Threads are not to be used
      static void noThreading();

      //////////
      // Returns true if this process is to use an accelerator or co-processor (e.g. GPU, MIC, etc), false otherwise
      static bool usingDevice();

      //////////
      // Sets whether or not to use available accelerators or co-processors (e.g. GPU, MIC, etc)
      static void setUsingDevice( bool state );

      //////////
      // Returns the number of threads that a processing element is
      // allowed to use to compute its tasks.  
      static int getNumThreads();

      //////////
      // Insert Documentation here:
      static void setNumThreads( int num );
      
   private:
      Parallel();
      Parallel( const Parallel& );
      ~Parallel();
      Parallel& operator=( const Parallel& );

//     static bool          allowThreads;

      static int             numThreads_;
      static bool            determinedIfUsingMPI_;

      static bool            initialized_;
      static bool            usingMPI_;
      static bool            usingDevice_;
//      static MPI_Comm        worldComm_;
      static int             worldRank_;
      static int             worldSize_;
      static ProcessorGroup* rootContext_;

};

} // End namespace Uintah

#endif // end CORE_PARALLEL_PARALLEL_H
