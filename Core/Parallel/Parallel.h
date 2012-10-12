/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

#include <iostream>


// Macro used by components to eliminate excess spew on large parallel runs...
//
//   Make sure that MPI_Init is called before using 'proc0cout'...
//
#define proc0cout if( Uintah::Parallel::getMPIRank() == 0 ) std::cout

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
      // Returns true if this process is to use GPUs, false otherwise
      static bool usingGPU();

      //////////
      // Sets whether or not to use available GPUs
      static void setUsingGPU( bool state );

      //////////
      // Returns the number of threads that a processing element is
      // allowed to use to compute its tasks.  
      static int getNumThreads();

      //////////
      // Insert Documentation here:
      static void setNumThreads( int num );
      
   private:
      Parallel();
      Parallel(const Parallel&);
      ~Parallel();
      Parallel& operator=(const Parallel&);
   };
} // End namespace Uintah

#endif
