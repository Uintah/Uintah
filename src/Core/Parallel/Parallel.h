/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <thread>


// Macros used to eliminate excess output on large parallel runs
//
//   Note, make sure that Uintah::MPI::Init (or Uintah::MPI::Init_thread)
//   is called before using isProc0_macro.
//
#define isProc0_macro ( Uintah::Parallel::getMPIRank()      == 0 ) && \
                      ( Uintah::Parallel::getMainThreadID() == std::this_thread::get_id() )

#define proc0cout if( isProc0_macro ) std::cout
#define proc0cerr if( isProc0_macro ) std::cerr

#define MAX_THREADS     64
#define MAX_HALO_DEPTH  5

namespace Uintah {

class ProcessorGroup;

/**************************************

CLASS
   Parallel
   

GENERAL INFORMATION

   Parallel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Parallel

DESCRIPTION
  
****************************************/

class Parallel {

   public:

      enum Circumstances {
            NormalShutdown
          , Abort
      };

      //////////
      // Initializes MPI if necessary. 
      static void initializeManager( int& argc, char**& arg );

      //////////
      // Check to see whether initializeManager has been called
      static bool isInitialized();
      
      //////////
      // Shuts down and finalizes the MPI runtime in a safe manner
      static void finalizeManager( Circumstances cirumstances = NormalShutdown );

      //////////
      // Returns the root context ProcessorGroup
      static ProcessorGroup* getRootProcessorGroup();

      //////////
      // Returns the MPI Rank of this process.  If this is not running under MPI,
      // than 0 is returned.  Rank value is set after call to initializeManager();
      static int getMPIRank();

      //////////
      // Returns the size of MPI_Comm
      static int getMPISize();
      
      //////////
      // Returns true if this process is using MPI
      static bool usingMPI();

      //////////
      // Returns true if this process is to use an accelerator or co-processor (e.g. GPU, MIC, etc), false otherwise
      static bool usingDevice();

      //////////
      // Sets whether or not to use available accelerators or co-processors (e.g. GPU, MIC, etc)
      static void setUsingDevice( bool state );

      //////////
      // Returns the number of threads that a processing element is allowed to use to compute its tasks.
      static int getNumThreads();

      //////////
      // Returns the number of thread partitions that a processing element is allowed to use to compute its tasks.
      static int getNumPartitions();

      //////////
      // Returns the number of threads per partition.
      static int getThreadsPerPartition();

      //////////
      // Returns the ID of the main thread, via std::this_thread::get_id()
      static std::thread::id getMainThreadID();

      //////////
      // Sets the number of task runner threads to the value specified
      static void setNumThreads( int num );
      
      //////////
      // Sets the number of task runner OMP thread partitions to the value specified
      static void setNumPartitions( int num );

      //////////
      // Sets the number of threads per OMP partition
      static void setThreadsPerPartition( int num );

      //////////
      // Passes the specified exit code to std::exit()
      static void exitAll( int code );


   private:

      // eliminate public construction/destruction, copy, assignment and move
      Parallel();
     ~Parallel();

      Parallel( const Parallel & )            = delete;
      Parallel& operator=( const Parallel & ) = delete;
      Parallel( Parallel && )                 = delete;
      Parallel& operator=( Parallel && )      = delete;

      static bool              s_initialized;
      static bool              s_using_device;
      static int               s_num_threads;
      static int               s_num_partitions;
      static int               s_threads_per_partition;
      static int               s_world_rank;
      static int               s_world_size;
      static std::thread::id   s_main_thread_id;
      static ProcessorGroup*   s_root_context;

};

} // End namespace Uintah

#endif // end CORE_PARALLEL_PARALLEL_H
