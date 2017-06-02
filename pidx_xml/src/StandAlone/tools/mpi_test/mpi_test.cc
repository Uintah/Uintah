/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
//
// mpi_test.cc
//
// Author: Justin Luitjens, J. Davison de St. Germain
//
// Date:   Mar. 2008
//

#include <Core/Parallel/UintahMPI.h>
#include <Core/Util/FileUtils.h>     // for testFilesystem()
#include <Core/Util/Timers/Timers.hpp>

#include <unistd.h> // for gethostname()

#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

using std::stringstream;
using std::cout;
using std::endl;
using std::vector;
using std::string;

///////////////////////////////////////////////////

const int HOST_NAME_SIZE = 100;
char      hostname[ HOST_NAME_SIZE ];
vector< string > hostnames;  // Only valid on rank 0
int       rank = -1;
int       procs = -1;

stringstream error_stream;

struct Args {
  bool testFileSystem;
  int  verbose;

  Args() :
    testFileSystem( true ),
    verbose( 0 )
  {}

} args;

///////////////////////////////////////////////////
// Pre-declarations of test functions

int allreduce_test();
int reduce_test();
int broadcast_test();
int allgather_test();
int gather_test();
int point2pointasync_test();
int point2pointsync_test();
int fileSystem_test();
int testme( int (*testfunc)(void),const char* name );

///////////////////////////////////////////////////

void
usage( const string & prog, const string & badArg )
{
  if( rank == 0 ) {
    cout << "\n";
    if( badArg != "" ) {
      cout << prog << ": Bad command line argument: '" << badArg << "'\n\n";
    }

    cout << "Usage: mpirun -np <number> mpi_test [options]\n";
    cout << "\n";
    cout << "       mpi_test runs a number of MPI calls attempting to verify\n";
    cout << "       that all nodes are up and running.  These tests include\n";
    cout << "       both synchronous and async point to point messages, broadcasts,\n";
    cout << "       reductions, and gathers.  Additionally, mpi_test will attempt\n";
    cout << "       to create and read files on the file system on each node\n";
    cout << "       (once per proc per node) to verify that the filesystem appears\n";
    cout << "       to be working on all nodes.\n";
    cout << "\n";
    cout << "       Note, on Inferno, if some of the tests fail and report the processor\n";
    cout << "       rank, you can look in the $PBS_NODEFILE for a list of processors.\n";
    cout << "       Rank corresponds to location in the file (ie, rank 0 is the first\n";
    cout << "       entry, rank 1 is the second entry, etc).\n";
    cout << "\n";
    cout << "       Options:\n";
    cout << "\n";
    cout << "         -nofs - Don't check filesystem.\n";
    cout << "         -v    - Be verbose!  (Warning, on a lot of processors this will\n";
    cout << "                   produce a lot of output! Also, this will initiate a number\n";
    cout << "                   of additional MPI calls (to send verbose information) which\n";
    cout << "                   could possibly hang and/or cause the outcome to be somewhat\n";
    cout << "                   different from the non-verbose execution.)\n";
    cout << "         -vv   - Be very verbose... see -v warning...\n";
    cout << "\n";
  }
  Uintah::MPI::Finalize();
  exit(1);
}

void
parseArgs( int argc, char *argv[] )
{
  for( int pos = 1; pos < argc; pos++ ) {
    string arg = argv[pos];
    if( arg == "-nofs" ) {
      args.testFileSystem = false;
    }
    else if( arg == "-v" ) {
      args.verbose = 1;
    }
    else if( arg == "-vv" ) {
      args.verbose = 2;
    }
    else {
      usage( argv[0], arg );
    }
  }
}

int
main( int argc, char* argv[] )
{
  gethostname( (char*)&hostname, HOST_NAME_SIZE );

  Uintah::MPI::Init( &argc, &argv );

  Uintah::MPI::Comm_rank( MPI_COMM_WORLD, &rank );
  Uintah::MPI::Comm_size( MPI_COMM_WORLD, &procs );

  parseArgs( argc, argv ); // Should occur after variables 'rank' and 'procs' set...

#if DO_DEBUG
  // Many times if there is a problem with a node, the Uintah::MPI::Init() call above will just hang.
  // If debugging, sometimes it is useful to print something out at this point to track
  // how many nodes have initialized... and to get an idea of how long it took.
  printf( "Finished Uintah::MPI::Init() on rank %d.\n", rank );
#endif

  if( rank == 0 ) {
    cout << "Testing mpi communication on " << procs << " processors.\n";
  }

  if( args.verbose ) { // Create 'rank to processor name' mapping:

    if( rank == 0 ) {
      hostnames.resize( procs ); // Reserve enough space for all the incoming names.

      hostnames[ 0 ] = hostname; // Save proc 0's name.

      for( int proc = 1; proc < procs; proc++ ) { // Get all the other procs names.
        char hnMessage[ HOST_NAME_SIZE ];
        std::fill(&hnMessage[0],&hnMessage[HOST_NAME_SIZE],'\0');
        MPI_Status status;

        Uintah::MPI::Recv( hnMessage, HOST_NAME_SIZE, MPI_CHAR, proc, 0, MPI_COMM_WORLD, &status );

        // int numBytesReceived = -1;
        // Uintah::MPI::Get_count( &status, MPI_CHAR, &numBytesReceived );
        // printf("result is %d, received bytes: %d\n", result, numBytesReceived);

        hostnames[ proc ] = hnMessage;
      }

      cout << "Machine rank to name mapping:\n";
      for( int r = 0; r < procs; r++ ) {
        cout << r << ": " << hostnames[ r ] << "\n";
      }
    }
    else {
      // everyone but rank 0 needs to send rank 0 a message with it's name...
      Uintah::MPI::Send( hostname, HOST_NAME_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD );
    }
  }
 
  // Run Point2PointASync_Test first, as it will hopefully tell us the
  // exact processor number (rank) if there is a problem...
  testme( point2pointasync_test,  "Point To Point ASync" );

  testme( allreduce_test,        "Uintah::MPI::Allreduce" );
  testme( reduce_test,           "Uintah::MPI::Reduce" );
  testme( broadcast_test,        "Uintah::MPI::Bcast" );
  testme( allgather_test,        "Uintah::MPI::Allgather" );
  testme( gather_test,           "Uintah::MPI::Gather" );
  testme( point2pointsync_test, "Point To Point Sync" );

  if( args.testFileSystem ) {
    testme( fileSystem_test,       "File System" );
  }
  
  Uintah::MPI::Finalize();
  return 0;
}

int
testme(int (*testfunc)(void),const char* name)
{
  if( rank == 0 ) {
    cout << "Testing '" << name << "': ";
    cout.flush();
  }

  Timers::Simple timer;

  timer.start();  
  int pass = testfunc();
  timer.stop();
  
  int all_pass = false;
  
  Uintah::MPI::Allreduce( &pass, &all_pass, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
 
  if( rank == 0) {

    if( all_pass ) {
      cout << "Passed" ;
    }
    else {
      cout << "Failed" ;
    }
    cout << " (Test took " << timer().seconds() << " seconds.)\n";
  }
  
  if( !all_pass ) {
    // Sync processors so output is in sync
    Uintah::MPI::Barrier(MPI_COMM_WORLD);
    cout << error_stream.str();
    cout.flush();
    error_stream.str("");
    // Sync processors so output is in sync
    Uintah::MPI::Barrier(MPI_COMM_WORLD);
  }
  return all_pass;
}

// Each Processor all reduces their rank
int
allreduce_test()
{
  int pass=true;
  int message;
  int n=procs-1;
  Uintah::MPI::Allreduce(&rank,&message,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  if( message != (n*(n+1))/2 ) {
    pass=false;
    error_stream << "     rank " << rank << ": Allreduce incorrect\n";
    //cout << "     rank " << rank << ": Allreduce incorrect\n";
  }
  return pass; 
}

// Each processor broadcasts their rank
int
broadcast_test()
{
  int pass = true;
  int message;
  for( int p=0;p<procs;p++ ) {
    message=rank;
    Uintah::MPI::Bcast(&message,1,MPI_INT,p,MPI_COMM_WORLD);
    if(message != p) {
      pass = false;
      error_stream << "     rank " << rank << ": Bcast from rank " << p << " incorrect\n";
    }
  }
  return pass;
}

// Each processor reduce sums their rank
int
reduce_test()
{
  int pass = true;
  int message;
  int n = procs - 1;

  for( int p = 0; p < procs; p++ ) {
    message=rank;
    Uintah::MPI::Reduce(&rank,&message,1,MPI_INT,MPI_SUM,p,MPI_COMM_WORLD);
    
    if( p == rank && message != (n*(n+1))/2 ) {
      pass = false;
      error_stream << "     rank " << rank << ": Reduce on rank " << p << " incorrect\n";
    }
  }
  return pass;
}

// Each Processor allgathers its rank
int
allgather_test()
{
  int         pass = true;
  vector<int> message(procs,0);
  
  Uintah::MPI::Allgather(&rank,1,MPI_INT,&message[0],1,MPI_INT,MPI_COMM_WORLD);

  for( int p = 0; p < procs; p++ ) {
    if( message[p] != p ) {
      pass = false;
      error_stream << "     rank " << rank << ": Allgather entry from " << p << " is invalid\n";
    }
  }
  return pass; 
}

// Each Processor gathers its rank
int
gather_test()
{
  int pass = true;
  
  for( int p=0; p < procs; p++ ) {
    vector<int> message(procs,0);
    
    Uintah::MPI::Gather(&rank,1,MPI_INT,&message[0],1,MPI_INT,p,MPI_COMM_WORLD);

    if( rank == p ) {
      for( int p = 0; p < procs; p++ ) {
        if( message[p] != p ) {
          pass=false;
          error_stream << "     rank " << rank << ": gather entry from " << p << " is invalid\n";
        }
      }
    }
  }
  return pass; 
}

// Each Processor sends its rank to each other processor
int
fileSystem_test()
{
  int pass = true;

  string host = string( hostname ).substr( 0, 3 );

  if( host == "inf" ) {
    // On inferno, test the raid disks, and the FS the code is being run from...
    bool raid1 = Uintah::testFilesystem( "/usr/csafe/raid1", error_stream, rank );
    bool raid2 = Uintah::testFilesystem( "/usr/csafe/raid2", error_stream, rank );
    bool raid3 = Uintah::testFilesystem( "/usr/csafe/raid3", error_stream, rank );
    bool raid4 = Uintah::testFilesystem( "/usr/csafe/raid4", error_stream, rank );
    bool home  = Uintah::testFilesystem( ".",                error_stream, rank );
    pass = raid1 && raid2 && raid3 && raid4 && home;
  } 
  else {
    // On other systems, (at least for now) just check the file system of the current dir.
    pass = Uintah::testFilesystem( ".", error_stream, rank );
  }
  
  if( args.verbose ) {
    if( rank == 0 ) { 

      cout << "\n";
      cout << "   Print outs in the form of '.name (rank).' correspond to processors that have successfully\n";
      cout << "   completed the test.  '<name (-rank)>' correspond to processors that failed file system check.\n";
      cout << "\n";

      vector<int>    messages(procs-1);
      MPI_Request  * rrequest = new MPI_Request[ procs-1 ];

      for( int proc = 1; proc < procs; proc++ ) {    
        Uintah::MPI::Irecv( &messages[proc-1], 1, MPI_INT, proc, proc, MPI_COMM_WORLD, &rrequest[proc-1] );
      }
      bool done = false;
      int  totalCompleted = 0;

      Timers::Simple timer;
      timer.start();

      int          totalPassed = (int)pass, totalFailed = (int)(!pass);
      int          numCompleted = -1;
      int        * completedBuffer = new int[ procs-1 ]; // Passed to MPI
      MPI_Status * status = new MPI_Status[ procs-1 ];

      usleep(1000000);//testing

      const double totalSecsToWait = 5 * 60; // 5 mintues
      int          generation = 1;

      while( !done ) {

        usleep( 100000 ); // Wait a .1 sec for messages to come in

        // See if any processors have reported their status...
        //
        Uintah::MPI::Testsome( procs-1, rrequest, &numCompleted, completedBuffer, status );

        if( numCompleted > 0 ) {
          for( int pos = 0; pos < numCompleted; pos++ ) {
            if( messages[completedBuffer[ pos ]] > 0 ) {
              cout << "." << hostnames[completedBuffer[pos]] << " (" << messages[completedBuffer[pos]] << ").";
              totalPassed++;
            } else { // failed
              cout << "<" << hostnames[completedBuffer[pos]] << " (" << messages[completedBuffer[pos]] << ")>";
              totalFailed++;
            }
            cout.flush();
          }
        }
        
        totalCompleted += numCompleted;
        if( totalCompleted == (procs-1) ) {
          cout << "\n\n";
          done = true;
        }
        else {
          const double secsToWait = 30.0;
	    
	  // Give it 'secsToWait' seconds, then print some info          
	  if( timer().seconds() > secsToWait*generation ) {
	    if( rank == 0 ) {
              cout << "\nWarning: Some processors have not responded after "
		   << generation * secsToWait << " seconds.\n"
                   << "           Continuing to wait...  "
		   << "Number of processors that have responded: " 
                   << totalPassed + totalFailed << " of " << procs-1 << ".\n";
              generation++;
            }
	  }
	  
	  // Give it 'totalSecsToWait' seconds to finish completely	
	  if( timer().seconds() > totalSecsToWait ) {
            done = true;
          }
        }
      } // end while (!done)

      if( rank == 0 ) {
        cout << "Total number of processors reporting FS check success: " << totalPassed << ".\n";
        cout << "Total number of processors reporting FS check failure: " << totalFailed << ".\n";
      }

      // Clean up memory
      delete [] completedBuffer;
      delete [] status;
      delete [] rrequest;
    }
    else {

      MPI_Request request;    
      // Tell rank 0 that we have succeeded or failed (-rank).
      int data = pass ? rank : -rank;

      Uintah::MPI::Isend( &data, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &request );
    }
  } // end if verbose

  return pass;
}

// Each Processor sends its rank to each other processor
int
point2pointasync_test()
{
  int                 pass = true;
  vector<int>         messages(procs);
  MPI_Request * srequest, * rrequest;
  
  srequest = new MPI_Request[ procs ];
  rrequest = new MPI_Request[ procs ];

  int data = rank;

  for( int p = 0; p < procs; p++ ) {

    //if( rank > 4 ) { // Slow things down just a little for testing purposes...
    //  usleep( 100000 * rank ); 
    //}

    //start send
    Uintah::MPI::Isend( &data, 1, MPI_INT, p, p, MPI_COMM_WORLD, &srequest[p] );
    
    //start recv
    Uintah::MPI::Irecv( &messages[p], 1, MPI_INT, p, rank, MPI_COMM_WORLD, &rrequest[p] );

    //if( rank == 5 ) { // Simulate proc 5 sending bad info....
    //  data = 99;
    //}

    //if( rank == 5 ) { // Simulate proc 5 broken....
    //  while(1) {}
    //}
  }

  int          numCompleted = -1;
  int        * completedBuffer = new int[ procs ]; // Passed to MPI
  bool       * completed       = new bool[ procs ]; // Used to keep track over all
  MPI_Status * status = new MPI_Status[ procs ];

  for( int pos = 0; pos < procs; pos++ ) {
    completed[ pos ] = false;
    // The following don't need to be initialized as they are overwritten... 
    // however, when debugging, it is sometimes useful to set values to
    // more easily see what has changed:
    //completedBuffer[ pos ] = -9;
    //status[ pos ].MPI_SOURCE = -2;
    //status[ pos ].MPI_TAG    = -3;
    //status[ pos ].MPI_ERROR  = -4;
  }

  bool done = false;
  int  totalCompleted = 0;

  Timers::Simple timer;
  timer.start();
  
  while( !done ) {

    // While it is unclear in the docs, apparently the Uintah::MPI::Testsome
    // not only tests for the data having arrived, but handles the
    // recv too (ie, places the data in the specified buffer).
    Uintah::MPI::Testsome( procs, rrequest, &numCompleted, completedBuffer, status );

    //reset timer if progress is made
    if(numCompleted>0)
      timer.reset( true );
    
    double secsToWait = 100.0;
    
    // Give it 'secsToWait' seconds to finish
    if( timer().seconds() > secsToWait ) {
      cout << "Proc " << rank << ": No progress has been made in the last "
	   << timer().seconds() << " seconds.\n";
 
      // Find out (and display) which processors did not successfully respond...
      for( int pos = 0; pos < procs; pos++ ) {
        if( completed[ pos ] == false ) {
          cout << "Proc " << rank << ": failed to hear from processor "
	       << pos << ".\n";
          pass = false;
        }
      }
      done = true;
    }

    // Record the fact that a receive has completed.
    if( numCompleted > 0 ) {
      for( int pos = 0; pos < numCompleted; pos++ ) {
        completed[ completedBuffer[ pos ] ] = true;
      }
    }

    totalCompleted += numCompleted;
    if( totalCompleted == procs ) {

      // All messages in... verify data is valid:

      for( int pos = 0; pos < procs; pos++ ) {

        if( messages[pos] != pos ) {
          pass = false;
          error_stream << "     rank " << rank << ": data in point to point async message from " << pos 
                       << " is invalid\n";
        }
      }
      done = true;
    }
  }

  delete [] status;
  delete [] completed;
  delete [] completedBuffer;
  delete [] srequest;
  delete [] rrequest;

  return pass;
}

// Each Processor gathers its rank
// ...this should probably be improved to work in parallel but right now it is done sequentially

int
point2pointsync_test()
{
  int pass = true;
  int message;

  if( rank == 0 && ( args.verbose > 1 )) {
    cout << "\nBeginning point 2 point sync tests...\n\n";
  }

  for( int pp = 0; pp < procs; pp++ ) {
    if( pp == rank ) { // sending
      for( int p = 0; p < procs; p++ ) {
        if( p != pp ) { // Don't send to our self...
          Uintah::MPI::Send( &rank, 1, MPI_INT, p, p, MPI_COMM_WORLD );
          if( rank == 0 && ( args.verbose > 1 )) {
            cout << "Proc 0 finished Uintah::MPI::Send to rank: " << p << "\n";
          }
        }
      }
    }
    else { // recieving
      MPI_Status status;
      message=-1;
      Uintah::MPI::Recv(&message,1,MPI_INT,pp,rank,MPI_COMM_WORLD,&status);
      if( rank == 0 && ( args.verbose > 1 ) ) {
        cout << "Proc 0 just Uintah::MPI::Recv'd from rank: " << pp << "\n";
      }
      if( message != pp ) {
        pass = false;
        error_stream << "     rank " << rank << ": point to point sync from  " << pp << " is invalid\n";
      }
    }
  }
  return pass;
}
