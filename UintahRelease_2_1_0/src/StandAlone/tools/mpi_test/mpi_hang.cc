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
// mpi_hang.cc
//
// Author: J. Davison de St. Germain
//
// Date:   July. 2013
//

#include <Core/Parallel/UintahMPI.h>

#include <stdlib.h>

#include <iostream>
#include <vector>

using namespace std;

///////////////////////////////////////////////////
// Global Variables

int       rank_g  = -1;
int       procs_g = -1;

///////////////////////////////////////////////////

void
usage( const string & prog, const string & badArg )
{
  if( rank_g == 0 ) {
    cout << "\n";
    if( badArg != "" ) {
      cout << prog << ": Bad command line argument: '" << badArg << "'\n\n";
    }

    cout << "Usage: mpirun -np <number> mpi_hang [options]\n";
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
    if( arg == "-v" ) {
      cout << "Don't know how to handle -v yet...\n";
    }
    else {
      usage( argv[0], arg );
    }
  }
}

int
main( int argc, char* argv[] )
{
  Uintah::MPI::Init( &argc, &argv );

  Uintah::MPI::Comm_rank( MPI_COMM_WORLD, &rank_g );
  Uintah::MPI::Comm_size( MPI_COMM_WORLD, &procs_g );

  parseArgs( argc, argv ); // Should occur after variables 'rank' and 'procs' set...

  if( rank_g == 0 ) {
    cout << "Testing to see if mpi hangs on a single processor abort.  (Running with " << procs_g << " processors.)\n";
    Uintah::MPI::Abort( MPI_COMM_WORLD, 1 );
  }

  cout << rank_g << ": Calling Uintah::MPI::Allgather.\n";

  vector<int> message( procs_g, 0 );

  Uintah::MPI::Allgather( &rank_g, 1, MPI_INT, &message[0], 1, MPI_INT, MPI_COMM_WORLD );

  cout << rank_g << ": All done.\n";
  
  Uintah::MPI::Finalize();
  return 0;
}

