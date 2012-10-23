/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
// 
// StackTrace.cc 
//
//   Used to display actual function names when looking at abridged
//   stack traces.  See README for more information.
//
// Author: J. Davison de St. Germain
// Date:   Feb. 27, 2008
// 
//

#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <vector>

using namespace std;

struct Info_S {
  string name;
  char   type;
  int    address;

  Info_S() { name = ""; type = 0; address = 0; }
};

vector< Info_S * > infoVector;

Info_S *
findLocation( int value ) 
{
  int low = 0;
  int high = infoVector.size() - 1;

  while( low <= high ) {
    int mid = (low + high) / 2;

    int addr = infoVector[mid]->address;

    if( addr > value ) {
      high = mid - 1;
    }
    else if( addr < value) {
      low = mid + 1;
    }
    else {
      return infoVector[mid];
    }
  }

  // See if it is 'close' to the address.
  if( low != 0 && low != infoVector.size() && value < infoVector[low]->address ) {
    return infoVector[ low-1 ];
  }

  return NULL;
}

void
readSymbolLocations( char * filename )
{
  FILE * fp = fopen( filename, "r" );

  if( fp == NULL ) {
    printf( "Error reading file '%s'.  Goodbye.\n\n", filename );
    exit( 1 );
  }

  int cnt = 0;

  while( !feof( fp ) ) {

    cnt++;
    if( cnt % 10000 == 0 ) {
      printf( "Number of symbols read: %d\n", cnt );
    }
    
    const int LINE_SIZE = 4000;
    char      line[ LINE_SIZE ];

    fgets( line, LINE_SIZE, fp );

    int maxLength = strlen( line );

    Info_S * info = new Info_S();

    char * name, * moreName, * address;

    name     = new char[ maxLength ];
    moreName = new char[ maxLength ];
    address  = new char[ maxLength ];

    // 'moreName' is used when the symbol name has a blank space
    // separated return value (otherwise we would just get 'void')...
    // (Actually, there are many symbols that have multiple blanks in them...
    //  we might want to be smarter about this... but by handling a single blank
    //  usually we get enough info that we don't need the entire symobl...)

    int numRead = sscanf( line, "%s %c %s %s", address, &info->type, name, moreName );

    if( numRead < 3 ) {
      printf( "Skipping invalid line. Reason: Failed to parse this line: '%s'\n", line );
      continue;
    }

    sscanf( address, "%x", &info->address );

    if( numRead == 4 ) {
      info->name = string( name ) + " " + moreName;
    }
    else {
      info->name = name;
    }

    if( info->address != 0 ) {
      infoVector.push_back( info );
    }
    delete [] name;
    delete [] address;
  }
  printf( "\nTotal Number of Symbols: %d\n\n", cnt );
}

void
printStackTrace( char * filename )
{
  FILE * fp = fopen( filename, "r" );

  if( fp == NULL ) {
    printf( "Error reading file '%s'.  Goodbye.\n\n", filename );
    exit( 1 );
  }

  vector< string > stacktrace;

  printf("stack trace (raw):\n\n");

  while( !feof( fp ) ) {
    const int LINE_SIZE = 4000;
    char      line[ LINE_SIZE ];

    char * result = fgets( line, LINE_SIZE, fp );

    if( result == NULL ) { 
      break;
    }
    printf("%s", line);
    stacktrace.push_back( line );
  }

  printf("\n\n");
  printf("stack trace (with names):\n\n");

  for( int pos = 0; pos < stacktrace.size(); ++pos ) {

    string line = stacktrace[ pos ];
    line[ line.length()-1 ] = 0; // remove \n from line
    int    loc  = line.rfind( "[0x" );

    if( loc == string::npos ) {
      printf( "Warning: Did not find valid address (0x...) in: '%s'\n", line.c_str() );
      continue;
    }

    string addrStr = line.substr( loc + 1 );
    int    addr = -1;
    int    numScanned = sscanf( addrStr.c_str(), "%x", &addr );

    if( addr == -1 || numScanned != 1 ) {
      printf( "Warning: Could not parse function address: '%s'\n", line.c_str() );
      continue;
    }

    Info_S * info;

    info = findLocation( addr ); //, infoVector.size()/2, 0 );
    string name = "unknown";
    if( info ) {
      name = info->name;
    }
    printf( "%x -- %s\n", addr, name.c_str() );
  }
  printf( "\n" );
}

#define DO_TESTING 0

#if DO_TESTING

// Get the stack trace and print it out.  1st in raw form, then with
// associated function names.
void
bar()
{
  void* callstack[128];
  int i, frames = backtrace(callstack, 128);
  char** strs = backtrace_symbols(callstack, frames);

  printf("stack trace (raw):\n\n");
  for (i = 0; i < frames; ++i) {
    printf("%d: %s\n", i, strs[i]);
  }
  printf("\n\n");

  printf("stack trace (with names):\n\n");
  for (i = 0; i < frames; ++i) {

    string line = strs[i];
    int    loc  = line.rfind( "[" );
    string addrStr = line.substr( loc + 1, 8 );

    int    addr;

    sscanf( addrStr.c_str(), "%x", &addr );

    Info_S * info;

    info = findLocation( addr ); //, infoVector.size()/2, 0 );
    string name = "unknown";
    if( info ) {
      name = info->name;
    }
    printf( "%x -- %s\n", addr, name.c_str() );
  }
  free(strs);
}

// Dummy functions  to create a deeper stack trace.
void
foo() 
{
  bar();
}

double
doit( int value ) 
{
  foo();
  return 1.0;
}

// Tests the binary search lookup with a number of addresses.
void
test() 
{
  printf( "\n\n" );
  vector<int> addresses;
  
  addresses.push_back( 0x400bbf );
  addresses.push_back( 0x400bc0 );
  addresses.push_back( 0x400bc1 );
  addresses.push_back( 0x400db0 );
  addresses.push_back( 0x400ddc );
  addresses.push_back( 0x401af1 );
  addresses.push_back( 0x401af2 );
  addresses.push_back( 0x401af3 );
  addresses.push_back( 0x504987 );
  addresses.push_back( 0x504988 );
  addresses.push_back( 0x504989 );

  for( int pos = 0; pos < addresses.size(); pos++ ) {
    int addr = addresses[pos];

    Info_S * info = findLocation( addr ); //, infoVector.size()/2, 0 );
    string name = "unknown";
    int    address = 0;
    if( info ) {
      name = info->name;
      address = info->address;
    }
    printf( "%x -- %x -- %s\n\n\n", addr, address, name.c_str() );
  }
} // end test()
#endif // DO_TESTING

void
usage( string arg = "" )
{
  if( arg != "" ) {
    printf( "\n" );
    printf( "Bad argument: %s\n", arg.c_str() );
  }
  printf( "\n" );
  printf( "Usage: StackTrace <symbol_file> <stack_trace_file>\n" );
  printf( "\n" );
  printf( "       Reads in function symbols from 'symbol_file'.  ('Symbol_file' must be sorted!)\n" );
  printf( "       Then reads in the stack trace from 'stack_trace_file', looks up the function\n" );
  printf( "       addresses, and displays their names.\n" );
  printf( "\n" );
  exit( 1 );
}

int
main( int argc, char *argv[] )
{
  if( argc != 3 ) {
    usage();
  }

  readSymbolLocations( argv[1] );

#if DO_TESTING
  doit(1);
#else
  printStackTrace( argv[2] );
#endif
}
