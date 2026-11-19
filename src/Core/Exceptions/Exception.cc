/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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


/*
 *  Exception.h: Base exception class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 */

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/Assert.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

#include <unistd.h>

#ifdef HAVE_EXC
#  include <libexc.h>
#elif (defined(__GNUC__) && defined(__linux))
#  include <execinfo.h>
#  include <cxxabi.h>
#  include <dlfcn.h>
#endif

#if defined(_AIX)
// Needed for strcasecmp on aix 4.3 (on 5.1 we don't need this.)
// currently blue is 4.3.
#  include <strings.h>
#endif

namespace Uintah {

using namespace std;
using namespace Uintah;

Exception::Exception(bool ignoreWait)
{
  // Save the return addresses only.  stackTrace() turns them into text
  // (function names, file:line) when the backtrace is actually requested.
#if !defined(HAVE_EXC) && defined(__GNUC__) && defined(__linux)
  naddresses_ = backtrace( addresses_, MAXSTACK );
#else
  naddresses_ = 0;
  stacktrace_ = getStackTrace();
#endif
  if(!ignoreWait) 
    WAIT_FOR_DEBUGGER(true);
}

Exception::~Exception()
{
}

// This is just to fool the compiler so that it will not complain about
// "loop expressions are constant"  (SGI mipspro)  See Exception.h for
// use - Steve.
bool Exception::alwaysFalse()
{
  return false;
}


void Exception::sci_throw(const Exception& exc)
{
  // This is a function invoked by the SCI_THROW macro.  It
  // can be useful for tracking down fatal errors, since the
  // normal exception mechanism will unwind the stack before
  // you can get anything useful out of it.

  // Set this environment variable if you want to have a default
  // response to the question below.  Value values are:
  // ask, dbx, cvd, throw, abort
  const char* emode = getenv("SCI_EXCEPTIONMODE");
  if(!emode)
    emode = "dbx"; // Default exceptionmode

  // If the mode is not "throw", we print out a message
  if(strcasecmp(emode, "throw") != 0){
    cerr << "\n\nAn exception was thrown.  Msg: " << exc.message() << "\n";
    cerr << exc.stackTrace();
    // Print out the exception type (clasname) and the message
    cerr << "\nException type: " << exc.type() << '\n';
    cerr << "Exception message: " << exc.message() << '\n';
  }
  // See what we should do
  for(;;){
    if(strcasecmp(emode, "ask") == 0){
      // Ask the user
      cerr << "\nthrow(t)/dbx(d)/cvd(c)/abort(a)? ";
      emode=0;
      while(!emode){
        char action;
        char buf[100];
        while(read(fileno(stdin), buf, 100) <= 0){
          if(errno != EINTR){
            cerr <<  "\nCould not read response, throwing exception\n";
            emode = "throw";
            break;
          }
        }
        action=buf[0];
        switch(action){
        case 't': case 'T':
          emode="throw";
          break;
        case 'd': case 'D':
          emode="dbx";
          break;
        case 'c': case 'C':
          emode="cvd";
          break;
        case 'a': case 'A':
          emode="abort";
          break;
        default:
          break;
        }
      }
    }

    if(strcasecmp(emode, "throw") == 0) {
      // We cannot throw from here, so we just return and the
      // exception will be thrown by the SCI_THROW macro
      return;
    } else if(strcasecmp(emode, "dbx") == 0){
      // Fire up the debugger
      char command[100];
      if(getenv("SCI_DBXCOMMAND")){
        sprintf(command, getenv("SCI_DBXCOMMAND"), getpid());
      } else {
#ifdef HAVE_EXC
        sprintf(command, "winterm -c dbx -p %d &", getpid());
#else
        sprintf(command, "xterm -e gdb %d&", getpid());
#endif
      }
      cerr << "Starting: " << command << '\n';
      system(command);
      emode="ask";
    } else if(strcasecmp(emode, "cvd") == 0){
      // Fire up the slow, fancy debugger
      char command[100];
      sprintf(command, "cvd -pid %d &", getpid());
      cerr << "Starting: " << command << '\n';
      system(command);
      emode="ask";
    } else if(strcasecmp(emode, "abort") == 0){
      // This will trigger the thread library, but we cannot
      // directly call the thread library here or it would create
      // a circular dependency
      abort();
    } else {
      cerr << "Unknown exception mode: " << emode << ", aborting\n";
      abort();
    }
  }
}

#if !defined(HAVE_EXC) && defined(__GNUC__) && defined(__linux)
//______________________________________________________________________
//______________________________________________________________________
//  Stack trace helpers, used only by getStackTrace():
//
//    Source line lookup (adds "file:line" to each frame):
//      lookupSymbol()           dladdr() wrapper
//      objectPath()             file addr2line should read (handles "sus" found through PATH)
//      isPositionIndependent()  shared library / PIE or fixed-address executable
//      addr2lineAddress()       address to pass to addr2line
//      hexAddress()             " 0x..." argument
//      readLine()               one line of addr2line output
//      knownValue()             "??" (unknown) -> ""
//      eraseFrom()              strip " (discriminator N)"
//      runAddr2line()           one addr2line call per executable/library
//      getSourceLines()         "file:line" and function name of every frame
//
//    Formatting simplification (makes the backtrace easier to read):
//      baseName()               "/a/b/libX.so" -> "libX.so"
//      libraryName()            library a frame is in, without the directory
//      relativeSourcePath()     "<build dir>/../../src/X.cc:1" -> "../../src/X.cc:1"
//      replaceAll()             std::__cxx11::basic_string<...> -> std::string
//      demangle()               C++ symbol -> readable name
//      isOperatorCall()         don't mistake "operator()" for the argument list
//      findClosingParen()       end of the argument list
//      shortenArguments()       "f(int, double) const" -> "f(...) const"
//      frameFunctionName()      best available name for a frame
//      isExceptionFrame()       frame of the exception itself (printing starts after the last one)
//      printFrame()             write one frame
//      formatStackTrace()       return addresses -> readable backtrace text
//______________________________________________________________________
//______________________________________________________________________

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  dladdr() for one address.  When the address is not found all fields
//  are nullptr, so callers only need to check the field they use.
static Dl_info
lookupSymbol( void * address )
{
  Dl_info info{};
  if( dladdr( address, &info ) == 0 ) {
    info = Dl_info{};
  }
  return info;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  The file addr2line should read for a frame.  For the main program
//  dladdr() reports the name it was started with (e.g. "sus" when it was
//  found through PATH), which may not exist in the current directory; use
//  the real path of this executable then.  It must be resolved here: in the
//  addr2line process, /proc/self/exe would be addr2line itself.
static string
objectPath( const Dl_info & info )
{
  string path = info.dli_fname;
  if( access( path.c_str(), R_OK ) == 0 ) {
    return path;
  }

  char exe[4096];
  ssize_t len = readlink( "/proc/self/exe", exe, sizeof(exe) );
  if( len <= 0 ) {
    return path;
  }

  string exePath( exe, len );
  return exePath;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  Returns true if the ELF object (shared library or PIE executable) is
//  position independent.  addr2line then needs the offset from the load
//  address rather than the absolute address.
static bool
isPositionIndependent( const char * filename )
{
  FILE * fp = fopen( filename, "rb" );
  if( fp == nullptr ) {
    return true;
  }

  unsigned char header[18] = {0};
  size_t nRead = fread( header, 1, sizeof(header), fp );
  fclose( fp );

  if( nRead != sizeof(header) ) {
    return true;
  }

  unsigned int e_type = header[16] | ( header[17] << 8 );   // little endian
  return ( e_type == 3 );                                     // ET_DYN
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  The address addr2line needs for one return address.  isPIC caches
//  isPositionIndependent() for each executable/library.
static uintptr_t
addr2lineAddress( void                 * address,
                  const Dl_info        & info,
                  const string         & object,
                  map< string, bool >  & isPIC )
{
  if( isPIC.count( object ) == 0 ) {
    isPIC[object] = isPositionIndependent( object.c_str() );
  }

  // A return address points just after the call: subtract 1 to get the call's line
  uintptr_t addr = (uintptr_t)address - 1;
  if( isPIC[object] ) {
    addr -= (uintptr_t)info.dli_fbase;
  }
  return addr;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  " 0x<address>", an argument for addr2line.
static string
hexAddress( uintptr_t addr )
{
  char hex[32];
  snprintf( hex, sizeof(hex), " 0x%lx", (unsigned long)addr );
  return hex;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  Read one line from a pipe (addr2line output), without the trailing newline.
static bool
readLine( FILE   * pipe,
          string & line )
{
  char   * buf  = nullptr;
  size_t   size = 0;
  ssize_t  len  = getline( &buf, &size, pipe );

  if( len < 0 ) {
    free( buf );
    return false;
  }
  line.assign( buf, len );
  free( buf );

  // getline() succeeded, so line has at least one character
  if( line.back() == '\n' ) {
    line.pop_back();
  }
  return true;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  addr2line prints "??", "??:0" or "??:?" when it does not know; return "" then.
static string
knownValue( const string & value )
{
  if( value.compare( 0, 2, "??" ) == 0 ) {
    return "";
  }
  return value;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  Remove everything from marker to the end, e.g. " (discriminator 3)".
static string
eraseFrom( const string & value,
           const string & marker )
{
  string trimmed = value.substr( 0, value.find( marker ) );
  return trimmed;
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  Run addr2line once for one executable/library.  For each address it
//  prints two lines: the (demangled) function name, then "file:line".
//  The results go into functions[i] and lines[i] for each frame index i.
static void
runAddr2line( const string        & object,
              const string        & addresses,
              const vector<int>   & frameIndices,
              vector<string>      & functions,
              vector<string>      & lines )
{
  string cmd = "addr2line -f -C -e '" + object + "'" + addresses + " 2>/dev/null";
  FILE * pipe = popen( cmd.c_str(), "r" );
  if( pipe == nullptr ) {
    return;
  }

  for( int i : frameIndices ) {
    string function;
    string line;
    if( !readLine( pipe, function ) ) {
      break;
    }
    if( !readLine( pipe, line ) ) {
      break;
    }
    functions[i] = knownValue( function );
    lines[i]     = knownValue( eraseFrom( line, " (discriminator" ) );
  }
  pclose( pipe );
}

//______________________________________________________________________
//  Stack trace helper - source line lookup.
//  Use addr2line to find the source "file:line" and the function name of
//  each return address.  Returns one string per address in lines and in
//  functions, empty when it is not known (no debug information, or
//  addr2line is not installed).
//  Set the environment variable SCI_STACKTRACE_NOLINES to skip the lookup.
static vector<string>
getSourceLines( void * const   * addresses,
                int              n,
                vector<string> & functions )
{
  vector<string> lines( n );
  functions.assign( n, "" );

  if( getenv( "SCI_STACKTRACE_NOLINES" ) != nullptr ) {
    return lines;
  }

  // Group the addresses by the executable/library they are in
  map< string, vector<int> > frames;
  map< string, string >      offsets;
  map< string, bool >        isPIC;

  for( int i = 0; i < n; i++ ) {
    Dl_info info = lookupSymbol( addresses[i] );
    if( info.dli_fname == nullptr ) {
      continue;
    }
    string object = objectPath( info );
    frames[object].push_back( i );
    offsets[object] += hexAddress( addr2lineAddress( addresses[i], info, object, isPIC ) );
  }

  for( auto & f : frames ) {
    runAddr2line( f.first, offsets[f.first], f.second, functions, lines );
  }
  return lines;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  "/a/b/libX.so" -> "libX.so".  Only slashes before position "end" count.
static string
baseName( const string & path,
          size_t         end = string::npos )
{
  size_t slash = path.find_last_of( '/', end );
  if( slash == string::npos ) {
    return path;
  }
  string name = path.substr( slash + 1 );
  return name;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  The executable/library a frame is in, without the directory ("" if unknown).
static string
libraryName( const Dl_info & info )
{
  if( info.dli_fname == nullptr ) {
    return "";
  }
  string name = baseName( info.dli_fname );
  return name;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  "<build dir>/../../src/X.cc:123" -> "../../src/X.cc:123"
static string
relativeSourcePath( const string & source )
{
  size_t up = source.find( "/../" );
  if( up == string::npos ) {
    return source;
  }
  string relative = source.substr( up + 1 );
  return relative;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  Replace every occurrence of "from" with "to".
static string
replaceAll( string         value,
            const string & from,
            const string & to )
{
  size_t pos = value.find( from );
  while( pos != string::npos ) {
    value.replace( pos, from.size(), to );
    pos = value.find( from, pos + to.size() );
  }
  return value;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  Demangle a C++ symbol.  C symbols (e.g. __libc_start_main) are returned
//  unchanged; a null symbol gives "".
static string
demangle( const char * symbol )
{
  if( symbol == nullptr ) {
    return "";
  }

  int stat;
  char * demangled = abi::__cxa_demangle( symbol, 0, 0, &stat );
  if( demangled == nullptr ) {
    return symbol;
  }

  string name = demangled;
  free( demangled );
  return name;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  True if the '(' at position "open" is part of the name "operator()".
static bool
isOperatorCall( const string & function,
                size_t         open )
{
  if( open == string::npos ) {
    return false;
  }
  if( open < 8 ) {
    return false;
  }
  bool isOperator = ( function.compare( open - 8, 10, "operator()" ) == 0 );
  return isOperator;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  Position of the ')' that matches the '(' at "open", or npos.
static size_t
findClosingParen( const string & value,
                  size_t         open )
{
  if( open == string::npos ) {
    return string::npos;
  }

  int depth = 0;
  for( size_t p = open; p < value.size(); p++ ) {
    if( value[p] == '(' ) {
      depth++;
    }
    if( value[p] == ')' ) {
      depth--;
    }
    if( depth == 0 ) {
      return p;
    }
  }
  return string::npos;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  Replace a function's argument list with "(...)" to keep the backtrace
//  readable:  "ns::f(int, double) const" -> "ns::f(...) const".
//  The file:line printed with each frame identifies overloads.
static string
shortenArguments( const string & function )
{
  size_t open = function.find( '(' );
  if( isOperatorCall( function, open ) ) {
    open = function.find( '(', open + 2 );
  }

  size_t close = findClosingParen( function, open );
  if( close == string::npos ) {
    return function;
  }
  if( close == open + 1 ) {
    return function;                         // no arguments: keep "()"
  }
  string shortened = function.substr( 0, open ) + "(...)" + function.substr( close + 1 );
  return shortened;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  The best available name for a frame, in order of preference:
//    1) the exported symbol (demangled)
//    2) the name from the debug information (e.g. main, which is not exported)
//    3) the raw backtrace_symbols() text, without the directory
static string
frameFunctionName( const Dl_info & info,
                   const string  & debugName,
                   const char    * rawSymbol )
{
  string function = demangle( info.dli_sname );

  if( function.empty() ) {
    function = debugName;
  }
  if( function.empty() ) {
    string raw  = rawSymbol;
    string name = baseName( raw, raw.find( '(' ) );
    return name;
  }

  function = shortenArguments( function );
  function = replaceAll( function,
                         "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >",
                         "std::string" );
  return function;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  True for a frame that belongs to the exception machinery itself
//  (Exception constructors, getStackTrace()).  Uses the source file when it
//  is known, which works for static and shared builds; otherwise the library.
static bool
isExceptionFrame( const string & source,
                  const string & object )
{
  if( !source.empty() ) {
    bool inExceptionSource = ( source.find( "/Core/Exceptions/" ) != string::npos );
    return inExceptionSource;
  }

  bool inExceptionLibrary = ( object == "libCore_Exceptions.so" );
  return inExceptionLibrary;
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  Write one frame:
//    #4  Uintah::ICE::actuallyComputeStableTimestep(...)
//          ../../src/CCA/Components/ICE/ICE.cc:2237   (libCCA_Components_ICE.so)
static void
printFrame( ostream      & out,
            int            frame,
            const string & function,
            const string & source,
            const string & object )
{
  out << "  #" << frame << "  " << function << '\n';

  if( !source.empty() ) {
    out << "        " << source << "   (" << object << ")\n";
    return;
  }
  if( !object.empty() ) {
    out << "        (" << object << ")\n";
  }
}

//______________________________________________________________________
//  Stack trace helper - formatting simplification.
//  Turn saved return addresses into the readable backtrace text.  The
//  frames up to and including the exception's own (constructors,
//  getStackTrace()) are skipped; the message already says where it was thrown.
static string
formatStackTrace( void * const * addresses,
                  int            n )
{
  ostringstream out;
  if( n < 1 ) {
    out << "Backtrace not available!\n";
    string text = out.str();
    return text;
  }

  out << "Backtrace (pid " << getpid() << "):\n";
  char ** names = backtrace_symbols( addresses, n );
  vector<string> debugNames;
  vector<string> sourceLines = getSourceLines( addresses, n, debugNames );

  // Pass 1: library and source of every frame.  Printing starts after the
  // last frame of the exception itself; that also hides frames of tools
  // that wrap backtrace() (e.g. AddressSanitizer's interceptor) above it.
  vector<Dl_info> infos( n );
  vector<string>  objects( n );
  vector<string>  sources( n );
  int firstFrame = 0;

  for( int i = 0; i < n; i++ ) {
    infos[i]   = lookupSymbol( addresses[i] );
    objects[i] = libraryName( infos[i] );
    sources[i] = relativeSourcePath( sourceLines[i] );

    if( isExceptionFrame( sources[i], objects[i] ) ) {
      firstFrame = i + 1;
    }
  }

  // Pass 2: print the frames from where the exception was thrown
  int frame = 0;
  for( int i = firstFrame; i < n; i++ ) {
    frame++;
    string function = frameFunctionName( infos[i], debugNames[i], names[i] );
    printFrame( out, frame, function, sources[i], objects[i] );
  }
  free( names );

  string text = out.str();
  return text;
}
#endif

//______________________________________________________________________
//  The backtrace text, built from the saved return addresses the first
//  time it is requested (see Exception.h).
const char*
Exception::stackTrace() const
{
#if !defined(HAVE_EXC) && defined(__GNUC__) && defined(__linux)
  if( stacktrace_.empty() ) {
    stacktrace_ = formatStackTrace( addresses_, naddresses_ );
  }
#endif
  return stacktrace_.c_str();
}

string getStackTrace(void* context /*=0*/)
{
  ostringstream stacktrace;
#if defined(HAVE_EXC) || (defined(__GNUC__) && defined(__linux))
  static const int MAXSTACK = 100;
#endif

#ifdef HAVE_EXC
  // Use -lexc to print out a stack trace
  static const int MAXNAMELEN = 1000;
  __uint64_t addrs[MAXSTACK];
  char* cnames_str = new char[MAXSTACK*MAXNAMELEN];
  char* names[MAXSTACK];
  for(int i=0;i<MAXSTACK;i++)
    names[i]=cnames_str+i*MAXNAMELEN;
  int nframes = trace_back_stack(0, addrs, names, MAXSTACK, MAXNAMELEN);
  if(nframes == 0){
    stacktrace << "Backtrace not available!\n";
  } else {
    stacktrace << "Backtrace:\n";
    stacktrace.flags(ios::hex);
    // Skip the first procedure (us)
    for( int i = 1; i < nframes; i++ ) {
      stacktrace << "0x" << (void*)addrs[i] << ": " << names[i] << '\n';
    }
  }

#elif defined(__GNUC__) && defined(__linux)
  // Local buffer, so threads that throw at the same time do not overwrite each other
  void *addresses[MAXSTACK];
  int n = backtrace( addresses, MAXSTACK );
  stacktrace << formatStackTrace( addresses, n );
#endif
  return stacktrace.str();
}

static bool wait_for_debugger = false;

void
TURN_ON_WAIT_FOR_DEBUGGER()
{
  wait_for_debugger = true;
}

void
TURN_OFF_WAIT_FOR_DEBUGGER()
{
  wait_for_debugger = false;
}

void
WAIT_FOR_DEBUGGER(bool useFlag)
{ 
  if(useFlag && !wait_for_debugger ) {
    return;
  }
  bool wait=true; 
  char hostname[100]; 
  gethostname(hostname,100); 
  printf( "Host %s (PID: %d) waiting for debugger\n", hostname, getpid() ); 
  while( wait )
  {
    sleep(1);
  }; 
}

} // End namespace Uintah
