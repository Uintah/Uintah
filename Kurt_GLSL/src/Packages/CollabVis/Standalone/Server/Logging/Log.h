/*
 *
 * Log: Provides logging capabilities
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __Log_h_
#define __Log_h_

#include <fstream>
#include <Thread/Mutex.h>

// Remove debugging variable....grrrr...
#ifdef DEBUG
#undef DEBUG
#endif

namespace SemotusVisum {
namespace Logging {

using namespace std;
//using namespace SemotusVisum::Thread;
using namespace SCIRun;

//////////
// Logging levels
typedef enum {
  ERROR,
  WARNING,
  MESSAGE,
  ENTER,
  LEAVE,
  DEBUG

} logLevel;

//////////
// Default logfile name.
const char * const DEFAULT_LOGFILE  = "Logfile.log";

//////////
// Default logging level.
const logLevel     DEFAULT_LOGLEVEL = DEBUG;

/**************************************
 
CLASS
   Log
   
KEYWORDS
   Log, Logging
   
DESCRIPTION

   Log provides a clean, abstract way to maintain a log file with different
   levels of logging.
   
****************************************/


class Log {
public:

  //////////
  // Closes the log file
  static void close();

  //////////
  // Sets the log file name and opens the logfile for writing.
  static void setLogFileName( const char * logFileName );

  //////////
  // Writes the given message to the logfile at the given logging
  // level IF messages at that level are being logged.
  static void log( logLevel level, const char * message );

  //////////
  // Writes the given binary data to the logfile at the given logging
  // level IF messages at that level are being logged.
  static void logBinary( logLevel level, const char * data,
			 int numBytes );
  //////////
  // Sets the new logging level.
  static void setLogLevel( logLevel level );
  
  
protected:
  //////////
  // Constructor
  Log() {}

  //////////
  // Destructor
  ~Log() {}
  
  static char * logStrings[];
  static ofstream output;
  static logLevel currentLogLevel;

  // Mutex to ensure we don't diddle on other threads' writes.
  static Mutex logMutex;
  
};

}
}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:12  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:25:58  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/10/11 16:38:07  luke
// Foo
//
// Revision 1.4  2001/05/29 03:09:45  luke
// Linux version mostly works - preparing to integrate IRIX compilation changes
//
// Revision 1.3  2001/05/01 21:39:04  luke
// Made Logging thread-safe
//
// Revision 1.2  2001/01/29 18:59:34  luke
// Commented Log
//
