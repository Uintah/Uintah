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

#include <Logging/Log.h>
#include <Malloc/Allocator.h>
#include <iostream>

#define ENTER_LEAVE 1

namespace SemotusVisum {
namespace Logging {

logLevel
Log::currentLogLevel = DEFAULT_LOGLEVEL;

char *
Log::logStrings[] = { "Error  : ",
		      "Warning: ",
		      "Message: ",
		      "Enter  : " ,
		      "Leave  : ",
		      "Debug  : " };

ofstream
Log::output;

Mutex
Log::logMutex("Logging Mutex");

void
Log::close() {
  std::cerr << "Closing Logfile!" << endl;
  logMutex.lock();
  output.close();
  logMutex.unlock();
}

void
Log::setLogFileName( const char * logFileName ) {
  
  if ( output.is_open() )
    close();
  logMutex.lock();
  output.open( logFileName );
  logMutex.unlock();
}

void
Log::log( logLevel level, const char * message ) {
  
  
  /* If the output logfile is closed, open it with the default
     logfile name. */

  if ( output.is_open() == false )
    setLogFileName( DEFAULT_LOGFILE );

  /* If we are logging messages at this level, write the message to 
     the logfile */
  if ( level <= currentLogLevel ) {
    logMutex.lock();
    output << logStrings[ level ] << " " << message << endl;
    //std::cout << logStrings[ level ] << " " << message << endl;
    logMutex.unlock();
  }

  /* Otherwise, do nothing. */
}

void
Log::logBinary( logLevel level, const char * data, int numBytes ) {

  /* If the output logfile is closed, open it with the default
     logfile name. */

  if ( output.is_open() == false )
    setLogFileName( DEFAULT_LOGFILE );

  /* If we are logging messages at this level, write the message to 
     the logfile */
  if ( level <= currentLogLevel ) {
    logMutex.lock();

    output << logStrings[ level ] << " ";
    output.write( data, numBytes );
    output << endl;
    
    logMutex.unlock();
  }

  /* Otherwise, do nothing. */
}

void
Log::setLogLevel( logLevel level ) {

  logMutex.lock();
  /* If the new logging level is valid, switch levels. */
  if ( level <= DEBUG &&
       level >= ERROR ) {

    if ( level == ENTER || level == LEAVE )
    {
#ifdef ENTER_LEAVE
      
      currentLogLevel = level;

      output << "Switching to log level " << logStrings[ level ] <<
      endl;
#endif
    }
    else
    {
      currentLogLevel = level;

      output << "Switching to log level " << logStrings[ level ] <<
      endl;
    }
  }
  logMutex.unlock();
}


}
}
