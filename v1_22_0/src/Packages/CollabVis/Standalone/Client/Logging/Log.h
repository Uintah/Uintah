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
#include <Util/stringUtil.h>

namespace SemotusVisum {

using namespace std;
using namespace SCIRun;

/** Logging levels */
typedef enum {
  ERROR,
  WARNING,
  MESSAGE,
  ENTER,
  LEAVE,
  DEBUG
  
} logLevel;

/** Default logfile name. */
const string DEFAULT_LOGFILE = "Logfile2.log";

/** Default logging level. */
const logLevel     DEFAULT_LOGLEVEL = DEBUG;

/**
 * This class serves as a general-purpose logging interface. It supports
 * multiple levels of logging.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Log {
public:

  /** Closes the log file */
  static void close();

  /** Sets the log file name and opens the logfile for writing.
   *
   * @param    logFileName      File name for the log.
   *
   */
  static void setLogFileName( const string logFileName );

  /** Writes the given message to the logfile at the given logging
   *  level IF messages at that level are being logged.
   *
   * @param       level     Level at which to log this.
   * @param       message   Message to write to the logfile.
   *
   */
  static void log( logLevel level, const string message );

  /** Sets the new logging level.
   *
   * @param      level        New logging level.
   *
   */
  static void setLogLevel( logLevel level );
  
  
protected:
  /** Constructor - Ensures that we never create a log by itself. */
  Log() {}

  /** Destructor */
  ~Log() {}

  /** Log entry prefixes - Debug:, Error:, etc.*/
  static string logStrings[];
  
  /** Output file stream */
  static ofstream output;

  /** Current log level */
  static logLevel currentLogLevel;

  /** Mutex to ensure we don't diddle on other threads' writes. */
  static Mutex logMutex;
  
};

}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:21  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 20:58:14  simpson
// Adding CollabVis files/dirs
//
