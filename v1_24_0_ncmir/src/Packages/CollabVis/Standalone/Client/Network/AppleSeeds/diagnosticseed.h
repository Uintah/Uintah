/* $Id$ */


/*
 * Copyright © 2000 The Regents of the University of California. 
 * All Rights Reserved. 
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice, this paragraph and the following three paragraphs
 * appear in all copies. 
 *
 * Permission to incorporate this software into commercial products may be
 * obtained by contacting
 * Eric Lund
 * Technology Transfer Office 
 * 9500 Gilman Drive 
 * 411 University Center 
 * University of California 
 * La Jolla, CA 92093-0093
 * (858) 534-0175
 * ericlund@ucsd.edu
 *
 * This software program and documentation are copyrighted by The Regents of
 * the University of California. The software program and documentation are
 * supplied "as is", without any accompanying services from The Regents. The
 * Regents does not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the program was
 * developed for research purposes and is advised not to rely exclusively on
 * the program for any reason. 
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
 * HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO
 * OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. 
 */


#ifndef DIAGNOSTICSEED_H
#define DIAGNOSTICSEED_H


/*
 * This package provides facilities for managing the production of diagnostic
 * messages.
 */


#include <stdio.h>  /* FILE */
#include <stdlib.h> /* exit */


#ifdef __cplusplus
extern "C" {
#endif


/* A FILE * value for suppressing messages of a particular type. */
#define ASDIAG_SUPPRESS 0


/*
 * Produces a #messageType# diagnostic message.  #message# is used as the
 * fprintf format string; additional fprintf arguments may be passed after it.
 */
void
ASDIAG_Diagnostic(int messageType,
                  const char *message,
                  ...);


/* Returns the file to which #messageType# diagnostic messages are directed. */
FILE *
ASDIAG_DiagnosticsDirection(int messageType);


/*
 * Directs #messageType# diagnostic messages to the file #whereTo#.  Messages
 * for any type may be supressed by passing ASDIAG_SUPPRESS as the second
 * parameter to this routine.  By default, all messages are suppressed.
 */
void
ASDIAG_DirectDiagnostics(int messageType,
                         FILE *whereTo);


/*
 * Produces a #messageType# diagnostic message.  #message# is used as the
 * fprintf format string; additional fprintf arguments may be passed after it.
 * #fileName# and #line# are prepended to the message.
 */
void
ASDIAG_PositionedDiagnostic(int messageType,
                            const char *fileName, 
                            unsigned line,
                            const char *message,
                            ...);


/*
 * Diagnostic types for debugging, error, fatal error, log, user information,
 * and warning messages.
 */
#define ASDIAG_DEBUG_MESSAGE   0
#define ASDIAG_ERROR_MESSAGE   1
#define ASDIAG_FATAL_MESSAGE   2
#define ASDIAG_LOG_MESSAGE     3
#define ASDIAG_INFO_MESSAGE    4
#define ASDIAG_WARNING_MESSAGE 5

/*
 * Issues the fatal message #message# with the file, line number, and any
 * specified parameters, then exits the program.
 */
#define ASDIAG_ABORT(message) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_FATAL_MESSAGE, __FILE__, __LINE__, message); \
    if(1) exit(1); \
  } while(0)
#define ASDIAG_ABORT1(message,a) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_FATAL_MESSAGE, __FILE__, __LINE__, message, a); \
    if(1) exit(1); \
  } while(0)
#define ASDIAG_ABORT2(message,a,b) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_FATAL_MESSAGE, __FILE__, __LINE__, message, a, b); \
    if(1) exit(1); \
  } while(0)
#define ASDIAG_ABORT3(message,a,b,c) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_FATAL_MESSAGE, __FILE__, __LINE__, message, a, b, c); \
    if(1) exit(1); \
  } while(0)
#define ASDIAG_ABORT4(message,a,b,c,d) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_FATAL_MESSAGE, __FILE__, __LINE__, message, a, b, c, d); \
    if(1) exit(1); \
  } while(0)
#define ASDIAG_ABORT5(message,a,b,c,d,e) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_FATAL_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e); \
    if(1) exit(1); \
  } while(0)

#ifdef ASDIAG_DISPLAY_DEBUG_DIAGNOSTICS
#  define ASDIAG_DEBUG(message) \
     ASDIAG_PositionedDiagnostic \
       (ASDIAG_DEBUG_MESSAGE, __FILE__, __LINE__, message)
#  define ASDIAG_DEBUG1(message,a) \
     ASDIAG_PositionedDiagnostic \
       (ASDIAG_DEBUG_MESSAGE, __FILE__, __LINE__, message, a)
#  define ASDIAG_DEBUG2(message,a,b) \
     ASDIAG_PositionedDiagnostic \
       (ASDIAG_DEBUG_MESSAGE, __FILE__, __LINE__, message, a, b)
#  define ASDIAG_DEBUG3(message,a,b,c) \
     ASDIAG_PositionedDiagnostic \
       (ASDIAG_DEBUG_MESSAGE, __FILE__, __LINE__, message, a, b, c)
#  define ASDIAG_DEBUG4(message,a,b,c,d) \
     ASDIAG_PositionedDiagnostic \
       (ASDIAG_DEBUG_MESSAGE, __FILE__, __LINE__, message, a, b, c, d)
#  define ASDIAG_DEBUG5(message,a,b,c,d,e) \
     ASDIAG_PositionedDiagnostic \
       (ASDIAG_DEBUG_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e)
#else
#  define ASDIAG_DEBUG(message)
#  define ASDIAG_DEBUG1(message,a)
#  define ASDIAG_DEBUG2(message,a,b)
#  define ASDIAG_DEBUG3(message,a,b,c)
#  define ASDIAG_DEBUG4(message,a,b,c,d)
#  define ASDIAG_DEBUG5(message,a,b,c,d,e)
#endif

/*
 * Issues the error message #message# with the file, line number, and any
 * specified parameters.
 */
#define ASDIAG_ERROR(message) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message)
#define ASDIAG_ERROR1(message,a) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a)
#define ASDIAG_ERROR2(message,a,b) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b)
#define ASDIAG_ERROR3(message,a,b,c) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b, c)
#define ASDIAG_ERROR4(message,a,b,c,d) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b, c, d)
#define ASDIAG_ERROR5(message,a,b,c,d,e) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e)

/*
 * Issues the error message #message# with the file, line number, and any
 * specified parameters, then returns 0 from the current function.
 */
#define ASDIAG_FAIL(message) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message); \
    if(1) return(0); \
  } while(0)
#define ASDIAG_FAIL1(message,a) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a); \
    if(1) return(0); \
  } while(0)
#define ASDIAG_FAIL2(message,a,b) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b); \
    if(1) return(0); \
  } while(0)
#define ASDIAG_FAIL3(message,a,b,c) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b, c); \
    if(1) return(0); \
  } while(0)
#define ASDIAG_FAIL4(message,a,b,c,d) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b, c, d); \
    if(1) return(0); \
  } while(0)
#define ASDIAG_FAIL5(message,a,b,c,d,e) \
  do { \
    ASDIAG_PositionedDiagnostic \
      (ASDIAG_ERROR_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e); \
    if(1) return(0); \
  } while(0)

/*
 * Issues the user information message #message# with the file, line number,
 * and any specified parameters.
 */
#define ASDIAG_INFO(message) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message)
#define ASDIAG_INFO1(message,a) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a)
#define ASDIAG_INFO2(message,a,b) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b)
#define ASDIAG_INFO3(message,a,b,c) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b, c)
#define ASDIAG_INFO4(message,a,b,c,d) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b, c, d)
#define ASDIAG_INFO5(message,a,b,c,d,e) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e)

/*
 * Issues the log message #message# with the file, line number, and any
 * specified parameters.
 */
#define ASDIAG_LOG(message) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message)
#define ASDIAG_LOG1(message,a) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a)
#define ASDIAG_LOG2(message,a,b) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b)
#define ASDIAG_LOG3(message,a,b,c) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b, c)
#define ASDIAG_LOG4(message,a,b,c,d) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b, c, d)
#define ASDIAG_LOG5(message,a,b,c,d,e) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_INFO_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e)

/*
 * Issues the warning message #message# with the file, line number, and any
 * specified parameters.
 */
#define ASDIAG_WARN(message) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_WARNING_MESSAGE, __FILE__, __LINE__, message)
#define ASDIAG_WARN1(message,a) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_WARNING_MESSAGE, __FILE__, __LINE__, message, a)
#define ASDIAG_WARN2(message,a,b) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_WARNING_MESSAGE, __FILE__, __LINE__, message, a, b)
#define ASDIAG_WARN3(message,a,b,c) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_WARNING_MESSAGE, __FILE__, __LINE__, message, a, b, c)
#define ASDIAG_WARN4(message,a,b,c,d) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_WARNING_MESSAGE, __FILE__, __LINE__, message, a, b, c, d)
#define ASDIAG_WARN5(message,a,b,c,d,e) \
  ASDIAG_PositionedDiagnostic \
    (ASDIAG_WARNING_MESSAGE, __FILE__, __LINE__, message, a, b, c, d, e)


#ifdef ASDIAG_SHORT_NAMES

#define SUPPRESS ASDIAG_SUPPRESS
#define Diagnostic ASDIAG_Diagnostic
#define DiagnosticsDirection ASDIAG_DiagnosticsDirection
#define DirectDiagnostics ASDIAG_DirectDiagnostics
#define PositionedDiagnostic ASDIAG_PositionedDiagnostic

#define DEBUG_MESSAGE   ASDIAG_DEBUG_MESSAGE
#define ERROR_MESSAGE   ASDIAG_ERROR_MESSAGE
#define FATAL_MESSAGE   ASDIAG_FATAL_MESSAGE
#define INFO_MESSAGE    ASDIAG_INFO_MESSAGE
#define LOG_MESSAGE     ASDIAG_LOG_MESSAGE
#define WARNING_MESSAGE ASDIAG_WARNING_MESSAGE

#define ABORT  ASDIAG_ABORT
#define ABORT1 ASDIAG_ABORT1
#define ABORT2 ASDIAG_ABORT2
#define ABORT3 ASDIAG_ABORT3
#define ABORT4 ASDIAG_ABORT4
#define ABORT5 ASDIAG_ABORT5
#ifdef DEBUG
#  undef DEBUG
#endif
#define DEBUG  ASDIAG_DEBUG
#define DEBUG1 ASDIAG_DEBUG1
#define DEBUG2 ASDIAG_DEBUG2
#define DEBUG3 ASDIAG_DEBUG3
#define DEBUG4 ASDIAG_DEBUG4
#define DEBUG5 ASDIAG_DEBUG5
#define ERROR  ASDIAG_ERROR
#define ERROR1 ASDIAG_ERROR1
#define ERROR2 ASDIAG_ERROR2
#define ERROR3 ASDIAG_ERROR3
#define ERROR4 ASDIAG_ERROR4
#define ERROR5 ASDIAG_ERROR5
#define FAIL   ASDIAG_FAIL
#define FAIL1  ASDIAG_FAIL1
#define FAIL2  ASDIAG_FAIL2
#define FAIL3  ASDIAG_FAIL3
#define FAIL4  ASDIAG_FAIL4
#define FAIL5  ASDIAG_FAIL5
#define INFO   ASDIAG_INFO
#define INFO1  ASDIAG_INFO1
#define INFO2  ASDIAG_INFO2
#define INFO3  ASDIAG_INFO3
#define INFO4  ASDIAG_INFO4
#define INFO5  ASDIAG_INFO5
#define LOG    ASDIAG_LOG
#define LOG1   ASDIAG_LOG1
#define LOG2   ASDIAG_LOG2
#define LOG3   ASDIAG_LOG3
#define LOG4   ASDIAG_LOG4
#define LOG5   ASDIAG_LOG5
#define WARN   ASDIAG_WARN
#define WARN1  ASDIAG_WARN1
#define WARN2  ASDIAG_WARN2
#define WARN3  ASDIAG_WARN3
#define WARN4  ASDIAG_WARN4
#define WARN5  ASDIAG_WARN5

#endif


#ifdef __cplusplus
}
#endif


#endif
