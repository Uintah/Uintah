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


#include "config.h"
#include <stdarg.h>    /* va_*, vfprintf() */
#include <stdio.h>     /* sprintf() */
#include <string.h>    /* strncpy() */
#include <sys/types.h> /* time_t */
#include <time.h>      /* time() */
#include <unistd.h>    /* getpid() */
#define ASDIAG_SHORT_NAMES
#include "diagnosticseed.h"


#define MAX_RECORDS 500
#ifndef NULL
#  define NULL 0
#endif


typedef struct {
  int type;
  FILE *direction;
  unsigned records;
} MessageTypeInfo;


static MessageTypeInfo *messageInfo = NULL;
static int messageInfoCount = 0;


/* Returns the entry for #type# in #messageInfo#, or NULL if it has none. */
static MessageTypeInfo *
FindInfo(int type) {
  int i;
  for(i = 0; i < messageInfoCount; i++) {
    if(messageInfo[i].type == type)
      return &messageInfo[i];
  }
  return NULL;
}


/*
 * Uses vprintf to print #message# and #arguments# to the file associated with
 * #type# messages.
 */
static void
PrintDiagnostic(int type,
                const char *message,
                va_list arguments) {

  MessageTypeInfo *info = NULL;

  if((info = FindInfo(type)) == NULL)
    return;

  /* Avoid filling up disk space from a program that runs a long time. */
  if(info->records++ >= MAX_RECORDS &&
     info->direction != stdout &&
     info->direction != stderr) {
    rewind(info->direction);
    info->records = 0;
  }

  if(info->direction != stdout && info->direction != stderr)
    fprintf(info->direction, "%ld %d ", (long)time(NULL), (int)getpid());
  vfprintf(info->direction, message, arguments);
  fflush(info->direction);

}


void
Diagnostic(int messageType,
           const char *message,
           ...) {
  va_list arguments;
  va_start(arguments, message);
  PrintDiagnostic(messageType, message, arguments);
  va_end(arguments);
}


FILE *
DiagnosticsDirection(int messageType) {
  MessageTypeInfo *info = FindInfo(messageType);
  return (info == NULL) ? ASDIAG_SUPPRESS : info->direction;
}


void
DirectDiagnostics(int messageType,
                  FILE *whereTo) {
  MessageTypeInfo *info = FindInfo(messageType);
  if(whereTo != ASDIAG_SUPPRESS) {
    if(info == NULL) {
      messageInfo = (MessageTypeInfo *)
        realloc(messageInfo, (messageInfoCount + 1) * sizeof(MessageTypeInfo));
      info = &messageInfo[messageInfoCount++];
      info->type = messageType;
      info->records = 0;
    }
    info->direction = whereTo;
  }
  else if(info != NULL) {
    *info = messageInfo[--messageInfoCount];
  }
}


void
PositionedDiagnostic(int messageType,
                     const char *fileName, 
                     unsigned line,
                     const char *message,
                     ...) {

  va_list arguments;
  char extendedMessage[512];
  char lineImage[10];

  sprintf(lineImage, ":%d ", line);
  va_start(arguments, message);
  if((strlen(fileName) + strlen(lineImage) + strlen(message)) <
     sizeof(extendedMessage)) {
    sprintf(extendedMessage, "%s%s%s", fileName, lineImage, message);
    PrintDiagnostic(messageType, extendedMessage, arguments);
  }
  else {
    PrintDiagnostic(messageType, message, arguments);
  }
  va_end(arguments);

}
