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
#include <sys/stat.h> /* stat */
#include <ctype.h>    /* isspace */
#include <stdlib.h>   /* malloc realloc */
#include <string.h>   /* string functions */
#include <unistd.h>   /* stat */
#define ASENV_SHORT_NAMES
#include "envseed.h"


#ifndef NULL
#  define NULL 0
#endif


/*
 * A position within argv -- an index into the array and an offset into the
 * element.  We assume <= 256 arguments, each <= 256 characters long.
 */
typedef struct {
  unsigned char index;
  unsigned char offset;
} ArgvPosition;


/*
 * Information about a single switch/value pair in argv -- the position and
 * length of the switch name and the position of its value (values always
 * extend to the end of the argv element).
 */
typedef struct {
  ArgvPosition namePosition;
  unsigned char nameLen;
  ArgvPosition valuePosition;
} ArgInfo;


/* Information created by ParseArgv for later reference by SwitchValueN */
static ArgInfo *arguments = NULL;
static int argumentsCount = 0;
const char *const *argvCopy = NULL;
const char *validCopy = NULL;


/* Argument value types recognized by ParseArgv. */
typedef enum {
  CHAR_TYPE, DOUBLE_TYPE, INT_TYPE, STRING_TYPE, VOID_TYPE
} ArgTypes;
#define ARG_TYPE_COUNT 5
static const char *ARG_TYPE_NAMES[ARG_TYPE_COUNT] = {
  "char", "double", "int", "string", "void"
};


static void
AppendInfo(const ArgInfo *info) {
  arguments = (ArgInfo *)
    realloc(arguments, (argumentsCount + 1) * sizeof(ArgInfo));
  arguments[argumentsCount++] = *info;
}


/* Returns 1 iff #value# is of type #type#. */
static int
CheckType(ArgTypes type,
          const char *value) {
  const char *valueEnd;
  if(value == NULL)
    return type == VOID_TYPE;
  if(type == DOUBLE_TYPE) {
    (void)strtod(value, (char **)&valueEnd);
    return *valueEnd == '\0';
  }
  else if(type == INT_TYPE) {
    (void)strtol(value, (char **)&valueEnd, 10);
    return *valueEnd == '\0';
  }
  else
    return (type == CHAR_TYPE) ? value[1] == '\0' :
           (type == VOID_TYPE) ? 0 : 1; /* STRING_TYPE */
}


/*
 * Searches #valid# for a switch specification for the #nameLen#-long #name#.
 * Returns a pointer to the specification if found, else NULL.
 */
static const char *
FindSpec(const char *valid,
         const char *name,
         int nameLen) {
  const char *spec = valid;
  while(1) {
    if(strncmp(spec, name, nameLen) == 0 && isspace((int)spec[nameLen]))
      return spec;
    if((spec = strchr(spec, '\n')) == NULL)
      break;
    spec++;
  }
  return NULL;
}


/* Returns the type noted in the argument specification #spec#. */
static ArgTypes
GetType(const char *spec) {

  int i;
  int nameLen;

  while(!isspace((int)*spec)) /* Skip switch name. */
    spec++;
  while(isspace((int)*spec))
    spec++;

  for(i = 0; i < ARG_TYPE_COUNT; i++) {
    nameLen = strlen(ARG_TYPE_NAMES[i]);
    if(strncmp(ARG_TYPE_NAMES[i], spec, nameLen) == 0 &&
       (isspace((int)spec[nameLen]) || spec[nameLen] == '\0'))
      return (ArgTypes)i;
  }
  return STRING_TYPE;

}


const char *
FindFile(const char *dirs,
         const char *names) {

  const char *dir;
  const char *dirEnd;
  struct stat ignored;
  const char *name;
  char *nameStart;
  const char *nameEnd;
  static char *returnValue = NULL;
  
  returnValue = realloc(returnValue, strlen(dirs) + 1 + strlen(names) + 1);

  for(dir = dirEnd = dirs; dirEnd != NULL; dir = dirEnd + 1) {
    strcpy(returnValue, dir);
    if((dirEnd = strchr(dir, ':')) == NULL) {
      nameStart = returnValue + strlen(returnValue);
    }
    else {
      nameStart = returnValue + (dirEnd - dir);
      *nameStart = '\0';
    }
    if(*(nameStart - 1) != '/') {
      *nameStart++ = '/';
      *nameStart = '\0';
    }
    for(name = nameEnd = names; nameEnd != NULL; name = nameEnd + 1) {
      strcpy(nameStart, name);
      if((nameEnd = strchr(name, ':')) != NULL) {
        *(nameStart + (nameEnd - name)) = '\0';
      }
      if(stat(returnValue, &ignored) == 0)
        return returnValue;
    }
  }

  return "";

}


int
ASENV_ParseArgv(const char *const *argv,
                const char *valid,
                ASENV_ErrorHandler handler) {

  const char *const *arg;
  const char *equals;
  ArgInfo info;
  int noMoreSwitches = 0;
  const char *single;
  const char *spec;
  const char *value;

  if(arguments != NULL) {
    free(arguments);
    arguments = NULL;
  }
  argvCopy = argv;
  if(validCopy != NULL)
    free((char *)validCopy);
  validCopy = strdup(valid);

  for(arg = argv + 1; *arg != NULL; arg++) {

    if(!noMoreSwitches && strcmp(*arg, "--") == 0) {
      noMoreSwitches = 1;
      continue;
    }

    info.namePosition.index = arg - argv;
    info.namePosition.offset = 1;

    if(noMoreSwitches || **arg != '-' || *(*arg + 1) == '\0') {
      /* Program argument. */
      if((spec = FindSpec(valid, "", 0)) == NULL) {
        if(handler == NULL ||
           !handler(ARGUMENTS_NOT_ALLOWED, *arg, 0, strlen(*arg), ""))
          return 0;
        else
          continue;
      }
      info.nameLen = 0;
      info.valuePosition.index = info.namePosition.index;
      info.valuePosition.offset = 0;
      value = *arg;
    }
    else if((spec = FindSpec(valid, *arg + 1, strlen(*arg) - 1)) != NULL) {
      /* The whole word is a single switch name. */
      info.nameLen = strlen(*arg) - 1;
      if(GetType(spec) == VOID_TYPE) {
        info.valuePosition.index = info.valuePosition.offset = 0;
        value = NULL;
      }
      else {
        info.valuePosition.index = info.namePosition.index + 1;
        info.valuePosition.offset = 0;
        value = (*(arg + 1) == NULL) ? NULL : *(++arg);
      }
    }
    else if((equals = strchr(*arg, '=')) != NULL &&
            (spec = FindSpec(valid, (*arg + 1), equals - *arg - 1)) != NULL) {
      /* -switch=value */
      info.nameLen = equals - *arg - 1;
      info.valuePosition.index = info.namePosition.index;
      info.valuePosition.offset = info.nameLen + 2;
      value = equals + 1;
    }
    else {
      /* Single-char switch list? */
      info.nameLen = 1;
      for(single = *arg + 1; *single != '\0'; single++) {
        if((spec = FindSpec(valid, single, 1)) == NULL) {
          if(handler == NULL ||
             !handler(UNKNOWN_SWITCH,
                      *arg,
                      single - *arg,
                      (single == *arg + 1) ? strlen(single) : 1,
                      ""))
            return 0;
          else
            break;
        }
        if(GetType(spec) == VOID_TYPE) {
          info.valuePosition.index = info.valuePosition.offset = 0;
          AppendInfo(&info);
        }
        else {
          if(*(single + 1) == '\0') {
            info.valuePosition.index = info.namePosition.index + 1;
            info.valuePosition.offset = 0;
            value = (*(arg + 1) == NULL) ? NULL : *(++arg);
          }
          else {
            info.valuePosition.index = info.namePosition.index;
            info.valuePosition.offset = info.namePosition.offset + 1;
            value = single + 1;
          }
          if(!CheckType(GetType(spec), value)) {
            if(handler == NULL)
              return 0;
            else if(value == NULL &&
                    !handler(MISSING_VALUE,
                             argv[info.namePosition.index],
                             info.namePosition.offset,
                             info.nameLen,
                             ""))
              return 0;
            else if(value != NULL &&
                    !handler(INVALID_VALUE,
                             argv[info.namePosition.index],
                             info.namePosition.offset,
                             info.nameLen,
                             value))
              return 0;
            else
              break;
          }
          AppendInfo(&info);
          break;
        }
        info.namePosition.offset++;
      }
      continue;
    }

    if(!CheckType(GetType(spec), value)) {
      if(handler == NULL)
        return 0;
      else if(value == NULL &&
              !handler(MISSING_VALUE,
                       argv[info.namePosition.index],
                       info.namePosition.offset,
                       info.nameLen,
                       ""))
        return 0;
      else if(value != NULL &&
              !handler(INVALID_VALUE,
                       argv[info.namePosition.index],
                       info.namePosition.offset,
                       info.nameLen,
                       value))
        return 0;
      else
        continue;
    }
    AppendInfo(&info);

  }

  return 1;

}


const char *
SwitchValueN(const char *switchName,
             unsigned instance,
             const char *defaultValue) {

  ArgInfo *beyond;
  ArgInfo *current;
  ArgInfo *first;
  int increment;
  ArgInfo *info = NULL;
  unsigned char switchLen = strlen(switchName);

  if(arguments != NULL) {
    if(instance == SWITCH_VALUE_LAST) {
      first = arguments + argumentsCount - 1;
      beyond = arguments - 1;
      increment = -1;
      instance = 1;
    }
    else {
      first = arguments;
      beyond = arguments + argumentsCount;
      increment = 1;
    }
    for(current = first; current != beyond; current += increment) {
      if(current->nameLen == switchLen &&
         strncmp(argvCopy[(int)current->namePosition.index] +
                 current->namePosition.offset,
                 switchName,
                 switchLen) == 0) {
        if(--instance == 0) {
          info = current;
          break;
        }
      }
    }
  }

  return (info == NULL) ? defaultValue :
         (info->valuePosition.index == 0 &&
          info->valuePosition.offset == 0) ? "" :
         argvCopy[(int)info->valuePosition.index] + info->valuePosition.offset;

}
