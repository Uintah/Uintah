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
#define ASRES_SHORT_NAMES
#include "resourceseed.h"
#include <ctype.h>     /* isspace */
#include <stdio.h>     /* fgets pclose popen sprintf */
#include <stdlib.h>    /* strtod */
#include <string.h>    /* strstr */
#include <unistd.h>    /* fork sysconf */


#ifdef ASRES_HAVE_STATVFS
#  include <sys/statvfs.h>
   typedef struct statvfs FileSystemStats;
#  define GetFileSystemStats statvfs
#  define BlockSizeField f_frsize
#else
#  include <sys/statfs.h>
   typedef struct statfs FileSystemStats;
#  define GetFileSystemStats statfs
#  define BlockSizeField f_bsize
#endif



#define BUFFER_SIZE 1024


/* Returns a pointer to the first char of the #column#th column of line #c#. */
static const char *
FindColumn(const char *c,
           unsigned column) {
  unsigned col;
  for( ; isspace((int)*c); c++)
    ; /* Nothing more to do. */
  for(col = 1; col < column; col++) {
    for( ; !isspace((int)*c) && *c != '\0'; c++)
      ; /* Nothing more to do. */
    for( ; isspace((int)*c); c++)
      ; /* Nothing more to do. */
  }
  return c;
}


double
FreeCpu(int nice) {

  char *ignored;
  char line[BUFFER_SIZE];
  FILE *psFile;
  double total = 0.0;

#ifdef ASRES_PS_SUPPORTS_SYSV

  const char *psCommand = "ps -A -o pcpu,nice";

  if((psFile = popen(psCommand, "r")) == NULL)
    return -1;

  while(fgets(line, sizeof(line), psFile) != NULL) {
    /* Note: SysV ps nice values are biased by +20. */
    if((atoi(FindColumn(line, 2)) - 20) > nice)
      continue;
    total += strtod(FindColumn(line, 1), &ignored);
  }

#else

  /*
   * Unfortunately, none of the BSD ps output formats include both nice and
   * %CPU, so we have to invoke ps twice to get everything we need.
   */
  const char *cpuCommand = "ps uax";
  const char *niceCommand = "ps lax";
  char *nicePids;
  char pidImage[15 + 1];

  if((psFile = popen(niceCommand, "r")) == NULL)
    return -1;

  /* Collect all the pids with a higher nice value than #nice#. */
  nicePids = strdup("");
  while(fgets(line, sizeof(line), psFile) != NULL) {
    if(atoi(FindColumn(line, 6)) <= nice)
      continue;
    strncpy(pidImage, FindColumn(line, 3), sizeof(pidImage));
    *strchr(pidImage, ' ') = '\0';
    nicePids = realloc(nicePids, strlen(nicePids) + 2 + strlen(pidImage));
    sprintf(nicePids, "%s %s", nicePids, pidImage);
  }
  pclose(psFile);

  if((psFile = popen(cpuCommand, "r")) == NULL) {
    free(nicePids);
    return -1;
  }

  /* Sum the %CPU of all processes with a nice value <= #nice#. */
  while(fgets(line, sizeof(line), psFile) != NULL) {
    strncpy(pidImage, FindColumn(line, 2), sizeof(pidImage));
    *strchr(pidImage, ' ') = '\0';
    if(strstr(nicePids, pidImage) != NULL)
      continue;
    total += strtod(FindColumn(line, 3), &ignored);
  }

  free(nicePids);

#endif

  pclose(psFile);
  return 100.0 - total;

}


double
FreeDisk(const char *path) {
  FileSystemStats fsStats;
  return (GetFileSystemStats(path, &fsStats) < 0) ? -1 :
         (fsStats.f_bavail * fsStats.BlockSizeField);
}


double
FreeMemory(void) {
#ifdef _SC_AVPHYS_PAGES
  /* Note: this method under-reports actual availability. */
  return sysconf(_SC_PAGE_SIZE) * sysconf(_SC_AVPHYS_PAGES);
#else
  return -1;
#endif
}


double
ProcessCpuTime(pid_t process) {

  FILE *commandFile;
  double hours = 0.0;
  char line[BUFFER_SIZE];
  double minutes = 0.0;
  char processImage[30];
  char psCommand[128 + 1];
#ifdef ASRES_PS_SUPPORTS_SYSV
  char *psFormat = "ps -A -o pid,tty,class,time -p %d";
#else
  char *psFormat = "ps ax %d";
#endif
  double seconds = 0.0;
  const char *time = NULL;

  sprintf(psCommand, psFormat, (int)process);
  if((commandFile = popen(psCommand, "r")) == NULL)
    return -1;
  sprintf(processImage, "%d", (int)process);
  while(fgets(line, sizeof(line), commandFile) != NULL) {
    if(strstr(line, processImage) == NULL)
      continue; /* Probably a header line. */
    time = FindColumn(line, 4);
    /*
     * BSD ps presents some STAT values in multiple columns, so the fourth
     * column is not necessarily the beginning of the process CPU time.
     */
    while(!isdigit((int)*time) && *time != '\0')
      time++;
    while(isdigit((int)*time) || *time == ':') {
      if(*time == ':') {
        hours = minutes;
        minutes = seconds;
        seconds = 0.0;
      }
      else {
        seconds = seconds * 10.0 + (*time - '0');
      }
      time++;
    }
  }
  pclose(commandFile);
  return (time == NULL) ? -1 : (hours * 3600.0 + minutes * 60.0 + seconds);

}
