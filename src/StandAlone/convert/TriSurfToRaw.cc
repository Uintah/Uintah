/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  TriSurfToRaw: Read in a TriSurf, output to a raw file
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TriSurf.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace SCIRun;

main(int argc, char **argv) {

  FieldHandle handle;
  if (argc !=4) {
    printf("Need the file name!\n");
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    printf("Couldn't open file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    printf("Error reading surface from file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  
  FILE *fout=fopen(argv[2], "wt");
//  int i;
//  for (i=0; i<ts->points.size(); i++) {
//    fprintf(fout, "%lf %lf %lf\n", ts->points[i].x(), ts->points[i].y(),
//	    ts->points[i].z());
    //	fprintf(stderr, "%lf %lf %lf\n", ts->points[i].x(), ts->points[i].y(),
    //		ts->points[i].z());
//  }
  fclose(fout);
  
  fout=fopen(argv[3], "wt");
//  for (i=0; i<ts->elements.size(); i++) {
//    fprintf(fout, "%d %d %d\n", ts->elements[i]->i1+1,
//	    ts->elements[i]->i2+1, ts->elements[i]->i3+1);
//  }
  fclose(fout);
  return 0;
}    
