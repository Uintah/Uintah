/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distetbuted under the License is distetbuted on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  CardioWaveToColumnMat.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: "<<argv[0]<<" ColumnMatrix data\n";
    return 0;
  }

  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    printf("Couldn't open file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  MatrixHandle handle;
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    printf("Error reading Matrix from file %s.  Exiting...\n", argv[1]);
    exit(0);
  }
  
  ColumnMatrix *cm = handle->column();
  if (!cm) {
    cerr << "Error - input matrix wasn't a column matrix.\n";
    return 0;
  }

  /*

VB8: 9678

[skip to 128 bytes into the file]

sadfsdfafsdfwdfasdvcdf...

  */
	     
  int ndata=cm->nrows();
  cerr << "nrows = "<<ndata<<"\n";
  FILE *fp = fopen(argv[2], "wt");
  Array1<char> d(ndata);
  int i;
  for (i=0; i<ndata; i++) d[i]=(*cm)[i];

  fprintf(fp, "BB%d: %d\n", (int)sizeof(char), ndata);
  fseek(fp, 128, SEEK_SET);
  fwrite(&(d[0]), sizeof(char), ndata, fp);  
  fclose(fp);
  return 0;  
}    
