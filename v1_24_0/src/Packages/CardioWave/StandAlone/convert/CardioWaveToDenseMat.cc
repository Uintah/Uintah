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

#include <Core/Datatypes/DenseMatrix.h>
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
  if (argc < 6) {
    cerr << "Usage: "<<argv[0]<<" nnodes ntimesteps {B|L} DataFile DenseMatrixFile\n";
    return 0;
  }
  
  FILE *fp = fopen(argv[4], "rt");
  int nnodes=atoi(argv[1]);
  int ntimesteps=atoi(argv[2]);
  int bigendian=1;
  if (argv[3][0] == 'L') bigendian=0;
  cerr << "nnodes="<<nnodes<<" ntimesteps="<<ntimesteps<<" bigendian="<<bigendian<<"\n";

  DenseMatrix *m = scinew DenseMatrix(ntimesteps, nnodes);
  double *d = m->getData();
  fread(d, sizeof(double), ntimesteps*nnodes, fp);
  fclose(fp);

  int dsize=8;
  if (!bigendian) {
    cerr << "Swapping endianness...\n";
    char *x = (char *)(d);
    char swap;
    int i, j;
    for (i=0; i<ntimesteps*nnodes; i++) {
      char *start=x;
      char *finish=x+dsize-1;
      for (j=0; j<dsize/2; j++) {
	swap = *start; 
	*start = *finish;
	*finish = swap;
	start++;
	finish--;
      }
      x+=dsize;
    }
  }
  cerr << "first="<<(*m)[0][0]<<" last="<<(*m)[ntimesteps-1][nnodes-1]<<"\n";

  BinaryPiostream out_stream(argv[5], Piostream::Write);
  MatrixHandle mH(m);
  Pio(out_stream, mH);
  return 0;  
}    
