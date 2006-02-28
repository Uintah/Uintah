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
#include <Core/Util/Endian.h>
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
    cerr << "Usage: "<<argv[0]<<" data ColumnMatrix\n";
    return 0;
  }

  FILE *fp = fopen(argv[1], "rt");
  char c=fgetc(fp);
  if (c != 'B') {
    cerr << "Error - thought the first char was supposed to be a 'V' -- but saw a '"<<c<<"'\n";
    return 0;
  }
  c=fgetc(fp);
  if (c != 'B' && c != 'L') {
    cerr << "Error - thought the second char was supposed to be a 'B' or 'L' -- but saw a '"<<c<<"'\n";
    return 0;
  }
  bool swap=(c == 'L' && isBigEndian() || c == 'B' && isLittleEndian());
  int dsize, ndata;
  fscanf(fp, "%d:%d", &dsize, &ndata);
  cerr << "entry size = "<<dsize<<"   ndata = "<<ndata<<"\n";
  fclose(fp);
  ColumnMatrix *cm = scinew ColumnMatrix(ndata);
  Array1<unsigned char> chardata(ndata);
  fp=fopen(argv[1], "rt");
  int count=0;
  while(count<128) {fgetc(fp); count++;}
  fread(&(chardata[0]), sizeof(unsigned char), ndata, fp);
  for (int i=0; i<ndata; i++) (*cm)[i]=chardata[i];
  fclose(fp);

  cerr << "size="<<ndata<<" first="<<(*cm)[0]<<" last="<<(*cm)[ndata-1]<<"\n";

  TextPiostream out_stream(argv[2], Piostream::Write);
  MatrixHandle cmH(cm);
  Pio(out_stream, cmH);
  return 0;  
}    
