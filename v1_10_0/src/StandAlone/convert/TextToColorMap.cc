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
 *  TextToColorMap.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a .txt (specifying the r/g/b/a values of each 
// color, one per line, entries separated by white space, file can have 
// an optional one line header specifying number of colors... and if it
// doesn't, you have to use the -noHeader command-line argument).  
// The SCIRun output ColorMap file is written in ASCII, unless you specify 
// -binOutput.

#include <Core/Geom/ColorMap.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool header;
bool binOutput;
bool debugOn;

void setDefaults() {
  header=true;
  binOutput=false;
  debugOn=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-binOutput")) {
      binOutput=true;
      currArg++;
    } else if (!strcmp(argv[currArg], "-debug")) {
      debugOn=true;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

int getNumNonEmptyLines(char *fname) {
  // read through the file -- when you see a non-white-space set a flag to one.
  // when you get to the end of the line (or EOF), see if the flag has
  // been set.  if it has, increment the count and reset the flag to zero.

  FILE *fin = fopen(fname, "rt");
  int count=0;
  int haveNonWhiteSpace=0;
  int c;
  while ((c=fgetc(fin)) != EOF) {
    if (!isspace(c)) haveNonWhiteSpace=1;
    else if (c=='\n' && haveNonWhiteSpace) {
      count++;
      haveNonWhiteSpace=0;
    }
  }
  if (haveNonWhiteSpace) count++;
  cerr << "number of nonempty lines was: "<<count<<"\n";
  return count;
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 6) {
    cerr << "Usage: "<<argv[0]<<" textfile ColorMap [-noHeader] [-binOutput] [-debug]\n";
    return 0;
  }
  setDefaults();

  char *textfileName = argv[1];
  char *colormapName = argv[2];
  if (!parseArgs(argc, argv)) return 0;

  int ncolors;
  if (!header) ncolors = getNumNonEmptyLines(textfileName);
  ifstream textstream(textfileName);
  if (header) textstream >> ncolors;
  cerr << "number of colormap entries = "<< ncolors <<"\n";
  int i;
  vector<Color> rgb(ncolors);
  vector<float> rgbT(ncolors);
  vector<float> alphas(ncolors);
  vector<float> alphaT(ncolors);
  for (i=0; i<ncolors; i++) {
    double r, g, b, a, t;
    textstream >> r >> g >> b >> a >> t;
    Color clr(r,g,b);
    rgb[i]=clr;
    rgbT[i]=t;
    alphas[i]=a;
    alphaT[i]=t;
    if (debugOn) 
      cerr << "Added color: ("<<r<<","<<g<<","<<b<<") alpha="<<a<<" t="<<t<<"\n";
  }
  cerr << "Done building colormap.\n";

  ColorMap *cm = new ColorMap(rgb, rgbT, alphas, alphaT);
  ColorMapHandle cmH(cm);
  
  if (binOutput) {
    BinaryPiostream out_stream(colormapName, Piostream::Write);
    Pio(out_stream, cmH);
  } else {
    TextPiostream out_stream(colormapName, Piostream::Write);
    Pio(out_stream, cmH);
  }
  return 0;  
}    
