/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//    File   : TextToHexTricubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004


// This program will read in a UCSD style .xls (specifying node variables of each 
// point, one per line, entries separated by white space, file can have 
// an optional one line header specifying number of points... and if it
// doesn't, you have to use the -noPtsCount command-line argument) and a
// .hex file (specifying i/j/k/l/m/n/o/p indices for each hex, also one per 
// line, again with an optional one line header (use -noElementsCount if it's 
// not there).  The hex entries are assumed to be zero-based, unless you 
// specify -oneBasedIndexing.  And the SCIRun output file is written in 
// ASCII, unless you specify -binOutput.

#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/NoData.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
#endif
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace SCIRun;

typedef HexVolMesh<HexTricubicHmt<Point> >    HVMesh;
typedef NoDataBasis<double>                DatBasis;
typedef GenericField<HVMesh, DatBasis, vector<double> > HVField;

typedef HexVolMesh<HexTrilinearLgn<Point> >    HVLinMesh;
typedef HexTrilinearLgn<double>                ScaDatBasis;
typedef HexTrilinearLgn<Vector>                VecDatBasis;
typedef GenericField<HVLinMesh, ScaDatBasis, vector<double> > HVLinSField;
typedef GenericField<HVLinMesh, VecDatBasis, vector<Vector> > HVLinVField;

using std::cerr;
using std::ifstream;
using std::endl;


bool ptsCountHeader;
int baseIndex;
bool elementsCountHeader;
bool binOutput;
bool debugOn;

void setDefaults() {
  ptsCountHeader=true;
  baseIndex=1;
  elementsCountHeader=true;
  binOutput=false;
  debugOn=false;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-noElementsCount")) {
      elementsCountHeader=false;
      currArg++;
    } else if (!strcmp(argv[currArg], "-oneBasedIndexing")) {
      baseIndex=1;
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

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" pts hexes HexVolMesh [-noPtsCount] [-noElementsCount] [-oneBasedIndexing] [-binOutput] [-debug]\n\n";
  cerr << "\t This program will read in a .pts (specifying the x/y/z \n";
  cerr << "\t coords of each point, one per line, entries separated by \n";
  cerr << "\t white space, file can have an optional one line header \n";
  cerr << "\t specifying number of points... and if it doesn't, you have \n";
  cerr << "\t to use the -noPtsCount command-line argument) and a .hex \n";
  cerr << "\t file (specifying i/j/k/l/m/n/o/p indices for each hex, also \n";
  cerr << "\t one per line, again with an optional one line header (use \n";
  cerr << "\t -noElementsCount if it's not there).  The hex entries are \n";
  cerr << "\t assumed to be zero-based, unless you specify \n";
  cerr << "\t -oneBasedIndexing.  And the SCIRun output file is written \n";
  cerr << "\t in ASCII, unless you specify -binOutput.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 4 || argc > 9) {
    printUsageInfo(argv[0]);
    return 0;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();
  vector<Vector> data_vals;
  HVLinMesh *hvm = new HVLinMesh();
  
  char *ptsName = argv[1];
  char *hexesName = argv[2];
  char *fieldName = argv[3];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 0;
  }
  int npts = 0;
  npts = getNumNonEmptyLines(ptsName) - 1;
  ifstream ptsstream(ptsName);
  cerr << "number of points = "<< npts << endl;
  
  // parse header out.
  vector<string> header;
  for (int i = 0; i < 8 * 5 * 3 + 2; i++) {
    string s;
    ptsstream >> s;
    header.push_back(s);
  }
  vector<string>::iterator iter = header.begin();
  while (iter != header.end()) {
    cerr << *iter++ << endl;
  }
  

  for (int i=0; i<npts; i++) {
    double x[3], xdx[3], xdy[3], xdxy[3], xdz[3], xdyz[3], xdxz[3], xdxyz[3];
    for (int j=0; j<3; j++) 
      ptsstream >> x[j] >> xdx[j] >> xdy[j] >> xdxy[j] >> xdz[j] 
		>> xdyz[j] >> xdxz[j] >> xdxyz[j];

    double f[3], fdx[3], fdy[3], fdxy[3], fdz[3], fdyz[3], fdxz[3], fdxyz[3];
    for (int j=0; j<3; j++) 
      ptsstream >> f[j] >> fdx[j] >> fdy[j] >> fdxy[j] >> fdz[j] 
		>> fdyz[j] >> fdxz[j] >> fdxyz[j];

    double phi, theta;
    const double torad = 3.145926535898 / 180.;
    // convert to radians
    phi = f[0] * torad;
    theta = f[1] * torad;
    // create unit vectors
    Vector fib_dir;
    fib_dir.x(sin(phi) * cos(theta));
    fib_dir.y(sin(phi) * sin(theta));
    fib_dir.z(cos(phi));

    //cerr << fib_dir << " len ->" << fib_dir.length() << endl;
    data_vals.push_back(fib_dir);

    double v[3][3], vdx[3][3], vdy[3][3], vdxy[3][3], vdz[3][3], vdyz[3][3], vdxz[3][3], vdxyz[3][3];
    for (int j=0; j<3; j++) 
      for (int k=0; k<3; k++) 
	ptsstream >> v[j][k] >> vdx[j][k] >> vdy[j][k] >> vdxy[j][k] 
		  >> vdz[j][k] >> vdyz[j][k] >> vdxz[j][k] >> vdxyz[j][k];

    string label;
    int n;
    ptsstream >> label >> n;
    const double sf = 12.0;
    hvm->add_point(Point(x[0],x[1],x[2]));
//     vector<Point> arr(7);
//     arr[0].x(sf * xdx[0]);
//     arr[0].y(sf * xdx[1]);
//     arr[0].z(sf * xdx[2]);
//     arr[1].x(sf * xdy[0]);
//     arr[1].y(sf * xdy[1]);
//     arr[1].z(sf * xdy[2]);
//     arr[2].x(sf * xdz[0]);
//     arr[2].y(sf * xdz[1]);
//     arr[2].z(sf * xdz[2]);
//     arr[3].x(sf * xdxy[0]);
//     arr[3].y(sf * xdxy[1]);
//     arr[3].z(sf * xdxy[2]);
//     arr[4].x(sf * xdyz[0]);
//     arr[4].y(sf * xdyz[1]);
//     arr[4].z(sf * xdyz[2]);
//     arr[5].x(sf * xdxz[0]);
//     arr[5].y(sf * xdxz[1]);
//     arr[5].z(sf * xdxz[2]);
//     arr[6].x(sf * xdxyz[0]);
//     arr[6].y(sf * xdxyz[1]);
//     arr[6].z(sf * xdxyz[2]);
//     hvm->get_basis().add_derivatives(arr);

//     if (debugOn) 
//       cerr << "Added point #" << i << ": (" << x[0] << ", " << x[1] 
// 	   << ", " << x[2] << ")" << endl;
  }

  cerr << "done adding points.\n";

  int nhexes;
  nhexes = getNumNonEmptyLines(hexesName) - 1;
  ifstream hexesstream(hexesName);

  cerr << "number of hexes = "<< nhexes <<"\n";

  // parse header out.
  vector<string> eheader;
  for (int i = 0; i < 10; i++) {
    string s;
    hexesstream >> s;
    eheader.push_back(s);
  }
  iter = eheader.begin();
  while (iter != eheader.end()) {
    cerr << *iter++ << endl;
  }

  for (int i = 0; i < nhexes; i++) {
    int n1, n2, n3, n4, n5, n6, n7, n8;
    string label;
    int e;
    hexesstream >> n1 >> n2 >> n3 >> n4 >> n5 >> n6 >> n7 >> n8 >> label >> e;
    n1-=baseIndex; 
    n2-=baseIndex; 
    n3-=baseIndex; 
    n4-=baseIndex; 
    n5-=baseIndex; 
    n6-=baseIndex; 
    n7-=baseIndex; 
    n8-=baseIndex; 
    if (n1<0 || n1>=npts) { 
      cerr << "Error -- n1 ("<<i<<") out of bounds: "<<n1<<"\n"; 
      return 0; 
    }
    if (n2<0 || n2>=npts) { 
      cerr << "Error -- n2 ("<<i<<") out of bounds: "<<n2<<"\n"; 
      return 0; 
    }
    if (n3<0 || n3>=npts) { 
      cerr << "Error -- n3 ("<<i<<") out of bounds: "<<n3<<"\n"; 
      return 0; 
    }
    if (n4<0 || n4>=npts) { 
      cerr << "Error -- n4 ("<<i<<") out of bounds: "<<n4<<"\n"; 
      return 0; 
    }
    if (n5<0 || n5>=npts) { 
      cerr << "Error -- n5 ("<<i<<") out of bounds: "<<n5<<"\n"; 
      return 0; 
    }
    if (n6<0 || n6>=npts) { 
      cerr << "Error -- n6 ("<<i<<") out of bounds: "<<n6<<"\n"; 
      return 0; 
    }
    if (n7<0 || n7>=npts) { 
      cerr << "Error -- n7 ("<<i<<") out of bounds: "<<n7<<"\n"; 
      return 0; 
    }
    if (n8<0 || n8>=npts) { 
      cerr << "Error -- n8 ("<<i<<") out of bounds: "<<n8<<"\n"; 
      return 0; 
    }
    
    hvm->add_hex(n1, n2, n4, n3, n5, n6, n8, n7);
    if (debugOn) 
      cerr << "Added hex #"<<i<<": ["<<n1<<" "<<n2<<" "<<n3<<" "<<n4<<" "<<n5<<" "<<n6<<" "<<n7<<" "<<n8<<"]\n";
  }
  cerr << "done adding elements.\n";

  HVLinVField *hv = scinew HVLinVField(hvm);
  FieldHandle hvH(hv);

  cerr << "loading in vector data" << endl;

  hv->resize_fdata();
  hv->fdata() = data_vals;
  

  
  if (binOutput) {
    BinaryPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, hvH);
  } else {
    TextPiostream out_stream(fieldName, Piostream::Write);
    Pio(out_stream, hvH);
  }
  return 0;  
}    
