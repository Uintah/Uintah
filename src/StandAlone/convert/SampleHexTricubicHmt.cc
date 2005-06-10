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


/*
 *  HexVolFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun HexVolField, and will save
// out the HexVolMesh into two files: a .pts file and a .hex file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.
// The .hex file will specify the i/j/k/l/m/n/o/p indices for each hex,
// also one per line, again with a one line header (unless a 
// -noElementsCount flag is used).  The hex entries will be zero-based, 
// unless the user specifies -oneBasedIndexing.

#include <Core/Basis/Constant.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Basis/HexTricubicHmtScaleFactors.h>
#include <Core/Datatypes/HexVolMesh.h>
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

typedef HexVolMesh<HexTricubicHmtScaleFactors<Point> > HVMesh;
typedef HexTricubicHmtScaleFactors<double> DatBasis;
typedef GenericField<HVMesh, DatBasis, vector<double> > HVField;

using namespace SCIRun;

bool ptsCountHeader;
int baseIndex;
bool elementsCountHeader;

void setDefaults() {
  ptsCountHeader=true;
  baseIndex=0;
  elementsCountHeader=true;
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
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" HexVolField dataIn PointCloadField dataOut \n\n";
  cerr << "\t This program will read in a SCIRun HexVolField...\n";
}

int
main(int argc, char **argv) {
  if (argc < 5 || argc > 5) {
    printUsageInfo(argv[0]);
    return 2;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  setDefaults();

  char *hvName = argv[1];
  char *dataIn = argv[2];
  char *pcName = argv[3];
  char *dataOut = argv[4];

  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 2;
  }

  FieldHandle hhandle;
  Piostream* hstream=auto_istream(hvName);
  if (!hstream) {
    cerr << "Couldn't open file "<<hvName<<".  Exiting...\n";
    return 2;
  }
  Pio(*hstream, hhandle);
  if (!hhandle.get_rep()) {
    cerr << "Error reading surface from file "<<hvName<<".  Exiting...\n";
    return 2;
  }
  if (hhandle->get_type_description(1)->get_name().find("HexVolField") != 
      string::npos) {
    cerr << "Error -- input field wasn't a HexVolField (type_name="
	 << hhandle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }

  MeshHandle hmH = hhandle->mesh();
  HVMesh *hvm = dynamic_cast<HVMesh *>(hmH.get_rep());


  FieldHandle phandle;
  Piostream* pstream=auto_istream(pcName);
  if (!pstream) {
    cerr << "Couldn't open file "<<pcName<<".  Exiting...\n";
    return 2;
  }
  Pio(*pstream, phandle);
  if (!phandle.get_rep()) {
    cerr << "Error reading surface from file "<<pcName<<".  Exiting...\n";
    return 2;
  }
  if (phandle->get_type_description(1)->get_name().find("PointCloudField") != 
      string::npos) 
  {
    cerr << "Error -- input field wasn't a PointCloudField (type_name="
	 << phandle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;  
  MeshHandle pmH = phandle->mesh();
  PCMesh *pcm = dynamic_cast<PCMesh *>(pmH.get_rep());
  PCMesh::Node::iterator niter; 
  PCMesh::Node::iterator niter_end; 
  PCMesh::Node::size_type nsize; 
  pcm->begin(niter);
  pcm->end(niter_end);
  pcm->size(nsize);






  FILE *fdataIn = fopen(dataIn, "r");
  if (!fdataIn) {
    cerr << "Error opening input file "<<dataIn<<"\n";
    return 2;
  }
  fclose(fdataIn);

  HVMesh::Cell::size_type csize; 
  HVMesh::Cell::iterator citer; 
  HVMesh::Cell::iterator citer_end; 
  HVMesh::Node::array_type cell_nodes(8);
  hvm->size(csize);
  hvm->begin(citer);
  hvm->end(citer_end);
  FILE *fdataOut = fopen(dataOut, "w");
  if (!fdataOut) {
    cerr << "Error opening output file "<<dataOut<<"\n";
    return 2;
  }
 

  while(citer != citer_end) {
    hvm->get_nodes(cell_nodes, *citer);
    fprintf(fdataOut, "%d %d %d %d %d %d %d %d\n", 
	    (int)cell_nodes[0]+baseIndex,
	    (int)cell_nodes[1]+baseIndex,
	    (int)cell_nodes[2]+baseIndex,
	    (int)cell_nodes[3]+baseIndex,
	    (int)cell_nodes[4]+baseIndex,
	    (int)cell_nodes[5]+baseIndex,
	    (int)cell_nodes[6]+baseIndex,
	    (int)cell_nodes[7]+baseIndex);
    ++citer;
  }
  fclose(fdataOut);

  return 0;  
}    
