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
 *  PointCloudFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun PointCloudField, and will save
// out the PointCloudMesh into a .pts file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying number of points, unless the
// user specifies the -noPtsCount command-line argument.

#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#include <Core/Init/init.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool ptsCountHeader;

void setDefaults() {
  ptsCountHeader=true;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 3;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noPtsCount")) {
      ptsCountHeader=false;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" PointCloudField pts [-noPtsCount]\n\n";
  cerr << "\t This program will read in a SCIRun PointCloudField, and \n";
  cerr << "\t will save out the PointCloudMesh into a .pts file. \n";
  cerr << "\t The .pts file will specify the x/y/z coordinates of each \n";
  cerr << "\t point, one per line, entries separated by white space; the \n";
  cerr << "\t file will also have a one line header, specifying number \n";
  cerr << "\t of points, unless the user specifies the -noPtsCount \n";
  cerr << "\t command-line argument.\n\n";
}

int
main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    printUsageInfo(argv[0]);
    return 2;
  }
  SCIRunInit();	       
  setDefaults();

  char *fieldName = argv[1];
  char *ptsName = argv[2];
  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 2;
  }


  FieldHandle handle;
  Piostream* stream=auto_istream(fieldName);
  if (!stream) {
    cerr << "Couldn't open file "<<fieldName<<".  Exiting...\n";
    return 2;
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<fieldName<<".  Exiting...\n";
    return 2;
  }
  if (handle->get_type_description(1)->get_name().find("PointCloudField") != 
      string::npos) 
  {
    cerr << "Error -- input field wasn't a PointCloudField (type_name="
	 << handle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;  
  MeshHandle mH = handle->mesh();
  PCMesh *pcm = dynamic_cast<PCMesh *>(mH.get_rep());
  PCMesh::Node::iterator niter; 
  PCMesh::Node::iterator niter_end; 
  PCMesh::Node::size_type nsize; 
  pcm->begin(niter);
  pcm->end(niter_end);
  pcm->size(nsize);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    return 2;
  }
  if (ptsCountHeader) fprintf(fPts, "%d\n", (unsigned)(nsize));
  cerr << "Number of points = "<< nsize <<"\n";
  while(niter != niter_end) {
    Point p;
    pcm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);
  return 0;  
}    
