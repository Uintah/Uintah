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
 *  StructHexVolFieldToText.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun StructHexVolField, and will save
// out the StructHexVolMesh into a .pts file.
// The .pts file will specify the x/y/z coordinates of each 
// point, one per line, entries separated by white space; the file will
// also have a one line header, specifying ni and nj, unless the user 
// specifies the -noHeader command-line argument.

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/StructHexVolMesh.h>
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

bool header;

void setDefaults() {
  header=true;
}

int parseArgs(int argc, char *argv[]) {
  int currArg = 4;
  while (currArg < argc) {
    if (!strcmp(argv[currArg],"-noHeader")) {
      header=false;
      currArg++;
    } else {
      cerr << "Error - unrecognized argument: "<<argv[currArg]<<"\n";
      return 0;
    }
  }
  return 1;
}

void printUsageInfo(char *progName) {
  cerr << "\n Usage: "<<progName<<" StructHexVolField pts [-noHeader]\n\n";
  cerr << "\t This program will read in a SCIRun StructHexVolField, and \n";
  cerr << "\t will save out the StructHexVolMesh into a .pts file.  The \n";
  cerr << "\t .pts file will specify the x/y/z coordinates of each point, \n";
  cerr << "\t one per line, entries separated by white space; the file \n";
  cerr << "\t will also have a one line header, specifying ni and nj, \n";
  cerr << "\t unless the user specifies the -noHeader command-line \n";
  cerr << "\t argument.\n\n";
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
  if (handle->get_type_description(1)->get_name().find("StructHexVolField") !=
      string::npos)
  {
    cerr << "Error -- input field wasn't a StructHexVolField (type_name="
	 << handle->get_type_description(1)->get_name() << std::endl;
    return 2;
  }

  typedef StructHexVolMesh<HexTrilinearLgn<Point> > SHVMesh;

  MeshHandle mH = handle->mesh();
  SHVMesh *shvm = dynamic_cast<SHVMesh *>(mH.get_rep());
  SHVMesh::Node::iterator niter; 
  SHVMesh::Node::iterator niter_end; 
  shvm->begin(niter);
  shvm->end(niter_end);
  vector<unsigned int> dims;
  shvm->get_dim(dims);
  FILE *fPts = fopen(ptsName, "wt");
  if (!fPts) {
    cerr << "Error opening output file "<<ptsName<<"\n";
    return 2;
  }
  if (header) fprintf(fPts, "%d %d %d\n", dims[0], dims[1], dims[2]);
  cerr << "ni="<< dims[0] << " nj="<<dims[1]<< " nk="<<dims[2]<<"\n";
  while(niter != niter_end) {
    Point p;
    shvm->get_center(p, *niter);
    fprintf(fPts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fPts);
  return 0;  
}    
