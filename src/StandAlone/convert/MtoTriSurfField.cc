//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : MtoTriSurfField.cc
//    Author : Martin Cole
//    Date   : Wed Jun 14 16:00:46 2006

#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/NoData.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Init/init.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

bool ptsCountHeader;
int baseIndex;
bool elementsCountHeader;
bool binOutput;
bool debugOn;

void setDefaults() {
  ptsCountHeader=true;
  baseIndex=0;
  elementsCountHeader=true;
  binOutput=false;
  debugOn=false;
}

int
main(int argc, char **argv) {

  if (argc != 3) {
    cerr << "MtoTriSurfField <.m file> <tri surf field name>" << endl;
    return 2;
  }

  SCIRunInit();
  setDefaults();
  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
  TSMesh *tsm = new TSMesh();

  char *infile = argv[1];
  char *outfile = argv[2];

  ifstream instr(infile);

  if (instr.fail()) {
    cerr << "Error -- Could not open file " << argv[1] << endl;
    return 2;
  }

  unsigned int line = 1;
  while (! instr.eof()) {
    string type;
    instr >> type;
    unsigned int  idx;
    instr >> idx;
    if (type == "Vertex") {
      double x, y, z;
      instr >> x >> y >> z;
      //unsigned i = tsm->add_point(Point(x,y,z));
      //cerr << "Added point #"<< i <<": ("
      //   << x << ", " << y << ", " << z << ")" << endl;    
    } else if (type == "Face") {
      TSMesh::Node::array_type n(3);
      unsigned int n1, n2, n3;
      instr >> n1 >> n2 >> n3;
      n1 -= 1; n[0] = n1;
      n2 -= 1; n[1] = n2;
      n3 -= 1; n[2] = n3;

      //unsigned int i = tsm->add_elem(n);
      //cerr << "Added face #"<< i <<": ("
      //   << n1 << ", " << n2 << ", " << n3 << ")" << endl; 
    } else {
      if (instr.eof()) break;
      cerr << "Error: Parsing error:" << type << ": at line: " 
	   << line << endl;
      exit(2);
    }
    line++;
  }

  cerr << "Done: parsed " << line - 1 << " lines: " << endl;
  typedef NoDataBasis<double>  DatBasis;
  typedef GenericField<TSMesh, DatBasis, vector<double> > TSField;

  TSField *ts = scinew TSField(tsm);
  FieldHandle tsH(ts);

  //TextPiostream out_stream(outfile, Piostream::Write);  
  BinaryPiostream out_stream(outfile, Piostream::Write);
  Pio(out_stream, tsH);
  return 0;  
}    
