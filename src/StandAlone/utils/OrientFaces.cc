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
//    File   : OrientFaces.cc
//    Author : Martin Cole
//    Date   : Wed Mar 30 13:02:39 2005

#include <Core/Datatypes/TriSurfField.h>
#include <Core/Persistent/Pstreams.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_off.h>

#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

// generate a mesh with bad orientations to test the fix with.
void randomizie_orientation(TriSurfMesh *tsm) {
  TriSurfMesh::Face::iterator fiter, fend;
  tsm->begin(fiter);
  tsm->end(fend);
  srand(69);
  while(fiter != fend) {
    if (rand() % 3) {
      tsm->flip_face(*fiter);
    }
    ++fiter;
  }
}


int
main(int argc, char **argv) {

  FieldHandle handle;
  Piostream* stream = auto_istream(argv[1]);
  if (!stream) {
    cerr << "Couldn't open file " << argv[1] << ".  Exiting..." << endl;
    exit(1);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file " << argv[1] << ".  Exiting..." 
	 << endl;
    exit(2);
  }
  
  MeshHandle mb = handle->mesh();
  TriSurfMesh *tsm = dynamic_cast<TriSurfMesh *>(mb.get_rep());
  if (! tsm) { cerr << "Error: not a TriSurf." << endl; return 99;}


  string fout;
  if (string(argv[2]) == string("-randomize")) {
    randomizie_orientation(tsm);
    fout = argv[3];
  }  if (string(argv[2]) == string("-flip")) {
    tsm->flip_faces();
    fout = argv[3];
  } else {
    fout = argv[2];
    tsm->orient_faces();
  }

  TextPiostream out_stream(fout, Piostream::Write);
  Pio(out_stream, handle);
  return 0;  
}    
