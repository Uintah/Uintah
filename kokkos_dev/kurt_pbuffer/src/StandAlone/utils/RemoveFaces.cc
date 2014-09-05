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
//    File   : RemoveFaces.cc
//    Author : Martin Cole
//    Date   : Sun Feb 27 07:36:54 2005

#include <Core/Datatypes/TriSurfField.h>
#include <Core/Persistent/Pstreams.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

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

  vector<int> faces;
  for (int i = 2; i < argc - 1; i++) {
    faces.push_back(atoi(argv[i]));
  }
  bool altered = false;
  // remove last index first.
  sort(faces.begin(), faces.end());
  vector<int>::reverse_iterator iter  = faces.rbegin();
  while (iter != faces.rend()) {
    int face = *iter++;
    altered |= tsm->remove_face(face);
    cout << "removed face " << face << endl;
  }

  if (altered) {
    BinaryPiostream out_stream(argv[argc - 1], Piostream::Write);
    Pio(out_stream, handle);
  } else {
    cerr << "No faces removed. Exiting..." << endl;
    exit(3);
  }
  return 0;  
}    
