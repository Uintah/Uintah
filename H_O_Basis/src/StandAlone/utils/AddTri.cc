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
//    File   : AddTri.cc
//    Author : Martin Cole
//    Date   : Thu Mar  3 20:13:23 2005

#include <Core/Datatypes/TriSurfField.h>
#include <Core/Persistent/Pstreams.h>

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
  if (! tsm) { cerr << "Error: not a TriSurf." << endl; return 99;}

  int n1 = atoi(argv[2]);
  int n2 = atoi(argv[3]);
  int n3 = atoi(argv[4]);

  tsm->add_triangle(n1, n2, n3);
  TextPiostream out_stream(argv[5], Piostream::Write);
  Pio(out_stream, handle);
  return 0;  
}    
