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
//    File   : HexVolToVtk.cc
//    Author : Martin Cole
//    Date   : Fri Feb  4 10:52:27 2005
//  NOTE: Adapted from TriSurfToVtk




#include <Core/Datatypes/HexVolField.h>
#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " scirun_field vtk_unstructured_grid" 
	 << std::endl;
    return 0;
  }

  FieldHandle handle;
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Couldn't open file " << argv[1] << ".  Exiting..." << std::endl;
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<argv[1]<<".  Exiting..." 
	 << std::endl;
    exit(0);
  }
  if (handle->get_type_name(0) != "HexVolField") {
    cerr << "Error -- input field wasn't a HexVolField (type_name="
	 << handle->get_type_name(0) << std::endl;
    exit(0);
  }

  MeshHandle mb = handle->mesh();
  HexVolMesh *hvm = dynamic_cast<HexVolMesh *>(mb.get_rep());

  ofstream fostr(argv[2]);
  if (!fostr) {
    cerr << "Error - can't create output file " << argv[2] << std::endl;
    exit(0);
  }
  
  fostr << "# vtk DataFile Version 3.0" << endl 
	<< "vtk output" << endl 
	<< "ASCII" << endl 
	<< "DATASET UNSTRUCTURED_GRID" 
	<< endl << endl;
  
  HexVolMesh::Node::iterator niter; hvm->begin(niter);
  HexVolMesh::Node::iterator niter_end; hvm->end(niter_end);
  HexVolMesh::Node::size_type nsize; hvm->size(nsize);

  cerr << "Writing "<< nsize << " points to " << argv[2] 
       << "..." << endl;
  fostr << "POINTS " << nsize << " float" << endl;

  while(niter != niter_end)
  {
    Point p;
    hvm->get_center(p, *niter);
    fostr << setprecision(9) << p.x() << " " << p.y() << " " << p.z() << endl;
    ++niter;
  }

  HexVolMesh::Cell::size_type csize; hvm->size(csize);
  HexVolMesh::Cell::iterator citer; hvm->begin(citer);
  HexVolMesh::Cell::iterator citer_end; hvm->end(citer_end);
  HexVolMesh::Node::array_type cnodes(8);
  cerr << "     and "<< csize << " cells." << endl; 
  fostr << endl << "CELLS " << csize <<  " " << csize * (8+1) << endl;
  while(citer != citer_end)
  {
    hvm->get_nodes(cnodes, *citer);
    fostr << "8 " << cnodes[0] << " " << cnodes[1] << " " << cnodes[2] << " " 
	  << cnodes[3] << " " << cnodes[4] << " " << cnodes[5] << " "
	  << cnodes[6] << " " << cnodes[7] << endl;
    ++citer;
  }

  // VTK_HEXAHEDRON = 12
  fostr << endl << "CELL_TYPES " << csize << endl;
  for (unsigned int i = 0; i < csize; i++) {
    fostr << 12 << endl;
  }
  
  return 0;  
}    
