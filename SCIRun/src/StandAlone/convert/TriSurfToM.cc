//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : TriSurfToM.cc
//    Author : Martin Cole
//    Date   : Wed Jun 14 09:39:11 2006

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
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
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " scirun_field output_basename" << endl;
    return 0;
  }

  FieldHandle handle;
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Couldn't open file "<< argv[1] << ".  Exiting..." << endl;
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file " << argv[1] 
	 << ".  Exiting..." << endl;
    exit(0);
  }
  const TypeDescription *mtd = handle->get_type_description(Field::MESH_TD_E);
  if (mtd->get_name().find("TriSurfMesh") == string::npos) 
  {
    cerr << "Error -- input field didn't have a TriSurfMesh (type_name="
	 << mtd->get_name() << endl;
    exit(0);
  }

  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
  MeshHandle mb = handle->mesh();
  TSMesh *tsm = dynamic_cast<TSMesh *>(mb.get_rep());

  string mname = string(argv[2]) + ".m";
  cerr << "writing file: " << mname << endl;

  ofstream fout(mname.c_str());
  
  if (! fout) {
    cerr << "Could not open " << mname << " for writing." << endl;
    exit(1);
  }

  TSMesh::Node::iterator niter; 
  TSMesh::Node::iterator niter_end; 
  TSMesh::Node::size_type nsize; 
  tsm->begin(niter);
  tsm->end(niter_end);
  tsm->size(nsize);
  unsigned int count = 1;
  while(niter != niter_end) {
    Point p;
    tsm->get_center(p, *niter);
    fout << "Vertex " << count++ << " " 
	 << p.x() << " " << p.y() << " " << p.z() << endl;
     ++niter;
  }

  TSMesh::Face::size_type fsize; 
  TSMesh::Face::iterator fiter; 
  TSMesh::Face::iterator fiter_end; 
  TSMesh::Node::array_type fac_nodes(3);
  tsm->size(fsize);
  tsm->begin(fiter);
  tsm->end(fiter_end);
  count = 1;
  while(fiter != fiter_end) {
    tsm->get_nodes(fac_nodes, *fiter);
    fout << "Face " << count++ << " " 
	 << fac_nodes[0] << " " << fac_nodes[1] << " " << fac_nodes[2] << endl;
    ++fiter;
  }
  return 0;  
}    
