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
//    File   : RemoveConnectedFaces.cc
//    Author : Martin Cole
//    Date   : Wed Apr  5 08:41:36 2006


#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/Field.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;
typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;

//slow recursive funtion...
void 
find_connected_faces(unsigned face, TSMesh *tsm, 
		     vector<TSMesh::Face::index_type> &connected)
{
  tsm->synchronize(Mesh::FACE_NEIGHBORS_E | Mesh::EDGE_NEIGHBORS_E | 
		   Mesh::EDGES_E);
  
  unsigned n = connected.size();
  if (n % 50 == 0) {
    cout << endl << "status: num connected: " << n << endl;
  } else {
    cout << ".";
  }

  if (connected.end() == find(connected.begin(), connected.end(), face)) {
    connected.push_back(face);
    //cerr << "f " << face << endl;
  }

  TSMesh::Edge::array_type edges(3);
  tsm->get_edges(edges, (TSMesh::Face::index_type)face);
  
  TSMesh::Face::index_type f0;
  
  if (tsm->get_neighbor(f0, face, edges[0]) && 
      connected.end() == find(connected.begin(), connected.end(), f0)) {
    find_connected_faces(f0, tsm, connected);
  }

  TSMesh::Face::index_type f1;
  if (tsm->get_neighbor(f1, face, edges[1]) &&
      connected.end() == find(connected.begin(), connected.end(), f1)) {
    find_connected_faces(f1, tsm, connected);
  }

  TSMesh::Face::index_type f2;
  if (tsm->get_neighbor(f2, face, edges[2]) && 
      connected.end() == find(connected.begin(), connected.end(), f2)) {
    find_connected_faces(f2, tsm, connected);
  }
}

bool node_idx_p = false;
void 
parse_args(int argc, char **argv) {
  if (argc != 5) {
    cerr << "arguments: <in field> <-(n | f)> <index> <out field>"
	 << endl << "where n and f specify the index type (node or face)"
	 << endl;
      exit(3);
  }
  if (string(argv[2]) == "-n") {
    node_idx_p = true;
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

  parse_args(argc, argv);
  
  MeshHandle mb = handle->mesh();
  TSMesh *tsm = dynamic_cast<TSMesh *>(mb.get_rep());
  if (!tsm) {
    cerr << "Input not a TriSurf. Exiting..." << endl;
    exit(3);
  }
  vector<TSMesh::Face::index_type> faces;
  if (node_idx_p) {
    tsm->synchronize(Mesh::NODE_NEIGHBORS_E);
    tsm->get_elems(faces, static_cast<TSMesh::Node::index_type>(atoi(argv[3])));
  } else {
    find_connected_faces(atoi(argv[3]), tsm, faces);
  }
  bool altered = false;
  // remove last index first.
  sort(faces.begin(), faces.end());
  vector<TSMesh::Face::index_type>::reverse_iterator iter  = faces.rbegin();
  while (iter != faces.rend()) {
    int face = *iter++;
    altered |= tsm->remove_face(face);
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
