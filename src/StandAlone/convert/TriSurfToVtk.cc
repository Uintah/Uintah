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
 *  TriSurfToVtk.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Init/init.h>

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
    cerr << "Usage: "<<argv[0]<<" scirun_field vtk_poly\n";
    return 0;
  }

  SCIRunInit();

  FieldHandle handle;
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  if (handle->get_type_name(0) != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<handle->get_type_name(0)<<"\n";
    exit(0);
  }
  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;

  MeshHandle mb = handle->mesh();
  TSMesh *tsm = dynamic_cast<TSMesh *>(mb.get_rep());
  FILE *fout = fopen(argv[2], "wt");
  if (!fout) {
    cerr << "Error - can't open input file "<<argv[2]<<"\n";
    exit(0);
  }

  fprintf(fout, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n");
  TSMesh::Node::iterator niter; tsm->begin(niter);
  TSMesh::Node::iterator niter_end; tsm->end(niter_end);
  TSMesh::Node::size_type nsize; tsm->size(nsize);
  cerr << "Writing "<<((unsigned int)nsize)<< " points to "<<argv[2]<<"...\n";
  fprintf(fout, "POINTS %d float\n", (int)nsize);
  while(niter != niter_end)
  {
    Point p;
    tsm->get_center(p, *niter);
    fprintf(fout, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }

  TSMesh::Face::size_type fsize; tsm->size(fsize);
  TSMesh::Face::iterator fiter; tsm->begin(fiter);
  TSMesh::Face::iterator fiter_end; tsm->end(fiter_end);
  TSMesh::Node::array_type fac_nodes(3);
  cerr << "     and "<< ((unsigned int)fsize)<<" faces.\n";
  fprintf(fout, "POLYGONS %d %d\n", (int)fsize, (int)fsize*4);
  while(fiter != fiter_end)
  {
    tsm->get_nodes(fac_nodes, *fiter);
    fprintf(fout, "3 %d %d %d\n", (int)fac_nodes[0], (int)fac_nodes[1], (int)fac_nodes[2]);
    ++fiter;
  }
  fclose(fout);
  return 0;  
}    
