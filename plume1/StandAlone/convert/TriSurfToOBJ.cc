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
 *  TriSurfToOBJ.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

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
  if (argc != 7) {
    cerr << "Usage: "<<argv[0]<<" scirun_field output_basename color_r color_g color_b Ns\n";
    return 0;
  }

  double red=atof(argv[3]);
  double green=atof(argv[4]);
  double blue=atof(argv[5]);
  double Ns=atof(argv[6]);

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

  MeshHandle mb = handle->mesh();
  TriSurfMesh *tsm = dynamic_cast<TriSurfMesh *>(mb.get_rep());
  char objname[512];
  sprintf(objname, "%s.obj", argv[2]);
  char mtlname[512];
  sprintf(mtlname, "%s.mtl", argv[2]);
  FILE *fobj = fopen(objname, "wt");
  FILE *fmtl = fopen(mtlname, "wt");
  if (!fobj || !fmtl) {
    cerr << "Error - can't open output file "<<objname<<" or "<<mtlname<<"\n";
    exit(0);
  }

  tsm->synchronize(Mesh::NORMALS_E);
  TriSurfMesh::Node::iterator niter; 
  TriSurfMesh::Node::iterator niter_end; 
  TriSurfMesh::Node::size_type nsize; 
  tsm->begin(niter);
  tsm->end(niter_end);
  tsm->size(nsize);
  while(niter != niter_end)
  {
    Point p;
    tsm->get_center(p, *niter);
    fprintf(fobj, "v %lf %lf %lf\n", p.x(), p.y(), p.z());
    Vector n;
    tsm->get_normal(n, *niter);
    fprintf(fobj, "vn %lf %lf %lf\n", n.x(), n.y(), n.z());
    ++niter;
  }
  fprintf(fobj, "usemtl Default\n");  
  TriSurfMesh::Face::size_type fsize; 
  TriSurfMesh::Face::iterator fiter; 
  TriSurfMesh::Face::iterator fiter_end; 
  TriSurfMesh::Node::array_type fac_nodes(3);
  tsm->size(fsize);
  tsm->begin(fiter);
  tsm->end(fiter_end);
  while(fiter != fiter_end)
  {
    tsm->get_nodes(fac_nodes, *fiter);
    fprintf(fobj, "f  %d//%d %d//%d %d//%d\n", 
	    (int)fac_nodes[0]+1, (int)fac_nodes[0]+1,
	    (int)fac_nodes[1]+1, (int)fac_nodes[1]+1,
	    (int)fac_nodes[2]+1, (int)fac_nodes[2]+1);
	    
    ++fiter;
  }
  fclose(fobj);
  fprintf(fmtl,"newmtl Default\n");
  fprintf(fmtl," \tKa 0 0 0\n");
  fprintf(fmtl," \tKd %lf %lf %lf\n", red, green, blue);
  fprintf(fmtl," \tKs 1 1 1\n");
  fprintf(fmtl," \tillum 2\n");
  fprintf(fmtl," \tNs %lf\n", Ns);
  fclose(fmtl);
  return 0;  
}    
