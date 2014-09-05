/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  OldSFRGtoNewLatticeVol.cc: Converter
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <FieldConverters/Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;

int main(int argc, char **argv) {
  VectorFieldHandle handle;
  
  if (argc !=3) {
    cerr << "Usage: "<<argv[0]<<" OldVFRG NewLatticeVol\n";
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Error - couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading VectorField from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }

  VectorFieldRG *base=dynamic_cast<VectorFieldRG*>(handle.get_rep());
  if (!base) {
    cerr << "Error - input Field wasn't an VFRG.\n";
    exit(0);
  }

  FieldHandle fH;

  // create a lattice Field and give it a handle
  LatticeVol<Vector> *lf = new LatticeVol<Vector>(Field::NODE);
  fH = FieldHandle(lf);

  // create a mesh identical to the base mesh
  LatVolMesh *mesh  = 
    dynamic_cast<LatVolMesh*>(lf->get_typed_mesh().get_rep());
  mesh->set_nx(base->nx);
  mesh->set_ny(base->ny);
  mesh->set_nz(base->nz);
  mesh->set_min(base->get_point(0,0,0));
  mesh->set_max(base->get_point(base->nx-1,base->ny-1,base->nz-1));
  cerr << "node index space extents = " 
       << base->nx << ", " << base->ny << ", " << base->nz << endl;
  cerr << "object space extents     = "
       << mesh->get_min() << ", " << mesh->get_max() << endl;

  // get the storage for the data, and copy base's data into it
  FData3d<Vector> &fdata = lf->fdata();
  fdata.newsize(base->nx,base->ny,base->nz);
  LatVolMesh::NodeIter iter = mesh->node_begin();
  int i=0,j=0,k=0;
  while (iter != mesh->node_end()) {
    fdata[*iter] = base->grid(i,j,k);
    ++iter;
    ++i;
    if (i>=base->nx) {
      i=0; ++j;
      if (j>=base->ny) {
	j=0; ++k;
      }
    }
  }

  TextPiostream out_stream(argv[2], Piostream::Write);
  Pio(out_stream, fH);

  return 0;  
}    
