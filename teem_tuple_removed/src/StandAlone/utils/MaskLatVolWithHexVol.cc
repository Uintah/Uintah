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
 *  MaskLatVolWithHexVol.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/MaskedLatVolMesh.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Geometry/Vector.h>
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
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" LatVol HexVolMask MaskedLatVol\n";
    return 0;
  }

  cerr << "Reading LatVolField... ";
  FieldHandle lvfh;
  Piostream* lvfs=auto_istream(argv[1]);
  if (!lvfs) {
    cerr << "Couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*lvfs, lvfh);
  cerr << "done.\n";
  if (!lvfh.get_rep()) {
    cerr << "Error reading LatVolField from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }

  LatVolMesh *lvm = dynamic_cast<LatVolMesh *>(lvfh->mesh().get_rep());
  if (!lvm) {
    cerr << "Error - input field wasn't a LatVolField\n";
    exit(0);
  }
  Point z(0,0,0);
  Point o(0,0,0);
  

  MaskedLatVolMesh *mlvm = new MaskedLatVolMesh(lvm->get_ni(), lvm->get_nj(), lvm->get_nk(),z,o); 

  mlvm->set_transform(lvm->get_transform());
  
  cerr << "Reading HexVolField... ";
  FieldHandle hvfh;
  Piostream* hvfs=auto_istream(argv[2]);
  if (!hvfs) {
    cerr << "Couldn't open file "<<argv[2]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*hvfs, hvfh);
  cerr << "done.\n";
  if (!hvfh.get_rep()) {
    cerr << "Error reading HexVolField from file "<<argv[2]<<".  Exiting...\n";
    exit(0);
  }
  HexVolMesh *hvm = dynamic_cast<HexVolMesh*>(hvfh->mesh().get_rep());
  if (!hvm) {
    cerr << "Error - input field wasn't a HexVolMesh\n";
    exit(0);
  }

  cerr << "Constructing locate info... ";
  hvm->synchronize(Mesh::LOCATE_E);
  cerr << "done.\n";

  LatVolMesh::Cell::iterator ci, cie;
  MaskedLatVolMesh::Cell::iterator mci, mcie;
  HexVolMesh::Cell::index_type c;
  lvm->begin(ci); lvm->end(cie);
  mlvm->begin(mci); mlvm->end(mcie);
  cerr << "Masking... ";
  while (ci != cie) {
    Point p;
    lvm->get_center(p, *ci);
    if (!hvm->locate(c, p)) mlvm->mask_cell(*mci);
    ++ci;
    ++mci;
  }
  cerr << "done.\n";

  FieldHandle mlvfH = new MaskedLatVolField<double>(mlvm, lvfh->data_at());
//  FieldHandle mlvfH = new LatVolField<double>(mlvm, lvfh->data_at());
  
  BinaryPiostream out_stream(argv[3], Piostream::Write);
  cerr << "Saving MaskedLatVolField... ";
  Pio(out_stream, mlvfH);
  cerr << "done.\n";
  return 0;  
}    
