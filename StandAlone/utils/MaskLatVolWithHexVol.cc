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

#include <Core/Datatypes/MaskedLatVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/HexVolField.h>
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

  FieldHandle mlvfH = new MaskedLatVolField<double>(mlvm, lvfh->basis_order());
//  FieldHandle mlvfH = new LatVolField<double>(mlvm, lvfh->basis_order());
  
  BinaryPiostream out_stream(argv[3], Piostream::Write);
  cerr << "Saving MaskedLatVolField... ";
  Pio(out_stream, mlvfH);
  cerr << "done.\n";
  return 0;  
}    
