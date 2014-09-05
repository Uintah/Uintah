/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  ConvertTVtoMLV.cc
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/MaskedLatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class ConvertTVtoMLV : public Module {
private:
  GuiInt nx_;
  GuiInt ny_;
  GuiInt nz_;
public:
  ConvertTVtoMLV(GuiContext* ctx);
  virtual ~ConvertTVtoMLV();
  virtual void execute();
};


DECLARE_MAKER(ConvertTVtoMLV)

ConvertTVtoMLV::ConvertTVtoMLV(GuiContext* ctx)
  : Module("ConvertTVtoMLV", ctx, Filter, "ChangeMesh", "SCIRun"),
    nx_(get_ctx()->subVar("nx"), 8),
    ny_(get_ctx()->subVar("ny"), 8),
    nz_(get_ctx()->subVar("nz"), 8)
{
}


ConvertTVtoMLV::~ConvertTVtoMLV()
{
}


void
ConvertTVtoMLV::execute()
{
  FieldHandle ifieldH;
  if (!get_input_handle("TetVolFieldMask", ifieldH)) return;

  typedef TetVolMesh<TetLinearLgn<Point> >    TVMesh;
  typedef TetLinearLgn<Vector>                TFDVectorBasis;
  typedef GenericField<TVMesh, TFDVectorBasis, vector<Vector> > TVV;    

  // we expect that the input field is a TetVolField<Vector>
  if ( ifieldH.get_rep()->get_type_description()->get_name() == 
       TVV::type_name() )
  {
    string msg("Input volume is not a ");
    msg + TVV::type_name();
    error(msg);
    return;
  }                     

  TVV *tv = (TVV *) ifieldH.get_rep();
  TVMesh *tvm = (TVMesh *) tv->get_typed_mesh().get_rep();

  BBox b = tvm->get_bounding_box();

  // break up the volume into cells, with nx/ny/nz specified via the GUI
  int nx = nx_.get();
  int ny = ny_.get();
  int nz = nz_.get();

  typedef MaskedLatVolMesh<HexTrilinearLgn<Point> > MLVMesh;
  typedef HexTrilinearLgn<Vector>                   FDVectorBasis;
  typedef GenericField<MLVMesh, FDVectorBasis, FData3d<Vector, MLVMesh> > MLVF;
  MLVMesh *mlvm =  scinew MLVMesh(nx, ny, nz, b.min(), b.max());
  MLVF *lv = scinew MLVF(mlvm);

  // for each node in the LatVol, check to see if it's inside the TetMesh
  //    if it is, use the weights from get_weights and interpolate
  //    the fiber vectors

  MLVMesh::Node::iterator ib, ie; mlvm->begin(ib); mlvm->end(ie);
  TVMesh::Cell::index_type tet;
  tvm->synchronize(Mesh::LOCATE_E); // for get_weights
  Point p;
  int cnt=0;
  while (ib != ie) {
    mlvm->get_center(p, *ib);
    if (tvm->locate(tet, p)) {
      cnt++;
      TVMesh::Node::array_type nodes;
      vector<double> coords(3, .0L);

      tvm->get_coords(coords, p, tet);
      Vector f1(0,0,0);
      tv->interpolate(f1, coords, tet);
      lv->fdata()[*ib]=f1;

    } else {
      mlvm->mask_cell(MLVMesh::Cell::index_type(mlvm, ib.i_, ib.j_, ib.k_));
    }
    ++ib;
  }

  FieldHandle tmp(lv);
  send_output_handle("LatVolField", tmp);
}

} // End namespace SCIRun
