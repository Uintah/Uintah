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
 *  SearchGridInfo.cc:  Make an ImageField that fits the source field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Tensor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class SearchGridInfo : public Module
{
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef NoDataBasis<Tensor>             NDTBasis;
  typedef NoDataBasis<Vector>             NDVBasis;
  typedef NoDataBasis<double>             NDDBasis;
  typedef ConstantBasis<Tensor>             CBTBasis;
  typedef ConstantBasis<Vector>             CBVBasis;
  typedef ConstantBasis<double>             CBDBasis;
  typedef HexTrilinearLgn<Tensor>             LBTBasis;
  typedef HexTrilinearLgn<Vector>             LBVBasis;
  typedef HexTrilinearLgn<double>             LBDBasis;
  typedef GenericField<LVMesh, NDTBasis,  
		       FData3d<Tensor, LVMesh> > LVFieldNDT;
  typedef GenericField<LVMesh, NDVBasis,  
		       FData3d<Vector, LVMesh> > LVFieldNDV;
  typedef GenericField<LVMesh, NDDBasis,  
		       FData3d<double, LVMesh> > LVFieldNDD;
  typedef GenericField<LVMesh, CBTBasis,  
		       FData3d<Tensor, LVMesh> > LVFieldCBT;
  typedef GenericField<LVMesh, CBVBasis,  
		       FData3d<Vector, LVMesh> > LVFieldCBV;
  typedef GenericField<LVMesh, CBDBasis,  
		       FData3d<double, LVMesh> > LVFieldCBD;
  typedef GenericField<LVMesh, LBTBasis,  
		       FData3d<Tensor, LVMesh> > LVFieldT;
  typedef GenericField<LVMesh, LBVBasis,  
		       FData3d<Vector, LVMesh> > LVFieldV;
  typedef GenericField<LVMesh, LBDBasis,  
		       FData3d<double, LVMesh> > LVField;

  SearchGridInfo(GuiContext* ctx);
  virtual ~SearchGridInfo();

  virtual void execute();
};


DECLARE_MAKER(SearchGridInfo)

SearchGridInfo::SearchGridInfo(GuiContext* ctx)
  : Module("SearchGridInfo", ctx, Filter, "FieldsOther", "SCIRun")
{
}



SearchGridInfo::~SearchGridInfo()
{
}


void
SearchGridInfo::execute()
{
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input Field", ifieldhandle)) return;

  // Create blank mesh.
  int sizex, sizey, sizez;
  Transform trans;

  if (!ifieldhandle->mesh()->get_search_grid_info(sizex, sizey, sizez, trans))
  {
    error("This field has no search grid information available.");
    return;
  }

  Point minb(0.0, 0.0, 0.0);
  Point maxb(1.0, 1.0, 1.0);
  LVMesh::handle_type mesh = scinew LVMesh(sizex+1, sizey+1, sizez+1,
                                           minb, maxb);
  mesh->transform(trans);
  
  // Create Image Field.
  FieldHandle ofh;
  LVFieldCBD *lvf = scinew LVFieldCBD(mesh);
  LVFieldCBD::fdata_type::iterator itr = lvf->fdata().begin();
  while (itr != lvf->fdata().end())
  {
    *itr = 0.0;
    ++itr;
  }   
  ofh = lvf;

  send_output_handle("Output Sample Field", ofh);
}


} // End namespace SCIRun

