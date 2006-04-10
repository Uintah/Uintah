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
 *  IsoValueController.h: Send a serries of Isovalues to the IsoSurface module
 *                        and collect the resulting surfaces.
 *
 *  Written by:
 *   Allen R. Sanderson
 *   University of Utah
 *   August 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace Fusion {

using namespace SCIRun;

class IsoValueController : public Module {
public:
  IsoValueController(GuiContext*);

  virtual ~IsoValueController();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString gui_IsoValueStr_;

  int nvalues_;
  vector< double > isovalues_;

  double prev_min_;
  double prev_max_;
  int last_orig_generation_;
  int last_tran_generation_;
 
  MatrixHandle mHandleIsoValue_;
  MatrixHandle mHandleIndex_;

  FieldHandle fHandle_orig_;
  FieldHandle fHandle_tran_;
  FieldHandle fHandle_N_1D_;
  FieldHandle fHandle_ND_;

  bool execute_error_;
};

class IsoValueControllerAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(vector<FieldHandle>& fields_ND,
		       vector<FieldHandle>& fields_N_1D,
		       FieldHandle& field_ND,
		       FieldHandle& field_N_1D) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd);
};


template< class IFIELD >
class IsoValueControllerAlgoT : public IsoValueControllerAlgo
{
public:
  //! virtual interface. 
  virtual void execute(vector<FieldHandle>& fields_ND,
		       vector<FieldHandle>& fields_N_1D,
		       FieldHandle& field_ND,
		       FieldHandle& field_N_1D);
};


template< class IFIELD >
void
IsoValueControllerAlgoT<IFIELD>::execute(vector<FieldHandle>& fields_ND,
					 vector<FieldHandle>& fields_N_1D,
					 FieldHandle& field_ND,
					 FieldHandle& field_N_1D)
{
  vector<IFIELD *> tfields_ND(fields_ND.size());
  vector<IFIELD *> tfields_N_1D(fields_N_1D.size());

  for (unsigned int i=0; i<fields_ND.size(); i++) {
    tfields_ND[i]   = (IFIELD *)(fields_ND[i].get_rep());
    tfields_N_1D[i] = (IFIELD *)(fields_N_1D[i].get_rep());
  }

  field_ND   = append_fields(tfields_ND);
  field_N_1D = append_fields(tfields_N_1D);
}

} // End namespace Fusion
