
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
 *  MaskVectorToMappingMatrix
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   August, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/MatrixPort.h>

namespace SCIRun {

using namespace SCIRun;

class MaskVectorToMappingMatrix : public Module {
public:
  MaskVectorToMappingMatrix(GuiContext*);
  virtual ~MaskVectorToMappingMatrix();
  virtual void	execute();
  virtual void	tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(MaskVectorToMappingMatrix)
MaskVectorToMappingMatrix::MaskVectorToMappingMatrix(GuiContext* ctx)
  : Module("MaskVectorToMappingMatrix", ctx, Source, "Math", "SCIRun")
{
}

MaskVectorToMappingMatrix::~MaskVectorToMappingMatrix()
{
}

void
MaskVectorToMappingMatrix::execute()
{
  update_state(Module::JustStarted);
  NrrdIPort * nrrd_iport_ = (NrrdIPort *)get_iport("MaskVector");
  MatrixOPort * matrix_oport_ = (MatrixOPort *)get_oport("MappingMatrix");
  update_state(Module::NeedData);
  
  NrrdDataHandle nrrdH;
  nrrd_iport_->get(nrrdH);
  if (!nrrdH.get_rep())
    throw "No input MaskVector";

  Nrrd *nrrd = nrrdH->nrrd;
  if (!nrrd)
    throw "Input MaskVector Nrrd empty";

  if (nrrd->type != nrrdTypeUChar) 
    throw "Input MaskVector not Unsigned Char";


  const unsigned int dim = nrrd->axis[0].size;
  int *rr = scinew int[dim+1];
  int *cc = scinew int[dim];
  double *data = scinew double[dim];
  const unsigned char *mask = (const unsigned char *)nrrd->data;
  for (unsigned int i = 0; i < dim; ++i) {
    rr[i] = i;
    cc[i] = i;
    data[i] = mask[i]?1.0:0.0;
  }
  
  rr[dim] = dim;  

  SparseRowMatrix *matrix = scinew SparseRowMatrix(dim, dim, rr, cc, dim, data);
  matrix_oport_->send(matrix);
}

void
 MaskVectorToMappingMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


