
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
 * MappingMatrixToMaskVector
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

class MappingMatrixToMaskVector : public Module {
public:
  MappingMatrixToMaskVector(GuiContext*);
  virtual ~MappingMatrixToMaskVector();
  virtual void	execute();
  virtual void	tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(MappingMatrixToMaskVector)
MappingMatrixToMaskVector::MappingMatrixToMaskVector(GuiContext* ctx)
  : Module("MappingMatrixToMaskVector", ctx, Source, "Math", "SCIRun")
{
}

MappingMatrixToMaskVector::~MappingMatrixToMaskVector()
{
}

void
MappingMatrixToMaskVector::execute()
{
  update_state(Module::JustStarted);
  MatrixIPort * matrix_iport_ = (MatrixIPort *)get_iport("MappingMatrix");
  NrrdOPort * nrrd_oport_ = (NrrdOPort *)get_oport("MaskVector");

  update_state(Module::NeedData);
  
  MatrixHandle matrixH;
  matrix_iport_->get(matrixH);
  if (!matrixH.get_rep())
    throw "No input MappingMatrixr";

  SparseRowMatrix *matrix = dynamic_cast<SparseRowMatrix *>(matrixH.get_rep());
  if (!matrix)
    throw "Input matrix not sparse";

  const unsigned int dim = matrix->ncols();

  NrrdDataHandle nrrdH = scinew NrrdData;
  Nrrd *nrrd = nrrdH->nrrd;
  nrrdAlloc(nrrd, nrrdTypeUChar, 1, dim);
  unsigned char *mask = (unsigned char *)nrrd->data;
  memset(mask, 0, dim*sizeof(unsigned char));
  int *rr = matrix->rows;
  int *cc = matrix->columns;
  double *data = matrix->a;
  for (unsigned int i = 0; i < matrix->nrows(); ++i) {
    if (rr[i+1] == rr[i]) continue; // No entires on this row
    int col = cc[rr[i]];
    if (data[rr[i]] > 0.0) {
      cerr << "Row: " << i << " Col: " << col << std::endl;
      mask[col] = 1;
    }
  }

  nrrd_oport_->send(nrrdH);
}

void
MappingMatrixToMaskVector::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


