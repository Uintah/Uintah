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
 *  BundleGetMatrix.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdString.h>

using namespace SCIRun;
using namespace std;

class BundleGetMatrix : public Module {
public:
  BundleGetMatrix(GuiContext*);

  virtual ~BundleGetMatrix();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guimatrix1name_;
  GuiString             guimatrix2name_;
  GuiString             guimatrix3name_;
  GuiInt                guitransposenrrd1_;
  GuiInt                guitransposenrrd2_;
  GuiInt                guitransposenrrd3_;
  GuiString             guimatrixs_;
};


DECLARE_MAKER(BundleGetMatrix)
  BundleGetMatrix::BundleGetMatrix(GuiContext* ctx)
    : Module("BundleGetMatrix", ctx, Source, "Bundle", "SCIRun"),
      guimatrix1name_(ctx->subVar("matrix1-name")),
      guimatrix2name_(ctx->subVar("matrix2-name")),
      guimatrix3name_(ctx->subVar("matrix3-name")),
      guitransposenrrd1_(ctx->subVar("transposenrrd1")),
      guitransposenrrd2_(ctx->subVar("transposenrrd2")),
      guitransposenrrd3_(ctx->subVar("transposenrrd3")),
      guimatrixs_(ctx->subVar("matrix-selection"))
{

}

BundleGetMatrix::~BundleGetMatrix(){
}

void
BundleGetMatrix::execute()
{
  string matrix1name = guimatrix1name_.get();
  string matrix2name = guimatrix2name_.get();
  string matrix3name = guimatrix3name_.get();
  int transposenrrd1 = guitransposenrrd1_.get();
  int transposenrrd2 = guitransposenrrd2_.get();
  int transposenrrd3 = guitransposenrrd3_.get();
  string matrixlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  MatrixHandle fhandle;
  MatrixOPort *ofport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("bundle"))))
    {
      error("Could not find bundle input port");
      return;
    }

  if (!(iport->get(handle)))
    {   
      warning("No bundle connected to the input port");
      return;
    }


  if (handle.get_rep() == 0)
    {   
      warning("Empty bundle connected to the input port");
      return;
    }


  int nummatrixs = handle->numMatrices();
  for (int p = 0; p < nummatrixs; p++)
    {
      matrixlist += "{" + handle->getMatrixName(p) + "} ";
    }

  guimatrixs_.set(matrixlist);
  ctx->reset();

  if (!(ofport = static_cast<MatrixOPort *>(get_oport("matrix1"))))
    {
      error("Could not find matrix 1 output port");
      return; 
    }
 
  NrrdIPort *niport = static_cast<NrrdIPort *>(getIPort("name1"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          matrix1name = nrrdstring.getstring();
          guimatrix1name_.set(matrix1name);
          ctx->reset();
        }
    } 
 
 
  if (handle->isMatrix(matrix1name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd1) handle->transposeNrrd(true);
      fhandle = handle->getMatrix(matrix1name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<MatrixOPort *>(get_oport("matrix2"))))
    {
      error("Could not find matrix 2 output port");
      return; 
    }
 
  niport = static_cast<NrrdIPort *>(getIPort("name2"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          matrix2name = nrrdstring.getstring();
          guimatrix2name_.set(matrix2name);
          ctx->reset();
        }
    } 
 
  if (handle->isMatrix(matrix2name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd2) handle->transposeNrrd(true);
      fhandle = handle->getMatrix(matrix2name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<MatrixOPort *>(get_oport("matrix3"))))
    {
      error("Could not find matrix 3 output port");
      return; 
    }
 
     
  niport = static_cast<NrrdIPort *>(getIPort("name3"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          matrix3name = nrrdstring.getstring();
          guimatrix3name_.set(matrix3name);
          ctx->reset();
        }
    } 
 
  if (handle->isMatrix(matrix3name))
    {
      handle->transposeNrrd(false);
      if (transposenrrd3) handle->transposeNrrd(true);    
      fhandle = handle->getMatrix(matrix3name);
      ofport->send(fhandle);
    }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      oport->send(handle);
    }
        
}

void
BundleGetMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




