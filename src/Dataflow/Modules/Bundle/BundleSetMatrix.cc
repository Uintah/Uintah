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
 *  BundleSetMatrix.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetMatrix : public Module {
public:
  BundleSetMatrix(GuiContext*);

  virtual ~BundleSetMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guimatrix1name_;
  GuiString     guimatrix2name_;
  GuiString     guimatrix3name_;
  GuiInt        guimatrix1usename_;
  GuiInt        guimatrix2usename_;
  GuiInt        guimatrix3usename_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetMatrix)
  BundleSetMatrix::BundleSetMatrix(GuiContext* ctx)
    : Module("BundleSetMatrix", ctx, Source, "Bundle", "SCIRun"),
      guimatrix1name_(ctx->subVar("matrix1-name")),
      guimatrix2name_(ctx->subVar("matrix2-name")),
      guimatrix3name_(ctx->subVar("matrix3-name")),
      guimatrix1usename_(ctx->subVar("matrix1-usename")),
      guimatrix2usename_(ctx->subVar("matrix2-usename")),
      guimatrix3usename_(ctx->subVar("matrix3-usename")),
      guibundlename_(ctx->subVar("bundlename"))
{
}

BundleSetMatrix::~BundleSetMatrix(){
}

void
BundleSetMatrix::execute()
{
  string matrix1name = guimatrix1name_.get();
  string matrix2name = guimatrix2name_.get();
  string matrix3name = guimatrix3name_.get();
  int matrix1usename = guimatrix1usename_.get();
  int matrix2usename = guimatrix2usename_.get();
  int matrix3usename = guimatrix3usename_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  MatrixHandle fhandle;
  MatrixIPort *ifport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("bundle"))))
    {
      error("Could not find bundle input port");
      return;
    }
        
  // Create a new bundle
  // Since a bundle consists of only handles we can copy
  // it several times without too much memory overhead
  if (iport->get(oldhandle))
    {   // Copy all the handles from the existing bundle
      handle = oldhandle->clone();
    }
  else
    {   // Create a brand new bundle
      handle = scinew Bundle;
    }
        
  // Scan bundle input port 1
  if (!(ifport = static_cast<MatrixIPort *>(get_iport("matrix1"))))
    {
      error("Could not find matrix 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (matrix1usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              matrix1name = name;
              guimatrix1name_.set(name);
              ctx->reset();
            }
        }
      handle->setMatrix(matrix1name,fhandle);
    }

  // Scan matrix input port 2     
  if (!(ifport = static_cast<MatrixIPort *>(get_iport("matrix2"))))
    {
      error("Could not find matrix 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (matrix2usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {    
              matrix2name = name;
              guimatrix2name_.set(name);
              ctx->reset();
            }    
        }

      handle->setMatrix(matrix2name,fhandle);
    }

  // Scan matrix input port 3     
  if (!(ifport = static_cast<MatrixIPort *>(get_iport("matrix3"))))
    {
      error("Could not find matrix 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (matrix3usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              matrix3name = name;
              guimatrix1name_.set(name);
              ctx->reset();
            }
        }
    
      handle->setMatrix(matrix3name,fhandle);
    }
        
  // Now post the output
        
  if (!(oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      error("Could not find bundle output port");
      return;
    }
    
  if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }
        
  oport->send(handle);
}

void
BundleSetMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


