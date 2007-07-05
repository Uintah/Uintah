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
 *  ComputeDataArray.cc:
 *
 *  Written by:
 *   Jeroen Stinstra
 *
 */

// Include all code for the dynamic engine
#include <Packages/ModelCreation/Core/Algorithms/ArrayObject.h>
#include <Packages/ModelCreation/Core/Algorithms/ArrayEngine.h>

// TensorVectorMath (TVM) is my namespace in which all Scalar, Vector, and Tensor math is defined.
// The classes in this namespace have a definition which is more in line with
// how functions are written in Algebra or Matlab than the native SCIRun Tensor
// and Vector classes. Hence all calculations are performed in this specially
// constructed namespace, to enhance the usability of SCIRun. 
// The TVMHelp system contains an almost complete list of functions that are
// defined in the TensorVectorMath, so when new functions are added, one does 
// not need to update the GUI, but the module dynamically looks up the available
// functions when it is created.

#include <Packages/ModelCreation/Core/Algorithms/TVMHelp.h>
#include <Packages/ModelCreation/Core/Algorithms/TVMMath.h>


#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/String.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/StringPort.h>

namespace ModelCreation {

using namespace SCIRun;

class ComputeDataArray : public Module {
public:
  ComputeDataArray(GuiContext*);

  virtual ~ComputeDataArray();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
    GuiString guifunction_;
    GuiString guiformat_;
};


DECLARE_MAKER(ComputeDataArray)
ComputeDataArray::ComputeDataArray(GuiContext* ctx)
  : Module("ComputeDataArray", ctx, Source, "TensorVectorMath", "ModelCreation"),
  guifunction_(get_ctx()->subVar("function")),
  guiformat_(get_ctx()->subVar("format"))
{
}

ComputeDataArray::~ComputeDataArray(){
}

void ComputeDataArray::execute()
{
  size_t numinputs = (num_input_ports()-3);
  
  ArrayObjectList inputlist(numinputs+1,ArrayObject(this));
  ArrayObjectList outputlist(1,ArrayObject(this));
  
  StringIPort* function_iport = dynamic_cast<StringIPort *>(get_input_port(1));
  if(function_iport == 0)
  {
    error("Could not locate function input port");
    return;
  }

  StringHandle func;
  
  if (function_iport->get(func))
  {
    if (func.get_rep())
    {
      guifunction_.set(func->get());
      get_ctx()->reset();
    }
  }

  if (numinputs > 26)
  {
    error("This module cannot handle more than 26 input matrices");
    return;
  }
  
  char mname = 'A';
  std::string matrixname("A");
  
  int n = 1;

  MatrixIPort *size_iport = dynamic_cast<MatrixIPort *>(get_input_port(0));
  MatrixHandle size;
  size_iport->get(size);
  if (size.get_rep())
  {
    if ((size->ncols() != 1)&&(size->nrows()!=1))
    {
      error("Size input needs to be a 1 by 1 matrix");
      return;
    }
    n = static_cast<int>(size->get(0,0));
    if (n == 0) n = 1;
  }
  
  // Add an object for getting the index and size of the array.
  if(!(inputlist[0].create_inputindex("INDEX","SIZE")))
  {
    error("Internal error in module");
    return;
  } 
   
  for (size_t p = 0; p < numinputs; p++)
  {
    MatrixIPort *iport = dynamic_cast<MatrixIPort *>(get_input_port(p+2));
    MatrixHandle handle;
    iport->get(handle);
    
    if (handle.get_rep())
    {
      if ((handle->ncols()==1)||(handle->ncols()==3)||(handle->ncols()==6)||(handle->ncols()==9))
      {
        matrixname[0] = mname++;
        inputlist[p+1].create_inputdata(handle,matrixname);
      }
      else
      {
        std::ostringstream oss;
        oss << "Input matrix " << p+1 << "is not a valid ScalarArray, VectorArray, or TensorArray";
        error(oss.str());
        return;
      }
      
      if (n > 1) 
      {
        if (n != handle->nrows()&&(handle->nrows() != 1))
        {
          std::ostringstream oss;
          oss << "The number of elements in each ScalarArray, VectorArray, or TensorArray is not equal";
          error(oss.str());
          return;          
        }
      }
      else
      {
        n = handle->nrows();
      }
    }
  }
  
  std::string format = guiformat_.get();
    
  MatrixHandle omatrix;  
  outputlist[0].create_outputdata(n,format,"RESULT",omatrix);
    
  get_gui()->lock();
  get_gui()->eval(get_id()+" update_text");
  get_gui()->unlock();
  
  std::string function = guifunction_.get();
  
  ArrayEngine engine(this);
  if (!engine.engine(inputlist,outputlist,function))
  {
    error("An error occured while executing function");
    return;
  }
  
  MatrixOPort *oport = dynamic_cast<MatrixOPort *>(get_oport(0));
  if (oport)
  {
    oport->send(omatrix);
  }
  
}

extern std::string tvm_help_matrix;

void ComputeDataArray::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("ComputeScalarArray needs a minor command");
    return;
  }

  if( args[1] == "gethelp" )
  {
    get_gui()->lock();
    get_gui()->eval("global " + get_id() +"-help");
    get_gui()->eval("set " + get_id() + "-help {" + tvm_help_matrix +"}");
    get_gui()->unlock();
  }

  else
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace ModelCreation


