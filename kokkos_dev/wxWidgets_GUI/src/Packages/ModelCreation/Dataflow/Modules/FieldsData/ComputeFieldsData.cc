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
 *  ComputeFieldsData.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
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
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/StringPort.h>

namespace ModelCreation {

using namespace SCIRun;

class ComputeFieldsData : public Module {
  public:
    ComputeFieldsData(GuiContext*);

    virtual ~ComputeFieldsData();

    virtual void execute();

    virtual void tcl_command(GuiArgs&, void*);

  private:
    GuiString guifunction_;     // function code
    GuiString guiformat_;       // scalar, vector, or tensor ?
};


DECLARE_MAKER(ComputeFieldsData)
ComputeFieldsData::ComputeFieldsData(GuiContext* ctx)
  : Module("ComputeFieldsData", ctx, Source, "FieldsData", "ModelCreation"),
  guifunction_(ctx->subVar("function")),
  guiformat_(ctx->subVar("format"))  
{
}

ComputeFieldsData::~ComputeFieldsData(){
}

void ComputeFieldsData::execute()
{
  // Get number of matrix ports with data (the last one is always empty)
  size_t numinputs = numIPorts()-5;
  size_t mstart = 6;
  
  if (numinputs > 23)
  {
    error("This module cannot handle more than 23 input matrices");
    return;
  }
  
  FieldIPort* field_iport = dynamic_cast<FieldIPort *>(getIPort(0));
  if(field_iport == 0)
  {
    error("Could not locate field input port 1");
    return;
  }

  FieldIPort* field_iport2 = dynamic_cast<FieldIPort *>(getIPort(1));
  if(field_iport2 == 0)
  {
    error("Could not locate field input port 2");
    return;
  }

  FieldIPort* field_iport3 = dynamic_cast<FieldIPort *>(getIPort(2));
  if(field_iport3 == 0)
  {
    error("Could not locate field input port 3");
    return;
  }

  // Get the inpuit field
  FieldHandle field;
  field_iport->get(field);
  if (field.get_rep() == 0)
  {
    error("No field was found on input port");
    return;
  }

  FieldHandle field2;
  field_iport2->get(field2);
  if (field2.get_rep() == 0)
  {
    mstart--;
  }
  
  FieldHandle field3;
  field_iport3->get(field3);
  if (field3.get_rep() == 0)
  {
    mstart--;
  }

  ArrayObjectList inputlist(numinputs+mstart,ArrayObject(this));
  ArrayObjectList outputlist(1,ArrayObject(this));

  // The function can be scripted. If a string is found on the input
  // use this one. It will be set in the GIU, after which it is retrieved the
  // normal way.
  StringIPort* function_iport = dynamic_cast<StringIPort *>(getIPort(3));
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
      ctx->reset();
    }
  }

  // Create the DATA object for the function
  // DATA is the data on the field
  size_t k = 0;
  
  if(!(inputlist[k++].create_inputdata(field,"DATA1")))
  {
    error("Failed to read field data");
    return;
  }
  
  if (field2.get_rep())
  {
    if(!(inputlist[k++].create_inputdata(field2,"DATA2")))
    {
      error("Failed to read field data");
      return;
    }
  }

  if (field3.get_rep())
  {
    if(!(inputlist[k++].create_inputdata(field3,"DATA3")))
    {
      error("Failed to read field data");
      return;
    }
  }

  // Create the POS, X,Y,Z, data location objects.  
  if(!(inputlist[k++].create_inputlocation(field,"POS","X","Y","Z")))
  {
    error("Failed to read node/element location data");
    return;
  }

  // Create the ELEMENT object describing element properties
  if(!(inputlist[k++].create_inputelement(field,"ELEMENT")))
  {
    error("Failed to read element data");
    return;
  }

  // Add an object for getting the index and size of the array.
  if(!(inputlist[k++].create_inputindex("INDEX","SIZE")))
  {
    error("Internal error in module");
    return;
  }

  // Loop through all matrices and add them to the engine as well
  char mname = 'A';
  std::string matrixname("A");
  
  for (size_t p = 0; p < numinputs; p++)
  {
    MatrixIPort *iport = dynamic_cast<MatrixIPort *>(getIPort(p+4));
    if(iport == 0)
    {
      error("Could not locate matrix input port");
      return;
    }
    
    MatrixHandle matrix;
    iport->get(matrix);
    if (matrix.get_rep() == 0)
    {
      error("No matrix was found on input port");
      return;      
    }

    matrixname[0] = mname++;    
    if (!(inputlist[p+mstart].create_inputdata(matrix,matrixname)))
    {
      std::ostringstream oss;
      oss << "Input Matrix " << p+1 << "cannot be used as an input";
      error(oss.str());
      return;
    }
  }

  // Check the validity of the input
  int n = 1;
  for (size_t r=0; r<inputlist.size();r++)
  {
    if (n == 1) n = inputlist[r].size();
    if ((inputlist[r].size() != n)&&(inputlist[r].size() != 1))
    {
      if (r < mstart)
      {
        error("Number of data entries does not seem to match number of elements/nodes");
        return;
      }
      else
      {
        std::ostringstream oss;
        oss << "The number of data entries in Field " << r-2 << "does not seem to match the number of data entries in the main field";
        error(oss.str());
        return;
      }
    }
  }

  // Create the engine to compute new data
  ArrayEngine engine(this);
  
  std::string format = guiformat_.get();
  if (format == "") format = "double";
  
  // Add as well the output object
  FieldHandle ofield;
  if(!(outputlist[0].create_outputdata(field,format,"RESULT",ofield)))
  {
    error("Could not allocate output field");
    return;
  }
  
  gui->lock();
  gui->eval(getID()+" update_text");
  gui->unlock();
  
  std::string function = guifunction_.get();
  
  // Actual engine call, which does the dynamic compilation, the creation of the
  // code for all the objects, as well as inserting the function and looping 
  // over every data point
  if (!engine.engine(inputlist,outputlist,function))
  {
    error("An error occured while executing function");
    return;
  }
  
  // If engine succeeded we have a new field at ofield.
  FieldOPort *oport = dynamic_cast<FieldOPort *>(getOPort(0));
  if (oport)
  {
    oport->send(ofield);
  }
}

extern std::string tvm_help_field;

void
 ComputeFieldsData::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("ComputeFieldsData needs a minor command");
    return;
  }

  if( args[1] == "gethelp" )
  {
    gui->lock();
    gui->eval("global " + getID() +"-help");
    gui->eval("set " + getID() + "-help {" + tvm_help_field +"}");
    gui->unlock();
    return;
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace ModelCreation


