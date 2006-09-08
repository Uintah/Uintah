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
 *  ClipFieldByFunction.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

// Include all code for the dynamic engine
#include <Core/Algorithms/ArrayMath/ArrayObject.h>
#include <Core/Algorithms/ArrayMath/ArrayEngine.h>

// TensorVectorMath (TVM) is my namespace in which all Scalar, Vector, and Tensor math is defined.
// The classes in this namespace have a definition which is more in line with
// how functions are written in Algebra or Matlab than the native SCIRun Tensor
// and Vector classes. Hence all calculations are performed in this specially
// constructed namespace, to enhance the usability of SCIRun. 
// The TVMHelp system contains an almost complete list of functions that are
// defined in the TensorVectorMath, so when new functions are added, one does 
// not need to update the GUI, but the module dynamically looks up the available
// functions when it is created.

#include <Core/Algorithms/ArrayMath/ArrayEngineHelp.h>
#include <Core/Algorithms/ArrayMath/ArrayEngineMath.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/String.h>

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/StringPort.h>

#include <Core/Algorithms/Fields/FieldsAlgo.h>

namespace ModelCreation {

using namespace SCIRun;

class ClipFieldByFunction : public Module {
  public:
    ClipFieldByFunction(GuiContext*);

    virtual void execute();
    virtual void tcl_command(GuiArgs&, void*);
    
  private:
    GuiString guifunction_;  
};


DECLARE_MAKER(ClipFieldByFunction)
ClipFieldByFunction::ClipFieldByFunction(GuiContext* ctx)
  : Module("ClipFieldByFunction", ctx, Source, "FieldsCreate", "ModelCreation"),
    guifunction_(get_ctx()->subVar("function"))
{
}


void ClipFieldByFunction::execute()
{
  // Define local handles of data objects:
  FieldHandle field;
  StringHandle func;
  std::vector<MatrixHandle> matrices;

  // Get the new input data:    
  if (!(get_input_handle("Field",field,true))) return;
  if ( get_input_handle("Function",func,false))
  {
    if (func.get_rep())
    {
      guifunction_.set(func->get());
      get_ctx()->reset();  
    }
  }
  get_dynamic_input_handles("Array",matrices,false);


  get_gui()->lock();
  get_gui()->eval(get_id()+" update_text");
  get_gui()->unlock();

  // Only do work if needed:
  if (inputs_changed_ || guifunction_.changed() || !oport_cached("Field") || !oport_cached("Mapping"))
  {

    // Get number of matrix ports with data (the last one is always empty)
    size_t numinputs = matrices.size();
    if (numinputs > 23)
    {
      error("This module cannot handle more than 23 input matrices");
      return;
    }
    
    SCIRunAlgo::ArrayObjectList inputlist(numinputs+4,SCIRunAlgo::ArrayObject(this));
    SCIRunAlgo::ArrayObjectList outputlist(1,SCIRunAlgo::ArrayObject(this));
    
    // Create the DATA object for the function
    // DATA is the data on the field
    if(!(inputlist[0].create_inputdata(field,"DATA")))
    {
      error("Failed to read field data");
      return;
    }

    // Create the POS, X,Y,Z, data location objects.  
    if(!(inputlist[1].create_inputlocation(field,"POS","X","Y","Z")))
    {
      error("Failed to read node/element location data");
      return;
    }

    // Create the ELEMENT object describing element properties
    if(!(inputlist[2].create_inputelement(field,"ELEMENT")))
    {
      error("Failed to read element data");
      return;
    }

    // Add an object for getting the index and size of the array.
    if(!(inputlist[3].create_inputindex("INDEX","SIZE")))
    {
      error("Internal error in module");
      return;
    }

    // Loop through all matrices and add them to the engine as well
    char mname = 'A';
    std::string matrixname("A");
    
    for (size_t p = 0; p < numinputs; p++)
    {
      if (matrices[p].get_rep() == 0)
      {
        error("No matrix was found on input port");
        return;      
      }
      matrixname[0] = mname++;    
      if (!(inputlist[p+4].create_inputdata(matrices[p],matrixname)))
      {
        std::ostringstream oss;
        oss << "Input matrix " << p+1 << "is not a valid ScalarArray, VectorArray, or TensorArray";
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
        if (r < 4)
        {
          error("Number of data entries does not seem to match number of elements/nodes");
          return;
        }
        else
        {
          std::ostringstream oss;
          oss << "The number of rows in matrix " << r-2 << "does not seem to match the number of datapoints in the field";
          error(oss.str());
        }
      }
    }

    // Create the engine to compute new data
    SCIRunAlgo::ArrayEngine engine(this);
    
    // Add as well the output object
    MatrixHandle selmask;
    if(!(outputlist[0].create_outputdata(n,"Scalar","RESULT",selmask)))
    {
      return;
    }
    
    
    // Define the function we are using for clipping:
    std::string function = std::string("RESULT = ") + guifunction_.get();
    
    // Actual engine call, which does the dynamic compilation, the creation of the
    // code for all the objects, as well as inserting the function and looping 
    // over every data point
    if (!engine.engine(inputlist,outputlist,function))
    {
      error("An error occured while executing selection function");
      return;
    }
    
    // Now do the clipping
    SCIRunAlgo::FieldsAlgo fieldalgo(this);

    FieldHandle output;
    MatrixHandle interpolant;
    if(!(fieldalgo.ClipFieldBySelectionMask(field,output,selmask,interpolant)))
    {
      error("Clipping algorithm failed");
      return;
    }

    send_output_handle("Field",output,false);
    send_output_handle("Mapping",interpolant,false);
  }
}


void
ClipFieldByFunction::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("ClipFieldByFunction needs a minor command");
    return;
  }

  if( args[1] == "gethelp" )
  {
    TensorVectorMath::ArrayEngineHelp Help;
    get_gui()->lock();
    get_gui()->eval("global " + get_id() +"-help");
    get_gui()->eval("set " + get_id() + "-help {" + Help.gethelp(true) +"}");
    get_gui()->unlock();
    return;
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}


} // End namespace ModelCreation


