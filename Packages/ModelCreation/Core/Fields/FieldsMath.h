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


#ifndef MODELCREATION_CORE_FIELDS_FIELDMATH_H
#define MODELCREATION_CORE_FIELDS_FIELDMATH_H 1

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Module.h>

// Dynamic code that already exist
// We wrap it only by a function call
// A more clean solution is to transfer all
// dynamic code to Algorithms


#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldsMath {

  public:
    FieldsMath(Module* module);       // if you want the module to pop up some
                                      // error message of dynamically compiled 
                                      // user supplied code
    FieldsMath(ProgressReporter* pr); // normal case
    virtual ~FieldsMath();

    // Funtions borrow from Core of SCIRun
    bool FieldBoundary(FieldHandle input, FieldHandle& output, MatrixHandle &interpolant);
    bool ApplyMappingMatrix(FieldHandle input, FieldHandle& output, MatrixHandle interpolant, FieldHandle datafield);
    bool Unstructure(FieldHandle input,FieldHandle& output);
    bool ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, int newbasis_order);
    
    // ManageFieldData split into two parts
    // Need to upgrade code for these when we are done with HO integration
    bool SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data);
    bool GetFieldData(FieldHandle input, MatrixHandle& data);
	
    // Due to some oddity in the FieldDesign information like this cannot be queried directly
    bool GetFieldInfo(FieldHandle input, int& numnodes, int& numelems);
    
    bool ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle SelectionMask,MatrixHandle &interpolant);
    bool DistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object);
    bool SignedDistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object);
    
    bool IsInsideSurfaceField(FieldHandle input, FieldHandle& output, FieldHandle object);
    bool IsInsideVolumeField(FieldHandle input, FieldHandle& output, FieldHandle object);

    // Change where the data is located
    bool FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method);
    bool FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method);

    // Check properties of surface field
    bool IsClosedSurface(FieldHandle input);
    bool IsClockWiseSurface(FieldHandle input);
    bool IsCounterClockWiseSurface(FieldHandle input);

    // More specialized functions
    // Function to split a field into different unconnected regions
    bool SplitFieldByElementData(FieldHandle input, FieldHandle& output);
    bool MappingMatrixToField(FieldHandle input, FieldHandle& output, MatrixHandle mappingmatrix);

  private:
    Module* module_;
    ProgressReporter* pr_;
    
    inline void error(std::string error);
    inline void warning(std::string warning);
    inline void remark(std::string remark);

};

inline void FieldsMath::error(std::string error)
{
  if (pr_) pr_->error(error); else std::cout << "ERROR: " << error << std::endl;
}

inline void FieldsMath::warning(std::string warning)
{
  if (pr_) pr_->warning(warning); else std::cout << "WARNING: " << warning << std::endl;
}

inline void FieldsMath::remark(std::string remark)
{
  if (pr_) pr_->remark(remark); else std::cout << "REMARK: " << remark << std::endl;
}


} // ModelCreation

#endif
