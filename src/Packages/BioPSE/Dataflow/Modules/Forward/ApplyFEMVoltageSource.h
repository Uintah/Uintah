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
 *  ApplyFEMVoltageSource.cc:  Builds the RHS of the FE matrix for voltage sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *  Modified by:
 *   Alexei Samsonov, March 2001
 *   Frank B. Sachse, February 2006
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class ApplyFEMVoltageSourceAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle &hField, ColumnMatrix *rhsIn, 
		       SparseRowMatrix *matIn, string bcFlag, 
		       SparseRowMatrix *mat, ColumnMatrix* rhs) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *mtd,
					    const TypeDescription *btd,
					    const TypeDescription *dtd);
};


template<class FIELD>
class ApplyFEMVoltageSourceAlgoT : public ApplyFEMVoltageSourceAlgo {
public:
  ApplyFEMVoltageSourceAlgoT() {}

  virtual ~ApplyFEMVoltageSourceAlgoT() {}
  
  //! Public methods
  virtual void execute(FieldHandle &hField, ColumnMatrix *rhsIn, 
		       SparseRowMatrix *matIn, string bcFlag, 
		       SparseRowMatrix *mat, ColumnMatrix* rhs)
  {
    //-- polling Field for Dirichlet BC
    vector<pair<int, double> > dirBC;
    if (bcFlag=="GroundZero") 
      dirBC.push_back(pair<int, double>(0,0.0));
    else if (bcFlag == "DirSub") 
      hField->get_property("dirichlet", dirBC);

    //! adjusting matrix for Dirichlet BC
    int *idcNz; 
    double *valNz;
    int idcNzsize;
    int idcNzstride;
      
    vector<double> dbc;
    unsigned int idx;
    for(idx = 0; idx<dirBC.size(); ++idx){
      int ni = dirBC[idx].first;
      double val = dirBC[idx].second;
    
      // -- getting column indices of non-zero elements for the current row
      mat->getRowNonzerosNoCopy(ni, idcNzsize, idcNzstride, idcNz, valNz);
    
      // -- updating rhs
      for (int i=0; i<idcNzsize; ++i){
	int j = idcNz?idcNz[i*idcNzstride]:i;
	(*rhs)[j] += - val * valNz[i*idcNzstride]; 
      }
    }
   
    //! zeroing matrix row and column corresponding to the dirichlet nodes
    for(idx = 0; idx<dirBC.size(); ++idx){
      int ni = dirBC[idx].first;
      double val = dirBC[idx].second;
    
      mat->getRowNonzerosNoCopy(ni, idcNzsize, idcNzstride, idcNz, valNz);
      
      for (int i=0; i<idcNzsize; ++i){
	int j = idcNz?idcNz[i*idcNzstride]:i;
	mat->put(ni, j, 0.0);
	mat->put(j, ni, 0.0); 
      }
      
      //! updating dirichlet node and corresponding entry in rhs
      mat->put(ni, ni, 1);
      (*rhs)[ni] = val;
    }
  }
};
    
} // End namespace BioPSE
