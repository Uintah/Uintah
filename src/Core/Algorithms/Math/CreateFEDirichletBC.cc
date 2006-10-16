/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Core/Algorithms/Math/CreateFEDirichletBC.h>

#include <sci_defs/teem_defs.h>

#ifdef HAVE_TEEM
#include <teem/air.h>
#endif


namespace SCIRunAlgo {

using namespace SCIRun;


bool CreateFEDirichletBCAlgo::CreateFEDirichletBC(ProgressReporter *pr,
                          MatrixHandle FEin, MatrixHandle RHSin,
                          MatrixHandle BC, MatrixHandle& FEout, 
                          MatrixHandle& RHSout)
{
  if (FEin.get_rep() == 0)
  {
    pr->error("CreateFEDirichletBC: No FE matrix was found");
    return (false);
  }

  // Check and process FE stiffness matrix
  if (!(FEin->is_sparse()))
  {
    pr->error("CreateFEDirichletBC: This module needs to have a boundary condition");
    return (false);
  }

  if (FEin->ncols() != FEin->nrows())
  {
    pr->error("CreateFEDirichletBC: The Finite Element stiffness matrix is not square");
    return (false);  
  }

  FEout = dynamic_cast<Matrix*>(FEin->sparse());
  FEout.detach();  

  int m = FEin->nrows();
  
  // Check and process RHS vector
  
  if (RHSin.get_rep() == 0)
  {
    RHSout = dynamic_cast<Matrix *>(scinew DenseMatrix(m,1));
    if (RHSout.get_rep() == 0)
    {
      pr->error("CreateFEDirichletBC: Could not allocate RHS matrix");
      return (false);       
    }
  }
  else
  {
    if (RHSin->nrows() * RHSin->ncols() != m)
    {
      pr->error("CreateFEDirichletBC: The RHS dimension does not match the stiffness matrix size");
      return (false);           
    }
  
    RHSout = dynamic_cast<Matrix *>(RHSin->dense());
    RHSout.detach();
    
    if (RHSout.get_rep() == 0)
    {
      pr->error("CreateFEDirichletBC: Could not allocate RHS matrix");
      return (false);      
    }
  }
  
  if (BC.get_rep() == 0)
  {
    pr->error("CreateFEDirichletBC: No BC vector was given");
    return (false);    
  }
  
  if (BC->nrows() * BC->ncols() != m)
  {
    pr->error("CreateFEDirichletBC: The Boundary Condition dimension does not match the stiffness matrix size");
    return (false);           
  }

  MatrixHandle Temp;
  
  Temp = dynamic_cast<Matrix *>(BC->dense());
  BC = Temp;
    
  if (BC.get_rep() == 0)
  {
    pr->error("CreateFEDirichletBC: Could not allocate Boundary Condition matrix");
    return (false);      
  }
  
  double* bc = BC->get_data_pointer();
  double* rhs = RHSout->get_data_pointer();
  int *idcNz; 
  double *valNz;
  int idcNzsize;
  int idcNzstride;
  
#ifdef HAVE_TEEM  
  for (int p=0; p<m;p++)
  {
    if (airExists(bc[p]))
    {
      FEout->getRowNonzerosNoCopy(p, idcNzsize, idcNzstride, idcNz, valNz);
      for (int i=0; i<idcNzsize; ++i)
      {
        int j = idcNz?idcNz[i*idcNzstride]:i;
        rhs[j] += - bc[p] * valNz[i*idcNzstride]; 
      }    
    }
  }
  
  
  for (int p=0; p<m;p++)
  {
    if (airExists(bc[p]))
    {
      FEout->getRowNonzerosNoCopy(p, idcNzsize, idcNzstride, idcNz, valNz);
 
      for (int i=0; i<idcNzsize; ++i)
      {
        int j = idcNz?idcNz[i*idcNzstride]:i;
        FEout->put(p, j, 0.0);
        FEout->put(j, p, 0.0); 
      }
      
      //! updating dirichlet node and corresponding entry in rhs
      FEout->put(p, p, 1.0);
      rhs[p] = bc[p];
    }
  }

#endif  
  
  return (true);
}

} // end namespace
