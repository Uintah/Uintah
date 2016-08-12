/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/FVM/FVMBoundCond.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <vector>
#include <iostream>
#include <string>

using namespace Uintah;

FVMBoundCond::FVMBoundCond()
{

}

FVMBoundCond::~FVMBoundCond()
{

}

void FVMBoundCond::setESBoundaryConditions(const Patch* patch, int dwi,
                                           CCVariable<Stencil7>& A, CCVariable<double>& rhs)
{
  Vector dx = patch->dCell();

  double a_n = dx.x() * dx.z(); double a_s = dx.x() * dx.z();
  double a_e = dx.y() * dx.z(); double a_w = dx.y() * dx.z();
  double a_t = dx.x() * dx.y(); double a_b = dx.x() * dx.y();
  // double vol = dx.x() * dx.y() * dx.z();

  double n = a_n / dx.y(); double s = a_s / dx.y();
  double e = a_e / dx.x(); double w = a_w / dx.x();
  double t = a_t / dx.z(); double b = a_b / dx.z();
  double center = n + s + e + w + t + b;

  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for(std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    std::string bc_kind  = "NotSet";
    int nCells = 0;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
    std::cout << "FaceType: " << (int)face << ", Children: " << numChildren << std::endl;

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;
      bool foundIterator =  getIteratorBCValueBCKind<double>( patch, face, child,
    			                      "Voltage", dwi, bc_value, bound_ptr,bc_kind);
      if(foundIterator){
        switch (face) {
          case Patch::xplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr);
              A[c].e = 0;
              rhs[c] -= bc_value*e;
            }
            nCells += bound_ptr.size();
            break;
          case Patch::xminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr);
              A[c].w = 0;
              rhs[c] -= bc_value*w;
             }
             nCells += bound_ptr.size();
             break;
          case Patch::yplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr);
              A[c].n = 0;
              rhs[c] -= bc_value*n;
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr);
              A[c].s = 0;
              rhs[c] -= bc_value*s;
             }
             nCells += bound_ptr.size();
             break;
          case Patch::zplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr);
              A[c].t = 0;
              rhs[c] -= bc_value*t;
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr);
              A[c].b = 0;
              rhs[c] -= bc_value*b;
            }
            nCells += bound_ptr.size();
            break;
          case Patch::numFaces:
            break;
          case Patch::invalidFace:
            break;
        } // end switch statment
      } // end foundIterator if statment
    } // end child loop
  } // end face loop

  /**
  // x minus face cells
  if(c.x() == low_idx.x() && low_offset.x() == 1){
            A_tmp.w = 0;
            rhs[c] -= w;
  }

  // x plus face cells
  if(c.x() == high_idx.x()-1 && high_offset.x() == 1){
            A_tmp.e = 0;
            rhs[c] -= 0;
  }

  // y minus face cells
  if(c.y() == low_idx.y() && low_offset.y() == 1){
            A_tmp.s = 0;
            rhs[c] -= 0;
  }

  // y plus face cells
  if(c.y() == high_idx.y()-1 && high_offset.y() == 1){
            A_tmp.n = 0;
            rhs[c] -= 0;
  }

  // z minus face cells
  if(c.z() == low_idx.z() && low_offset.z() == 1){
            A_tmp.b = 0;
            rhs[c] -= 0;
  }

  // z plus face cells
  if(c.z() == high_idx.z()-1 && high_offset.z() == 1){
            A_tmp.t = 0;
            rhs[c] -= 0;
  }

  **/

}
