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

void FVMBoundCond::setConductivityBC(const Patch* patch, int dwi, CCVariable<double>& conductivity)
{
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for(std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    IntVector oneCell = patch->faceDirection(face);
    std::string bc_kind  = "NotSet";
    int nCells = 0;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;
      bool foundIterator =  getIteratorBCValueBCKind<double>( patch, face, child,
                                "Conductivity", dwi, bc_value, bound_ptr,bc_kind);

      if(foundIterator){
        if(bc_kind == "Dirichlet"){
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            conductivity[*bound_ptr] = bc_value;
          }
          nCells += bound_ptr.size();
        }
      } // end foundIterator if statment
    } // end child loop
  } // end face loop
}

void FVMBoundCond::setESBoundaryConditions(const Patch* patch, int dwi,
                                           CCVariable<Stencil7>& A, CCVariable<double>& rhs,
                                           constSFCXVariable<double>& fcx_conductivity,
                                           constSFCYVariable<double>& fcy_conductivity,
                                           constSFCZVariable<double>& fcz_conductivity)
{
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  IntVector zoffset(0,0,1);
  Vector dx = patch->dCell();

  double a_n = dx.x() * dx.z(); double a_s = dx.x() * dx.z();
  double a_e = dx.y() * dx.z(); double a_w = dx.y() * dx.z();
  double a_t = dx.x() * dx.y(); double a_b = dx.x() * dx.y();
  // double vol = dx.x() * dx.y() * dx.z();

  double n = a_n / dx.y(); double s = a_s / dx.y();
  double e = a_e / dx.x(); double w = a_w / dx.x();
  double t = a_t / dx.z(); double b = a_b / dx.z();

  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for(std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    std::string bc_kind  = "NotSet";
    int nCells = 0;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;
      bool foundIterator =  getIteratorBCValueBCKind<double>( patch, face, child,
    			                      "Voltage", dwi, bc_value, bound_ptr,bc_kind);
      if(foundIterator){
        switch (face) {
          case Patch::xplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr - xoffset);
              if(bc_kind == "Dirichlet"){
                A[c].e = 0;
                rhs[c] -= bc_value * fcx_conductivity[c + xoffset] * e;
              }else if(bc_kind == "Neumann"){
                A[c].e = 0;
                A[c].p += fcx_conductivity[c + xoffset] * e;
                rhs[c] -= bc_value;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::xminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr + xoffset);
              if(bc_kind == "Dirichlet"){
                A[c].w = 0;
                rhs[c] -= bc_value * fcx_conductivity[c] * w;
              }else if(bc_kind == "Neumann"){
                A[c].w = 0;
                A[c].p += fcx_conductivity[c] * w;
                rhs[c] -= bc_value;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr - yoffset);
              if(bc_kind == "Dirichlet"){
                A[c].n = 0;
                rhs[c] -= bc_value * fcy_conductivity[c + yoffset] * n;
              }else if(bc_kind == "Neumann"){
                A[c].n = 0;
                A[c].p += fcy_conductivity[c + yoffset] * n;
                rhs[c] -= bc_value;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr + yoffset);
              if(bc_kind == "Dirichlet"){
                A[c].s = 0;
                rhs[c] -= bc_value * fcy_conductivity[c] * s;
              }else if(bc_kind == "Neumann"){
                A[c].s = 0;
                A[c].p += fcy_conductivity[c] * s;
                rhs[c] -= bc_value;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr - zoffset);
              if(bc_kind == "Dirichlet"){
                A[c].t = 0;
                rhs[c] -= bc_value * fcz_conductivity[c + zoffset] * t;
              }else if(bc_kind == "Neumann"){
                A[c].t = 0;
                A[c].p += fcz_conductivity[c + zoffset] * t;
                rhs[c] -= bc_value;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr + zoffset);
              if(bc_kind == "Dirichlet"){
                A[c].b = 0;
                rhs[c] -= bc_value * fcz_conductivity[c] * b;
              }else if(bc_kind == "Neumann"){
                A[c].b = 0;
                A[c].p = fcz_conductivity[c] * b;
                rhs[c] -= bc_value;
              }
            }
            nCells += bound_ptr.size();
            break;
        } // end switch statment
      } // end foundIterator if statment
    } // end child loop
  } // end face loop
}


void FVMBoundCond::setESPotentialBC(const Patch* patch, int dwi, CCVariable<double>& es_potential)
{
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for(std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    IntVector oneCell = patch->faceDirection(face);
    std::string bc_kind  = "NotSet";
    int nCells = 0;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;
      bool foundIterator = getIteratorBCValueBCKind<double>( patch, face, child,
                                  "Voltage", dwi, bc_value, bound_ptr,bc_kind);
      if(foundIterator){
        if(bc_kind == "Dirichlet"){
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            es_potential[*bound_ptr] = bc_value;
          }
          nCells += bound_ptr.size();
        }else if(bc_kind == "Neumann"){
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            es_potential[*bound_ptr] = bc_value;
          }
          nCells += bound_ptr.size();
        }// end bc_kind if statement
      } // end foundIterator if statement
    } // end child loop
  } // end face loop
}

