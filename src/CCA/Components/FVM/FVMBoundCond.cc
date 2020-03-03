/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <Core/Grid/Patch.h>

#include <vector>
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
      bool foundIterator = getIteratorBCValueBCKind<double>( patch, face, child,
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
                rhs[c] -= bc_value * a_e;
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
                rhs[c] -= bc_value * a_w;
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
                rhs[c] -= bc_value * a_n;
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
                rhs[c] -= bc_value * a_s;
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
                rhs[c] -= bc_value * a_t;
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
                A[c].p += fcz_conductivity[c] * b;
                rhs[c] -= bc_value * a_b;
              }
            }
            nCells += bound_ptr.size();
            break;
          default:
            break;
        } // end switch statment
      } // end foundIterator if statment
    } // end child loop
  } // end face loop
}

void FVMBoundCond::setESPotentialBC(const Patch* patch, int dwi,
                                    CCVariable<double> &es_potential)
{
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector dx = patch->dCell();
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  IntVector zoffset(0,0,1);

  IntVector xoffset2(2,0,0);
  IntVector yoffset2(0,2,0);
  IntVector zoffset2(0,0,2);

  IntVector min = patch->getExtraCellLowIndex();
  IntVector max = patch->getExtraCellHighIndex();

  int xmin = -99999999;
  int xmax = -99999999;
  int ymin = -99999999;
  int ymax = -99999999;
  int zmin = -99999999;
  int zmax = -99999999;

  bool xminus_bd = false;
  bool yminus_bd = false;
  bool zminus_bd = false;

  bool xplus_bd = false;
  bool yplus_bd = false;
  bool zplus_bd = false;


  if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
    xminus_bd = true;
    xmin = min.x();
  }

  if(patch->getBCType(Patch::yminus) != Patch::Neighbor){
    yminus_bd = true;
    ymin = min.y();
  }

  if(patch->getBCType(Patch::zminus) != Patch::Neighbor){
    zminus_bd = true;
    zmin = min.z();
  }

  if(patch->getBCType(Patch::xplus) != Patch::Neighbor){
    xplus_bd = true;
    xmax = max.x() - 1;
  }

  if(patch->getBCType(Patch::yplus) != Patch::Neighbor){
    yplus_bd = true;
    ymax = max.y() - 1;
  }

  if(patch->getBCType(Patch::zplus) != Patch::Neighbor){
    zplus_bd = true;
    zmax = max.z() - 1;
  }

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
        switch (face) {
          case Patch::xplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.y() == ymin || c.y() == ymax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = bc_value*dx.x();
                  potential += es_potential[c - xoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::xminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[*bound_ptr] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.y() == ymin || c.y() == ymax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = bc_value*dx.x();
                  potential += es_potential[c + xoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.x() == xmin || c.x() == xmax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = bc_value*dx.y();
                  potential += es_potential[c - yoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                es_potential[c] = bc_value;
                if(c.x() == xmin || c.x() == xmax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = bc_value*dx.y();
                  potential += es_potential[c + yoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.x() == xmin || c.x() == xmax || c.y() == ymin || c.y() == ymax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = bc_value*dx.z();
                  potential += es_potential[c - zoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.x() == xmin || c.x() == xmax || c.y() == ymin || c.y() == ymax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = bc_value*dx.z();
                  potential += es_potential[c + zoffset];
                  es_potential[c] = potential;
                };
              }
            }
            nCells += bound_ptr.size();
            break;
          default:
            break;
        }// end switch statement
      } // end foundIterator if statement
    } // end child loop
  } // end face loop

  if(xminus_bd && yminus_bd){
    IntVector c(min.x(), min.y(), min.z());
    for(int i = min.z(); i < max.z(); i++){
      if(i == min.z()){
        es_potential[c] = es_potential[c + xoffset + yoffset + zoffset]
                        - (es_potential[c + xoffset2 + yoffset + zoffset]
                        -  es_potential[c + yoffset + zoffset]
                        +  es_potential[c + xoffset + yoffset2 + zoffset]
                        -  es_potential[c + xoffset + zoffset]
                        +  es_potential[c + xoffset + yoffset + zoffset2]
                        -  es_potential[c + xoffset + yoffset])/2;
      }else if(i == (max.z()-1)){
        es_potential[c] = es_potential[c + xoffset + yoffset - zoffset]
                        + (es_potential[c + yoffset - zoffset]
                        -  es_potential[c + xoffset2 + yoffset - zoffset]
                        -  es_potential[c + xoffset + yoffset2 - zoffset]
                        +  es_potential[c + xoffset - zoffset]
                        -  es_potential[c + xoffset + yoffset - zoffset2]
                        +  es_potential[c + xoffset + yoffset])/2;

      }else{
        es_potential[c] = es_potential[c + xoffset + yoffset]
                        - (es_potential[c + xoffset2 + yoffset]
                        -  es_potential[c + yoffset]
                        +  es_potential[c + xoffset + yoffset2]
                        -  es_potential[c + xoffset])/2;
      }
      c += zoffset;
    }
  }

  if(xplus_bd && yminus_bd){
    IntVector c(max.x()-1, min.y(), min.z());
    for(int i = min.z(); i < max.z(); i++){
      if(i == min.z()){
        es_potential[c] = es_potential[c - xoffset + yoffset + zoffset]
                        + (es_potential[c + yoffset + zoffset]
                        -  es_potential[c - xoffset2 + yoffset + zoffset]
                        -  es_potential[c - xoffset + yoffset2 + zoffset]
                        +  es_potential[c - xoffset + zoffset]
                        -  es_potential[c - xoffset + yoffset + zoffset2]
                        +  es_potential[c - xoffset + yoffset])/2;
      }else if(i == (max.z()-1)){
        es_potential[c] = es_potential[c - xoffset + yoffset - zoffset]
                        + (es_potential[c + yoffset - zoffset]
                        -  es_potential[c - xoffset2 + yoffset - zoffset]
                        -  es_potential[c - xoffset + yoffset2 - zoffset]
                        +  es_potential[c - xoffset - zoffset]
                        +  es_potential[c - xoffset + yoffset]
                        -  es_potential[c - xoffset + yoffset - zoffset2])/2;
      }else{
        es_potential[c] = es_potential[c - xoffset + yoffset]
                        + (es_potential[c + yoffset]
                        -  es_potential[c - xoffset2 + yoffset]
                        -  es_potential[c - xoffset + yoffset2]
                        +  es_potential[c - xoffset])/2;
      }
      c += zoffset;
    }
  }
  if(xplus_bd && yplus_bd){
    IntVector c(max.x()-1, max.y()-1, min.z());
    for(int i = min.z(); i < max.z(); i++){
      if(i == min.z()){
        es_potential[c] = es_potential[c - xoffset - yoffset + zoffset]
                        + (es_potential[c - yoffset + zoffset]
                        -  es_potential[c - xoffset2 - yoffset + zoffset]
                        +  es_potential[c - xoffset + zoffset]
                        -  es_potential[c - xoffset - yoffset2 + zoffset]
                        -  es_potential[c - xoffset - yoffset + zoffset2]
                        +  es_potential[c - xoffset - yoffset])/2;
      }else if(i == (max.z()-1)){
        es_potential[c] = es_potential[c - xoffset - yoffset - zoffset]
                        + (es_potential[c - yoffset - zoffset]
                        -  es_potential[c - xoffset2 - yoffset - zoffset]
                        +  es_potential[c - xoffset - zoffset]
                        -  es_potential[c - xoffset - yoffset2 - zoffset]
                        +  es_potential[c - xoffset - yoffset]
                        -  es_potential[c - xoffset - yoffset - zoffset2])/2;
      }else{
        es_potential[c] = es_potential[c - xoffset - yoffset]
                        + (es_potential[c - yoffset]
                        -  es_potential[c - xoffset2 - yoffset]
                        +  es_potential[c - xoffset]
                        -  es_potential[c - xoffset - yoffset2])/2;
      }
      c += zoffset;
    }
  }
  if(xminus_bd && yplus_bd){
    IntVector c(min.x(), max.y()-1, min.z());
    for(int i = min.z(); i < max.z(); i++){
      if(i == min.z()){
        es_potential[c] = es_potential[c + xoffset - yoffset + zoffset]
                        + (es_potential[c - yoffset + zoffset]
                        -  es_potential[c + xoffset2 - yoffset + zoffset]
                        +  es_potential[c + xoffset + zoffset]
                        -  es_potential[c + xoffset - yoffset2 + zoffset]
                        +  es_potential[c + xoffset - yoffset]
                        -  es_potential[c + xoffset - yoffset + zoffset2])/2;
      }else if(i == (max.z()-1)){
        es_potential[c] = es_potential[c + xoffset - yoffset - zoffset]
                        + (es_potential[c - yoffset - zoffset]
                        -  es_potential[c + xoffset2 - yoffset - zoffset]
                        +  es_potential[c + xoffset - zoffset]
                        -  es_potential[c + xoffset - yoffset2 - zoffset]
                        +  es_potential[c + xoffset - yoffset]
                        -  es_potential[c + xoffset - yoffset - zoffset2])/2;
      }else{
        es_potential[c] = es_potential[c + xoffset - yoffset]
                        + (es_potential[c - yoffset]
                        -  es_potential[c + xoffset2 - yoffset]
                        +  es_potential[c + xoffset]
                        -  es_potential[c + xoffset - yoffset2])/2;
      }
      c += zoffset;
    }
  }
  if(yminus_bd && zminus_bd){
    IntVector c(min.x(), min.y(), min.z());
    for(int i = min.x()+1; i < max.x(); i++){
      es_potential[c] =  es_potential[c + yoffset  + zoffset]
                      - (es_potential[c + yoffset2 + zoffset]
                      -  es_potential[c + zoffset]
                      +  es_potential[c + yoffset  + zoffset2]
                      -  es_potential[c + yoffset])/2;
      c += xoffset;
    }
  }
  if(yminus_bd && zplus_bd){
    IntVector c(min.x(), min.y(), max.z()-1);
    for(int i = min.x()+1; i < max.x(); i++){
      es_potential[c] =  es_potential[c + yoffset  - zoffset]                       + (es_potential[c - zoffset]
                      -  es_potential[c + yoffset2 - zoffset]
                      +  es_potential[c + yoffset]
                      -  es_potential[c + yoffset  - zoffset2])/2;
      c += xoffset;
    }
  }
  if(yplus_bd && zplus_bd){
    IntVector c(min.x(), max.y()-1, max.z()-1);
    for(int i = min.x()+1; i < max.x(); i++){
      es_potential[c] = es_potential[c - yoffset - zoffset]
                      + (es_potential[c - zoffset]
                      -  es_potential[c - yoffset2 - zoffset]
                      +  es_potential[c - yoffset]
                      -  es_potential[c - yoffset - zoffset2])/2;
      c += xoffset;
    }
  }
  if(yplus_bd && zminus_bd){
    IntVector c(min.x(), max.y()-1, min.z());
    for(int i = min.x()+1; i < max.x(); i++){
      es_potential[c] = es_potential[c - yoffset + zoffset]
                      + (es_potential[c + zoffset]
                      -  es_potential[c - yoffset2 + zoffset]
                      -  es_potential[c - yoffset + zoffset2]
                      +  es_potential[c - yoffset])/2;
      c += xoffset;
    }
  }
  if(zminus_bd && xminus_bd){
    IntVector c(min.x(), min.y(), min.z());
    for(int i = min.y()+1; i < max.y(); i++){
      es_potential[c] =  es_potential[c + zoffset  + xoffset]
                      - (es_potential[c + zoffset2 + xoffset]
                      -  es_potential[c + xoffset]
                      +  es_potential[c + zoffset  + xoffset2]
                      -  es_potential[c + zoffset])/2;
      c += yoffset;
    }
  }
  if(zminus_bd && xplus_bd){
    IntVector c(max.x()-1, min.y(), min.z());
    for(int i = min.y()+1; i < max.y(); i++){
      es_potential[c] =  es_potential[c + zoffset  - xoffset]
                      + (es_potential[c - xoffset]
                      -  es_potential[c + zoffset2 - xoffset]
                      +  es_potential[c + zoffset]
                      -  es_potential[c + zoffset  - xoffset2])/2;
      c += yoffset;
    }
  }
  if(zplus_bd && xplus_bd){
    IntVector c(max.x()-1, min.y(), max.z()-1);
    for(int i = min.y()+1; i < max.y(); i++){
      es_potential[c] = es_potential[c - zoffset - xoffset]
                      + (es_potential[c - xoffset]
                      -  es_potential[c - zoffset2 - xoffset]
                      +  es_potential[c - zoffset]
                      -  es_potential[c - zoffset - xoffset2])/2;
      c += yoffset;
    }
  }
  if(zplus_bd && xminus_bd){
    IntVector c(min.x(), min.y(), max.z()-1);
    for(int i = min.y()+1; i < max.y(); i++){
      es_potential[c] = es_potential[c - zoffset + xoffset]
                      + (es_potential[c + xoffset]
                      -  es_potential[c - zoffset2 + xoffset]
                      -  es_potential[c - zoffset + xoffset2]
                      +  es_potential[c - zoffset])/2;
      c += yoffset;
    }
  }
}

void FVMBoundCond::setESPotentialBC(const Patch* patch, int dwi,
                                    CCVariable<double> &es_potential,
                                    constSFCXVariable<double> &fcx_conductivity,
                                    constSFCYVariable<double> &fcy_conductivity,
                                    constSFCZVariable<double> &fcz_conductivity)
{
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector dx = patch->dCell();
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  IntVector zoffset(0,0,1);

  IntVector min = patch->getExtraCellLowIndex();
  IntVector max = patch->getExtraCellHighIndex();
  int xmin = -99999;
  int xmax = -99999;
  int ymin = -99999;
  int ymax = -99999;
  int zmin = -99999;
  int zmax = -99999;

  if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
    xmin = min.x();
  }

  if(patch->getBCType(Patch::xplus) != Patch::Neighbor){
    xmax = max.x() - 1;
  }

  if(patch->getBCType(Patch::yminus) != Patch::Neighbor){
    ymin = min.y();
  }

  if(patch->getBCType(Patch::yplus) != Patch::Neighbor){
    ymax = max.y() - 1;
  }

  if(patch->getBCType(Patch::zminus) != Patch::Neighbor){
    zmin = min.z();
  }

  if(patch->getBCType(Patch::zplus) != Patch::Neighbor){
    zmax = max.z() - 1;
  }

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
        switch (face) {
          case Patch::xplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.y() == ymin || c.y() == ymax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = (bc_value/fcx_conductivity[c])*dx.x();
                  potential += es_potential[c - xoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::xminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[*bound_ptr] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.y() == ymin || c.y() == ymax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = (bc_value/fcx_conductivity[c + xoffset])*dx.x();
                  potential += es_potential[c + xoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.x() == xmin || c.x() == xmax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = (bc_value/fcy_conductivity[c])*dx.y();
                  potential += es_potential[c - yoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                es_potential[c] = bc_value;
                if(c.x() == xmin || c.x() == xmax || c.z() == zmin || c.z() == zmax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = (bc_value/fcy_conductivity[c + yoffset])*dx.y();
                  potential += es_potential[c + yoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.x() == xmin || c.x() == xmax || c.y() == ymin || c.y() == ymax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = (bc_value/fcz_conductivity[c])*dx.z();
                  potential += es_potential[c - zoffset];
                  es_potential[c] = potential;
                }
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              if(bc_kind == "Dirichlet"){
                es_potential[c] = bc_value;
              }else if(bc_kind == "Neumann"){
                if(c.x() == xmin || c.x() == xmax || c.y() == ymin || c.y() == ymax){
                  es_potential[c] = 0.0;
                }else{
                  double potential = (bc_value/fcz_conductivity[c + zoffset])*dx.z();
                  potential += es_potential[c + zoffset];
                  es_potential[c] = potential;
                };
              }
            }
            nCells += bound_ptr.size();
            break;
          default:
            break;
        }// end switch statement
      } // end foundIterator if statement
    } // end child loop
  } // end face loop
}

void FVMBoundCond::setG1BoundaryConditions(const Patch* patch, int dwi,
                                           CCVariable<Stencil7>& A,
                                           CCVariable<double>& rhs)
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
                rhs[c] -= bc_value * e;
              }else if(bc_kind == "Neumann"){
                A[c].e = 0;
                A[c].p += e;
                rhs[c] -= bc_value * a_e;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::xminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr + xoffset);
              if(bc_kind == "Dirichlet"){
                A[c].w = 0;
                rhs[c] -= bc_value * w;
              }else if(bc_kind == "Neumann"){
                A[c].w = 0;
                A[c].p += w;
                rhs[c] -= bc_value * a_w;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr - yoffset);
              if(bc_kind == "Dirichlet"){
                A[c].n = 0;
                rhs[c] -= bc_value * n;
              }else if(bc_kind == "Neumann"){
                A[c].n = 0;
                A[c].p += n;
                rhs[c] -= bc_value * a_n;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::yminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr + yoffset);
              if(bc_kind == "Dirichlet"){
                A[c].s = 0;
                rhs[c] -= bc_value * s;
              }else if(bc_kind == "Neumann"){
                A[c].s = 0;
                A[c].p += s;
                rhs[c] -= bc_value * a_s;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr - zoffset);
              if(bc_kind == "Dirichlet"){
                A[c].t = 0;
                rhs[c] -= bc_value * t;
              }else if(bc_kind == "Neumann"){
                A[c].t = 0;
                A[c].p += t;
                rhs[c] -= bc_value * a_t;
              }
            }
            nCells += bound_ptr.size();
            break;
          case Patch::zminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c(*bound_ptr + zoffset);
              if(bc_kind == "Dirichlet"){
                A[c].b = 0;
                rhs[c] -= bc_value * b;
              }else if(bc_kind == "Neumann"){
                A[c].b = 0;
                A[c].p += b;
                rhs[c] -= bc_value * a_b;
              }
            }
            nCells += bound_ptr.size();
            break;
          default:
            break;
        } // end switch statment
      } // end foundIterator if statment
    } // end child loop
  } // end face loop
}
