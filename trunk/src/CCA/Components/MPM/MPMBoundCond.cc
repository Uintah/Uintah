#include <CCA/Components/MPM/MPMBoundCond.h>
#include <Core/Grid/BoundaryConditions/VelocityBoundCond.h>
#include <Core/Grid/BoundaryConditions/SymmetryBoundCond.h>
#include <Core/Grid/BoundaryConditions/TemperatureBoundCond.h>
#include <Core/Grid/BoundaryConditions/PressureBoundCond.h>
#include <Core/Grid/BoundaryConditions/fillFace.h>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <vector>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using std::vector;
using std::cout;
using std::endl;

MPMBoundCond::MPMBoundCond()
{
}

MPMBoundCond::~MPMBoundCond()
{
}

void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
                                        const string& type, 
                                        NCVariable<Vector>& variable,
                                        string interp_type)
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    IntVector oneCell = patch->faceDirection(face);
    const BoundCondBase *vel_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      IntVector l(0,0,0),h(0,0,0),off(0,0,0);
      if(interp_type=="gimp" || interp_type=="3rdorderBS"){
        patch->getFaceExtraNodes(face,0,l,h);
      }
      for (int child = 0; child < numChildren; child++) {
        vector<IntVector> *nbound_ptr;
       vector<IntVector> *nu;        // not used;
        vector<IntVector>::const_iterator b;  // boundary cell iterator
        if (type == "Acceleration"){
          vel_bcs = patch->getArrayBCValues(face,dwi,"Velocity",nu,
                                            nbound_ptr,nu,nu,nu,child);
        }
        else{
          vel_bcs  = patch->getArrayBCValues(face,dwi,type,nu,
                                             nbound_ptr,nu,nu,nu,child);
        }

        if (type == "Velocity"){
          if (vel_bcs != 0) {
            const VelocityBoundCond* bc =
              dynamic_cast<const VelocityBoundCond*>(vel_bcs);
            if (bc->getKind() == "Dirichlet") {
              Vector bcv = bc->getValue();
              for (b=nbound_ptr->begin();b!=nbound_ptr->end();b++){ 
                IntVector nd = *b;
                variable[nd] = bcv;
              }
              if(interp_type=="gimp" || interp_type=="3rdorderBS"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = bcv;
                }
              }
            }
            delete vel_bcs;
          }
        }
        if (type == "Acceleration"){
          if (vel_bcs != 0) {
            const VelocityBoundCond* bc =
              dynamic_cast<const VelocityBoundCond*>(vel_bcs);
            if (bc->getKind() == "Dirichlet") {
              for (b=nbound_ptr->begin();b != nbound_ptr->end();b++){
                IntVector nd = *b;
                variable[nd] = Vector(0,0,0);
              }
              if(interp_type=="gimp" || interp_type=="3rdorderBS"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(0,0,0);
                }
              }
            }
            delete vel_bcs;
          }
        }
        if (type == "Symmetric"){
          if (vel_bcs != 0) {
            if (face == Patch::xplus || face == Patch::xminus){
              for (b=nbound_ptr->begin(); b != nbound_ptr->end();b++) {
                IntVector nd = *b;
                variable[nd] = Vector(0.,variable[nd].y(), variable[nd].z());
              }
              if(interp_type=="gimp"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(0.,variable[nd].y(), variable[nd].z());
                }
              }
              if(interp_type=="3rdorderBS"){
                if (face == Patch::xplus){
                   off=IntVector(-2,0,0);
                }
                else{
                   off=IntVector(2,0,0);
                }
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  IntVector od = nd + off;
                  variable[nd] = Vector(-variable[od].x(),
                                         variable[od].y(), variable[od].z());
                }
              }
            }
            if (face == Patch::yplus || face == Patch::yminus){
              for (b=nbound_ptr->begin(); b != nbound_ptr->end();b++){
                IntVector nd = *b;
                variable[nd] = Vector(variable[nd].x(),0.,variable[nd].z());
              }
              if(interp_type=="gimp"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(variable[nd].x(),0.,variable[nd].z());
                }
              }
              if(interp_type=="3rdorderBS"){
                if (face == Patch::yplus){
                   off=IntVector(0,-2,0);
                }
                else{
                   off=IntVector(0,2,0);
                }
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  IntVector od = nd + off;
                  variable[nd] = Vector(variable[od].x(),
                                       -variable[od].y(), variable[od].z());
                }
              }
            }
            if (face == Patch::zplus || face == Patch::zminus){
              for (b=nbound_ptr->begin(); b != nbound_ptr->end();b++){
                IntVector nd = *b;
                variable[nd] = Vector(variable[nd].x(), variable[nd].y(),0.);
              }
              if(interp_type=="gimp"){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(variable[nd].x(), variable[nd].y(),0.);
                }
              }
              if(interp_type=="3rdorderBS"){
                if (face == Patch::zplus){
                   off=IntVector(0,0,-2);
                }
                else{
                   off=IntVector(0,0,2);
                }
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  IntVector od = nd + off;
                  variable[nd] = Vector(variable[od].x(),
                                        variable[od].y(), -variable[od].z());
                }
              }
            }
            delete vel_bcs;
          }
        }
      }
    } else
      continue;
  }
}

void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
                                        const string& type, 
                                        NCVariable<double>& variable,
                                        string interp_type)

{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    IntVector oneCell = patch->faceDirection(face);
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      IntVector l(0,0,0),h(0,0,0);
      if(interp_type=="gimp" || interp_type=="3rdorderBS"){
        patch->getFaceExtraNodes(face,0,l,h);
      }
      for (int child = 0; child < numChildren; child++) {
       vector<IntVector> *nbound_ptr;
       vector<IntVector> *nu;  // not used
       vector<IntVector>::const_iterator b;
       if(type=="Temperature"){
        const BoundCondBase *temp_bcs = patch->getArrayBCValues(face,dwi,
                                                       type,nu,nbound_ptr,
                                                      nu,nu,nu,child);
        if (temp_bcs != 0){
          const TemperatureBoundCond* bc =
            dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
          if (bc->getKind() == "Dirichlet") {
            double bcv = bc->getValue();
            for (b = nbound_ptr->begin(); b != nbound_ptr->end();b++){
              IntVector nd = *b;
              variable[nd] = bcv;
            }
            if(interp_type=="gimp" || interp_type=="3rdorderBS"){
              for(NodeIterator it(l,h); !it.done(); it++) {
                IntVector nd = *it;
                variable[nd] = bcv;
              }
            }
          }
          delete temp_bcs;
        }
       }

       if(type=="Pressure"){
        const BoundCondBase *press_bcs = patch->getArrayBCValues(face,dwi,
                                                        type,nu,nbound_ptr,
                                                        nu,nu,nu,child);
        if (press_bcs != 0) {
          const PressureBoundCond* bc =
            dynamic_cast<const PressureBoundCond*>(press_bcs);
          if (bc->getKind() == "Dirichlet") {
            double bcv = bc->getValue();
            for (b = nbound_ptr->begin(); b != nbound_ptr->end();b++){
              IntVector nd = *b;
              variable[nd] = bcv;
            }
            if(interp_type=="gimp" || interp_type=="3rdorderBS"){
              for(NodeIterator it(l,h); !it.done(); it++) {
                IntVector nd = *it;
                variable[nd] = bcv;
              }
            }
          }

          if (bc->getKind() == "Neumann" && (interp_type=="gimp" 
                                         ||  interp_type=="3rdorderBS")) {
            Vector deltax = patch->dCell();
            double dx = -9;
            IntVector off(-9,-9,-9);
            if (face == Patch::xplus){
              dx = deltax.x();
              off=IntVector(1,0,0);
            }
            else if (face == Patch::xminus){
              dx = deltax.x();
              off=IntVector(-1,0,0);
            }
            else if (face == Patch::yplus){
              dx = deltax.y();
              off=IntVector(0,1,0);
            }
            else if (face == Patch::yminus){
              dx = deltax.y();
              off=IntVector(0,-1,0);
            }
            else if (face == Patch::zplus){
              dx = deltax.z();
              off=IntVector(0,0,1);
            }
            else if (face == Patch::zminus){
              dx = deltax.z();
              off=IntVector(0,0,-1);
            }

            double gradv = bc->getValue();
            for(NodeIterator it(l,h); !it.done(); it++) {
              IntVector nd = *it;
              variable[nd] = variable[nd-off] + gradv*dx;
//            if(face==Patch::xminus){
//              cout << "node = " << nd << " variable = " << variable[nd]
//                   << " variable-off = " << variable[nd-off] << endl;
//              }
            }
          }

          delete press_bcs;
        }
       }

      }
    } else
      continue;
  }
}

void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
                                        const string& type, 
                                        NCVariable<double>& variable,
                                        constNCVariable<double>& gvolume,
                                        string interp_type)
{
  Vector deltax = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase* temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = 
        patch->getBCDataArray(face)->getNumberChildren(dwi);
      for (int child = 0; child < numChildren; child++) {
        vector<IntVector> *nbound_ptr;
       vector<IntVector> *nu;    // not used
        vector<IntVector>::const_iterator boundary;
        temp_bcs  = patch->getArrayBCValues(face,dwi,"Temperature",nu,
                                            nbound_ptr,nu,nu,nu,child);
        
        double dx = -9;
        if (face == Patch::xplus || face == Patch::xminus) dx = deltax.x();
        if (face == Patch::yplus || face == Patch::yminus) dx = deltax.y();
        if (face == Patch::zplus || face == Patch::zminus) dx = deltax.z();
        
        if (temp_bcs != 0) {
          const TemperatureBoundCond* bc =
            dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
          if (bc->getKind() == "Neumann"){
            double value = bc->getValue();
            for (boundary=nbound_ptr->begin(); boundary != nbound_ptr->end(); boundary++){
              IntVector nd = *boundary;
              variable[nd] += value*2.*gvolume[nd]/dx;
            }
          }
          delete temp_bcs;
        }
      }
    } else
      continue;
  }
}
