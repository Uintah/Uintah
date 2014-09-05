#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/PressureBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/fillFace.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
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
					NCVariable<Vector>& variable,int n8or27)
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    IntVector oneCell = patch->faceDirection(face);
    const BoundCondBase *vel_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      IntVector l(0,0,0),h(0,0,0);
      if(n8or27==27){
        patch->getFaceExtraNodes(face,0,l,h);
      }
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound,nbound,sfx,sfy,sfz;
	vector<IntVector>::const_iterator b;  // boundary cell iterator
	if (type == "Acceleration"){
	  vel_bcs = patch->getArrayBCValues(face,dwi,"Velocity",bound,
					    nbound,sfx,sfy,sfz,child);
        }
	else{
	  vel_bcs  = patch->getArrayBCValues(face,dwi,type,bound,
					     nbound,sfx,sfy,sfz,child);
        }

	if (type == "Velocity"){
	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc =
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
              Vector bcv = bc->getValue();
	      for (b=nbound.begin();b!=nbound.end();b++){ 
                IntVector nd = *b;
		variable[nd] = bcv;
              }
              if(n8or27==27){
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
	      for (b=nbound.begin();b != nbound.end();b++){
                IntVector nd = *b;
		variable[nd] = Vector(0,0,0);
	      }
              if(n8or27==27){
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
	      for (b=nbound.begin(); b != nbound.end();b++) {
                IntVector nd = *b;
		variable[nd] = Vector(0.,variable[nd].y(), variable[nd].z());
	      }
              if(n8or27==27){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(0.,variable[nd].y(), variable[nd].z());
                }
              }
            }
	    if (face == Patch::yplus || face == Patch::yminus){
	      for (b=nbound.begin(); b != nbound.end();b++){
                IntVector nd = *b;
		variable[nd] = Vector(variable[nd].x(),0.,variable[nd].z());
              }
              if(n8or27==27){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(variable[nd].x(),0.,variable[nd].z());
                }
              }
            }
	    if (face == Patch::zplus || face == Patch::zminus){
	      for (b=nbound.begin(); b != nbound.end();b++){
                IntVector nd = *b;
		variable[nd] = Vector(variable[nd].x(), variable[nd].y(),0.);
              }
              if(n8or27==27){
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector nd = *it;
                  variable[nd] = Vector(variable[nd].x(), variable[nd].y(),0.);
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
                                        int n8or27)

{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    IntVector oneCell = patch->faceDirection(face);
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      IntVector l(0,0,0),h(0,0,0);
      if(n8or27==27){
        patch->getFaceExtraNodes(face,0,l,h);
      }
      for (int child = 0; child < numChildren; child++) {
       vector<IntVector> bound, nbound,sfx,sfy,sfz;
       vector<IntVector>::const_iterator b;
       if(type=="Temperature"){
	const BoundCondBase *temp_bcs = patch->getArrayBCValues(face,dwi,
	                                               type,bound,nbound,
                                                       sfx,sfy,sfz,child);
	if (temp_bcs != 0){
	  const TemperatureBoundCond* bc =
	    dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
	  if (bc->getKind() == "Dirichlet") {
            double bcv = bc->getValue();
	    for (b = nbound.begin(); b != nbound.end();b++){
              IntVector nd = *b;
	      variable[nd] = bcv;
            }
            if(n8or27==27){
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
	                                                type,bound,nbound,
                                                        sfx,sfy,sfz,child);
	if (press_bcs != 0) {
	  const PressureBoundCond* bc =
	    dynamic_cast<const PressureBoundCond*>(press_bcs);
	  if (bc->getKind() == "Dirichlet") {
            double bcv = bc->getValue();
	    for (b = nbound.begin(); b != nbound.end();b++){
              IntVector nd = *b;
	      variable[nd] = bcv;
            }
            if(n8or27==27){
              for(NodeIterator it(l,h); !it.done(); it++) {
                IntVector nd = *it;
	        variable[nd] = bcv;
              }
            }
          }

	  if (bc->getKind() == "Neumann" && n8or27==27) {
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
//	      if(face==Patch::xminus){
//	        cout << "node = " << nd << " variable = " << variable[nd]
//		     << " variable-off = " << variable[nd-off] << endl;
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
                                        int n8or27)
{
  Vector deltax = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase* temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = 
	patch->getBCDataArray(face)->getNumberChildren(dwi);
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound,nbound,sfx,sfy,sfz;
	vector<IntVector>::const_iterator boundary;
	temp_bcs  = patch->getArrayBCValues(face,dwi,"Temperature",bound,
					    nbound,sfx,sfy,sfz,child);
	
	double dx = -9;
	if (face == Patch::xplus || face == Patch::xminus) dx = deltax.x();
	if (face == Patch::yplus || face == Patch::yminus) dx = deltax.y();
	if (face == Patch::zplus || face == Patch::zminus) dx = deltax.z();
	
	if (temp_bcs != 0) {
	  const TemperatureBoundCond* bc =
	    dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
	  if (bc->getKind() == "Neumann"){
	    double value = bc->getValue();
	    for (boundary=nbound.begin(); boundary != nbound.end(); boundary++){
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
