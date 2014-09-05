#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/fillFace.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
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
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound,nbound,sfx,sfy,sfz;
	vector<IntVector>::const_iterator b;  // boundary cell iterator
	if (type == "Acceleration")
	  vel_bcs = patch->getArrayBCValues(face,dwi,"Velocity",bound,
					    nbound,sfx,sfy,sfz,child);
	else
	  vel_bcs  = patch->getArrayBCValues(face,dwi,type,bound,
					     nbound,sfx,sfy,sfz,child);
	if (type == "Velocity")
	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc =
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      for (b=nbound.begin();b!=nbound.end();b++){ 
		variable[*b] = bc->getValue();
              }
              if(n8or27==27){
	        for (b=nbound.begin();b!=nbound.end();b++){
                  variable[*b+oneCell] = bc->getValue();
                }
              }
	    }
	    delete vel_bcs;
	  }
	if (type == "Acceleration")
	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc =
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      for (b=nbound.begin();b != nbound.end();b++){
		variable[*b] = Vector(0,0,0);
	      }
              if(n8or27==27){
	        for (b=nbound.begin();b!=nbound.end();b++){
                  variable[*b+oneCell] = Vector(0.,0.,0.);
                }
              }
	    }
	    delete vel_bcs;
	  }
	if (type == "Symmetric")
	  if (vel_bcs != 0) {
	    if (face == Patch::xplus || face == Patch::xminus)
	      for (b=nbound.begin(); b != nbound.end();b++) {
		variable[*b] = Vector(0.,variable[*b].y(), variable[*b].z());
	      }
              if(n8or27==27){
	        for (b=nbound.begin(); b != nbound.end();b++){
                  variable[*b+oneCell] = Vector(0.,variable[*b+oneCell].y(),
                                                   variable[*b+oneCell].z());
                }
              }
	    if (face == Patch::yplus || face == Patch::yminus)
	      for (b=nbound.begin(); b != nbound.end();b++)
		variable[*b] = Vector(variable[*b].x(),0.,variable[*b].z());
              if(n8or27==27){
	        for (b=nbound.begin(); b != nbound.end();b++){
                  variable[*b+oneCell] = Vector(variable[*b+oneCell].x(),0.,
                                                variable[*b+oneCell].z());
                }
              }
	    if (face == Patch::zplus || face == Patch::zminus)
	      for (b=nbound.begin(); b != nbound.end();b++)
		variable[*b] = Vector(variable[*b].x(), variable[*b].y(),0.);
              if(n8or27==27){
	        for (b=nbound.begin(); b != nbound.end();b++){
                  variable[*b+oneCell] = Vector(variable[*b+oneCell].x(),
                                                variable[*b+oneCell].y(),0.);
                }
              }
	    delete vel_bcs;
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
    const BoundCondBase *temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound, nbound,sfx,sfy,sfz;
	vector<IntVector>::const_iterator b;
	temp_bcs = patch->getArrayBCValues(face,dwi,type,bound,nbound,
					   sfx,sfy,sfz,child);
    
	if (temp_bcs != 0) {
	  const TemperatureBoundCond* bc =
	    dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
	  if (bc->getKind() == "Dirichlet") {
	    for (b = nbound.begin(); b != nbound.end();b++)
	      variable[*b] = bc->getValue();
              if(n8or27==27){
               for (b = nbound.begin();b!=nbound.end();b++)
                 variable[*b+oneCell] = bc->getValue();
              }
	  }
	  delete temp_bcs;
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
	    for (boundary=nbound.begin(); boundary != nbound.end(); boundary++)
	      variable[*boundary]+= value*2.*gvolume[*boundary]/dx;
	  }
	  delete temp_bcs;
	}
      }
    } else
      continue;
  }
}
