#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>
#include <Core/Geometry/IntVector.h>
#include <vector>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using std::vector;
using std::cout;
using std::endl;

#define JOHN
//#undef JOHN

MPMBoundCond::MPMBoundCond()
{
}

MPMBoundCond::~MPMBoundCond()
{
}

#ifndef JOHN
void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
					const string& type, 
					NCVariable<Vector>& variable)
{
  IntVector offset =  IntVector(0,0,0);
  
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *vel_bcs, *sym_bcs;
    if (patch->getBCType(face) == Patch::None) {
      if (type == "Acceleration")
	vel_bcs = patch->getBCValues(dwi,"Velocity",face);
      else
	vel_bcs  = patch->getBCValues(dwi,type,face);
      sym_bcs  = patch->getBCValues(dwi,type,face);
    } else
      continue;
    if (type == "Velocity")
      if (vel_bcs != 0) {
	const VelocityBoundCond* bc =
	  dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	if (bc->getKind() == "Dirichlet") {
	  //cout << "Velocity bc value = " << bc->getValue() << endl;
	  fillFace(variable,patch, face,bc->getValue(),offset);
	}
      }
    if (type == "Acceleration")
      if (vel_bcs != 0) {
	const VelocityBoundCond* bc =
	  dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	if (bc->getKind() == "Dirichlet") {
	  fillFace(variable,patch, face,Vector(0.,0.,0.),offset);
	}
      }
    if (type == "Symmetric")
      if (sym_bcs != 0) {
	fillFaceNormal(variable,patch, face,offset);
      }
    delete vel_bcs;
    delete sym_bcs;
  }
}

#else
void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
					const string& type, 
					NCVariable<Vector>& variable)
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *vel_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound,inter,nbound;
	vector<IntVector>::const_iterator boundary;
	if (type == "Acceleration")
	  vel_bcs = patch->getArrayBCValues(face,dwi,"Velocity",bound,inter,
					    nbound,child);
	else
	  vel_bcs  = patch->getArrayBCValues(face,dwi,type,bound,inter,
					     nbound,child);
	if (type == "Velocity")
	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc =
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      for (boundary=nbound.begin(); boundary != nbound.end();
		   boundary++) 
		variable[*boundary] = bc->getValue();
	    }
	    delete vel_bcs;
	  }
	if (type == "Acceleration")
	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc =
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      for (boundary=nbound.begin(); boundary != nbound.end();
		   boundary++) {
		variable[*boundary] = Vector(0,0,0);
	      }
	    }
	    delete vel_bcs;
	  }
	if (type == "Symmetric")
	  if (vel_bcs != 0) {
	    if (face == Patch::xplus || face == Patch::xminus)
	      for (boundary=nbound.begin(); boundary != nbound.end(); 
		   boundary++)
		variable[*boundary] = Vector(0.,variable[*boundary].y(),
					      variable[*boundary].z());
	    if (face == Patch::yplus || face == Patch::yminus)
	      for (boundary=nbound.begin(); boundary != nbound.end(); 
		   boundary++)
		variable[*boundary] = Vector(variable[*boundary].x(),0.,
					      variable[*boundary].z());
	    if (face == Patch::zplus || face == Patch::zminus)
	      for (boundary=nbound.begin(); boundary != nbound.end(); 
		   boundary++)
		variable[*boundary] = Vector(variable[*boundary].x(),
					      variable[*boundary].y(),0.);
	    delete vel_bcs;
	  }
      }
    } else
      continue;
  }
}
#endif

#ifndef JOHN
void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
					const string& type, 
					NCVariable<double>& variable)

{
  IntVector offset =  IntVector(0,0,0);
  
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      temp_bcs = patch->getBCValues(dwi,type,face);
    } else
      continue;
    
    if (temp_bcs != 0) {
      const TemperatureBoundCond* bc =
	dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
      if (bc->getKind() == "Dirichlet") {
	fillFace(variable,patch, face,bc->getValue(),offset);
      }
      delete temp_bcs;
    }
  }
}

#else
void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
					const string& type, 
					NCVariable<double>& variable)

{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(dwi);
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound, inter, nbound;
	vector<IntVector>::const_iterator boundary;
	temp_bcs = patch->getArrayBCValues(face,dwi,type,bound,inter,nbound,
					   child);
    
	if (temp_bcs != 0) {
	  const TemperatureBoundCond* bc =
	    dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
	  if (bc->getKind() == "Dirichlet") {
	    for (boundary = nbound.begin(); boundary != nbound.end();
		 boundary++)
	      variable[*boundary] = bc->getValue();
	  }
	  delete temp_bcs;
	}
      }
    } else
      continue;
  }
}
#endif

#ifndef JOHN
void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
					const string& type, 
					NCVariable<double>& variable,
					constNCVariable<double>& gvolume)
{
  IntVector offset =  IntVector(0,0,0);
  
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      temp_bcs = patch->getBCValues(dwi,type,face);
    } else
      continue;
    
    if (temp_bcs != 0) {
      const TemperatureBoundCond* bc =
	dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
      if (bc->getKind() == "Neumann") {
	
	double value = bc->getValue();
	
	IntVector low = patch->getInteriorNodeLowIndex();
	IntVector hi  = patch->getInteriorNodeHighIndex();     
	if(face==Patch::xplus || face==Patch::xminus){
	  int I = 0;
	  if(face==Patch::xminus){ I=low.x(); }
	  if(face==Patch::xplus){ I=hi.x()-1; }
	  for (int j = low.y(); j<hi.y(); j++) {
	    for (int k = low.z(); k<hi.z(); k++) {
	      variable[IntVector(I,j,k)] +=
		value*(2.0*gvolume[IntVector(I,j,k)]/dx.x());
	    }
	  }
	}
	if(face==Patch::yplus || face==Patch::yminus){
	  int J = 0;
	  if(face==Patch::yminus){ J=low.y(); }
	  if(face==Patch::yplus){ J=hi.y()-1; }
	  for (int i = low.x(); i<hi.x(); i++) {
	    for (int k = low.z(); k<hi.z(); k++) {
	      variable[IntVector(i,J,k)] +=
		value*(2.0*gvolume[IntVector(i,J,k)]/dx.y());
	    }
	  }
	}
	if(face==Patch::zplus || face==Patch::zminus){
	  int K = 0;
	  if(face==Patch::zminus){ K=low.z(); }
	  if(face==Patch::zplus){ K=hi.z()-1; }
	  for (int i = low.x(); i<hi.x(); i++) {
	    for (int j = low.y(); j<hi.y(); j++) {
	      variable[IntVector(i,j,K)] +=
		value*(2.0*gvolume[IntVector(i,j,K)]/dx.z());
	    }
	  }
	}
      }
    }
    delete temp_bcs;
  }
}

#else
void MPMBoundCond::setBoundaryCondition(const Patch* patch,int dwi,
					const string& type, 
					NCVariable<double>& variable,
					constNCVariable<double>& gvolume)
{
  Vector deltax = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase* temp_bcs;
    if (patch->getBCType(face) == Patch::None) {
      int numChildren = 
	patch->getBCDataArray(face)->getNumberChildren(dwi);
      for (int child = 0; child < numChildren; child++) {
	vector<IntVector> bound,inter,nbound;
	vector<IntVector>::const_iterator boundary;
	temp_bcs  = patch->getArrayBCValues(face,dwi,"Temperature",bound,
					    inter,nbound,child);
	
	double dx;
	if (face == Patch::xplus || face == Patch::xminus) dx = deltax.x();
	if (face == Patch::yplus || face == Patch::yminus) dx = deltax.y();
	if (face == Patch::zplus || face == Patch::zminus) dx = deltax.z();
	
	if (temp_bcs != 0) {
	  const TemperatureBoundCond* bc =
	    dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
	  if (bc->getKind() == "Neumann"){
	    double value = bc->getValue();
	    for (boundary=nbound.begin(); boundary != nbound.end();
		 boundary++)
	      variable[*boundary]+= value*2.*gvolume[*boundary]/dx;
	  }
	  delete temp_bcs;
	}
      }
    } else
      continue;
  }
}
#endif
