#include <Packages/Uintah/CCA/Components/MPM/BoundaryCond.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

using namespace Uintah;
namespace Uintah {

  
template<class T> void fillFace(NCVariable<T>& var,const Patch* patch, 
				 Patch::FaceType face, const T& value, 
				 IntVector offset)

{ 
  // cout <<"NCVariable.h: fillFace face "<<face<<endl;
  IntVector low,hi;
  low = patch->getInteriorNodeLowIndex();
  low+= offset;
  cout <<"low "<<low-offset<<"  Low + offset "<<low<<endl;
  
  hi = patch->getInteriorNodeHighIndex();      
  hi -= offset;
  cout <<"high "<<hi+offset<<"  hi - offset "<<hi<<endl;
  
  switch (face) {
  case Patch::xplus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(hi.x()-1,j,k)] = value;
	cout<<"fillFace xPlus "<<"patch "<<patch->getID()<<" "<<
	   IntVector(hi.x()-1,j,k)<<endl;
      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(low.x(),j,k)] = value;
	cout<<"fillFace xMinus "<<"patch "<<patch->getID()<<" "<<
	   IntVector(low.x(),j,k)<<endl;
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(i,hi.y()-1,k)] = value;
	cout<<"fillFace yplus "<<"patch "<<patch->getID()<<" "<<
	   IntVector(i,hi.y()-1,k)<<endl;
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(i,low.y(),k)] = value;
	cout<<"fillFace yminus "<<"patch "<<patch->getID()<<" "<<
	   IntVector(i,low.y(),k)<<endl;
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	var[IntVector(i,j,hi.z()-1)] = value;
	cout<<"fillFace zplus "<<"patch "<<patch->getID()<<" "<<
	   IntVector(i,j,hi.z()-1)<<endl;
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	var[IntVector(i,j,low.z())] = value;
	cout<<"fillFace zminus "<<"patch "<<patch->getID()<<" "<<
	   IntVector(i,j,low.z())<<endl;
      }
    }
    break;
  default:
    throw InternalError("Illegal FaceType in NCVariable::fillFace");
  }
  
}
  
  
  // Use to apply symmetry boundary conditions.  On the
  // indicated face, replace the component of the vector
  // normal to the face with 0.0
 
template<class T> void fillFaceNormal(NCVariable<T>& var,const Patch*, 
				      Patch::FaceType, 
				      IntVector)
{
  return;
} 
  
template<> void fillFaceNormal(NCVariable<Vector>& var, 
			       const Patch* patch,
			       Patch::FaceType face, 
			       IntVector offset)
{
  //cout <<"NCVariable.h: fillFaceNormal face "<<face<<endl;
  IntVector low,hi;
  low = patch->getInteriorNodeLowIndex();
  low+= offset;
  //cout <<"low "<<low-offset<<"  Low + offset "<<low<<endl;
                        
  hi = patch->getInteriorNodeHighIndex();      
  hi -= offset;
  //cout <<"high "<<hi+offset<<"  hi - offset "<<hi<<endl;

  switch (face) {
  case Patch::xplus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(hi.x()-1,j,k)] =
	  Vector(0.0,var[IntVector(hi.x()-1,j,k)].y(),
		 var[IntVector(hi.x()-1,j,k)].z());
	//cout<<"fillFaceFlux xPlus "<<"patch "<<patch->getID()<<" "<<
	//    IntVector(hi.x()-1,j,k)<<endl;
      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(low.x(),j,k)] = 
	  Vector(0.0,var[IntVector(low.x(),j,k)].y(),
		 var[IntVector(low.x(),j,k)].z());
	//cout<<"fillFaceFlux xMinus "<<"patch "<<patch->getID()<<" "<<
	//    IntVector(low.x(),j,k)<<endl;
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(i,hi.y()-1,k)] =
	  Vector(var[IntVector(i,hi.y()-1,k)].x(),0.0,
		 var[IntVector(i,hi.y()-1,k)].z());
	//cout<<"fillFaceFlux yplus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,hi.y()-1,k)<<endl;
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	var[IntVector(i,low.y(),k)] =
	  Vector(var[IntVector(i,low.y(),k)].x(),0.0,
		 var[IntVector(i,low.y(),k)].z());
	//cout<<"fillFaceFlux yminus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,low.y(),k)<<endl;
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	var[IntVector(i,j,hi.z()-1)] =
	  Vector(var[IntVector(i,j,hi.z()-1)].x(),
		 var[IntVector(i,j,hi.z()-1)].y(),0.0);
	//cout<<"fillFaceFlux zplus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,j,hi.z()-1)<<endl;
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	var[IntVector(i,j,low.z())] =
	  Vector(var[IntVector(i,j,low.z())].x(),
		 var[IntVector(i,j,low.z())].y(),0.0);
	//cout<<"fillFace zminus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,j,low.z())<<endl;
      }
    }
    break;
  default:
    throw InternalError("Illegal FaceType in NCVariable::fillFaceNormal");
  }
}

template void fillFace<double>(NCVariable<double>& var, const Patch* patch,
			       Patch::FaceType face, const double& value,
			       IntVector offset);

template void fillFace<Vector>(NCVariable<Vector>& var, const Patch* patch,
			       Patch::FaceType face, const Vector& value,
			       IntVector offset);

}  // using namespace Uintah

