#include <Packages/Uintah/Core/Grid/NCVariable.h>

namespace Uintah {

// Use to apply symmetry boundary conditions.  On the
// indicated face, replace the component of the vector
// normal to the face with 0.0
template<>
void
NCVariable<Vector>::fillFaceNormal(const Patch* patch,
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
	(*this)[IntVector(hi.x()-1,j,k)] =
	  Vector(0.0,(*this)[IntVector(hi.x()-1,j,k)].y(),
		 (*this)[IntVector(hi.x()-1,j,k)].z());
	//cout<<"fillFaceFlux xPlus "<<"patch "<<patch->getID()<<" "<<
	//    IntVector(hi.x()-1,j,k)<<endl;
      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(low.x(),j,k)] = 
	  Vector(0.0,(*this)[IntVector(low.x(),j,k)].y(),
		 (*this)[IntVector(low.x(),j,k)].z());
	//cout<<"fillFaceFlux xMinus "<<"patch "<<patch->getID()<<" "<<
	//    IntVector(low.x(),j,k)<<endl;
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(i,hi.y()-1,k)] =
	  Vector((*this)[IntVector(i,hi.y()-1,k)].x(),0.0,
		 (*this)[IntVector(i,hi.y()-1,k)].z());
	//cout<<"fillFaceFlux yplus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,hi.y()-1,k)<<endl;
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(i,low.y(),k)] =
	  Vector((*this)[IntVector(i,low.y(),k)].x(),0.0,
		 (*this)[IntVector(i,low.y(),k)].z());
	//cout<<"fillFaceFlux yminus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,low.y(),k)<<endl;
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	(*this)[IntVector(i,j,hi.z()-1)] =
	  Vector((*this)[IntVector(i,j,hi.z()-1)].x(),
		 (*this)[IntVector(i,j,hi.z()-1)].y(),0.0);
	//cout<<"fillFaceFlux zplus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,j,hi.z()-1)<<endl;
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	(*this)[IntVector(i,j,low.z())] =
	  Vector((*this)[IntVector(i,j,low.z())].x(),
		 (*this)[IntVector(i,j,low.z())].y(),0.0);
	//cout<<"fillFace zminus "<<"patch "<<patch->getID()<<" "<<
	//     IntVector(i,j,low.z())<<endl;
      }
    }
    break;
  default:
    throw InternalError("Illegal FaceType in NCVariable::fillFaceNormal");
  }
}


} // end namespace Uintah
