#include <Packages/Uintah/Core/Grid/CCVariable.h>


namespace Uintah {

// Use to apply symmetry boundary conditions.  On the
// indicated face, replace the component of the vector
// normal to the face with 0.0
template <>
void
CCVariable<Vector>::fillFaceNormal(Patch::FaceType face,
				   IntVector offset)
{
#if 0
  IntVector low,hi;
  low = getLowIndex() + offset;
  hi = getHighIndex() - offset;
  switch (face) {
  case Patch::xplus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(hi.x()-1,j,k)] =
	  Vector(0.0,(*this)[IntVector(hi.x()-1,j,k)].y(),
		 (*this)[IntVector(hi.x()-1,j,k)].z());
      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(low.x(),j,k)] = 
	  Vector(0.0,(*this)[IntVector(low.x(),j,k)].y(),
		 (*this)[IntVector(low.x(),j,k)].z());
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(i,hi.y()-1,k)] =
	  Vector((*this)[IntVector(i,hi.y()-1,k)].x(),0.0,
		 (*this)[IntVector(i,hi.y()-1,k)].z());
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	(*this)[IntVector(i,low.y(),k)] =
	  Vector((*this)[IntVector(i,low.y(),k)].x(),0.0,
		 (*this)[IntVector(i,low.y(),k)].z());
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	(*this)[IntVector(i,j,hi.z()-1)] =
	  Vector((*this)[IntVector(i,j,hi.z()-1)].x(),
		 (*this)[IntVector(i,j,hi.z()-1)].y(),0.0);
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	(*this)[IntVector(i,j,low.z())] =
	  Vector((*this)[IntVector(i,j,low.z())].x(),
		 (*this)[IntVector(i,j,low.z())].y(),0.0);
      }
    }
    break;
  case Patch::numFaces:
    break;
  case Patch::invalidFace:
    break;
  }
#endif
}

} // end namespace Uintah
