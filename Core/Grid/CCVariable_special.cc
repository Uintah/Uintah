#include <Packages/Uintah/Core/Grid/CCVariable.h>


namespace Uintah {

 // Use to apply symmetry boundary conditions. 
 // Tangential components: zero gradient
 // Normal components:   -(interior value)
 // This only takes care of the symmetric faces
 // Need to set the rest of the faces with fillFaceFlux
template <>
void
CCVariable<Vector>::fillFaceNormal(Patch::FaceType face,
				   IntVector offset)
{
  IntVector low,hi;
  low = getLowIndex() + offset;
  hi = getHighIndex() - offset;
  switch (face) {
  case Patch::xplus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
        (*this)[IntVector(hi.x()-1,j,k)] =
          Vector(-(*this)[IntVector(hi.x()-2,j,k)].x(),
                 (*this)[IntVector(hi.x()-2,j,k)].y(),
                   (*this)[IntVector(hi.x()-2,j,k)].z());
         //cout<<"fillFaceNORMAL Xplus "<<IntVector(hi.x()-1,j,k)<<endl;

      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
        (*this)[IntVector(low.x(),j,k)] = 
          Vector(-(*this)[IntVector(low.x()+1,j,k)].x(),
                 (*this)[IntVector(low.x()+1,j,k)].y(),
                   (*this)[IntVector(low.x()+1,j,k)].z());
         //cout<<"fillFaceNORMAL Xminus "<<IntVector(low.x(),j,k)<<endl;
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
        (*this)[IntVector(i,hi.y()-1,k)] =
          Vector( (*this)[IntVector(i,hi.y()-2,k)].x(),
                -(*this)[IntVector(i,hi.y()-2,k)].y(),
                   (*this)[IntVector(i,hi.y()-2,k)].z());
         //cout<<"fillFaceNORMAL Yplus "<<IntVector(i,hi.y()-1,k)<<endl;
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
        (*this)[IntVector(i,low.y(),k)] =
          Vector( (*this)[IntVector(i,low.y()+1,k)].x(),
                -(*this)[IntVector(i,low.y()+1,k)].y(),
                   (*this)[IntVector(i,low.y()+1,k)].z());
         //cout<<"fillFaceNORMAL Yminus "<<IntVector(i,low.y(),k)<<endl;
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
        (*this)[IntVector(i,j,hi.z()-1)] =
          Vector( (*this)[IntVector(i,j,hi.z()-2)].x(),
                   (*this)[IntVector(i,j,hi.z()-2)].y(),
                -(*this)[IntVector(i,j,hi.z()-2)].z());
         //cout<<"fillFaceNORMAL Zplus "<<IntVector(i,j,hi.z()-1)<<endl;
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
        (*this)[IntVector(i,j,low.z())] =
          Vector( (*this)[IntVector(i,j,low.z()+1)].x(),
                   (*this)[IntVector(i,j,low.z()+1)].y(),
                -(*this)[IntVector(i,j,low.z()+1)].z());
         //cout<<"fillFaceNORMAL Zminus "<<IntVector(i,j,low.z())<<endl;
      }
    }
    break;
  case Patch::numFaces:
    break;
  case Patch::invalidFace:
    break;
  }
}

} // end namespace Uintah
