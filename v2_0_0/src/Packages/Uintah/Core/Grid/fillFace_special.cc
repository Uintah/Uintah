#include <Packages/Uintah/Core/Grid/fillFace.h>

namespace Uintah {
  
template<> void fillFaceNormal<Vector>(NCVariable<Vector>& var, 
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
    SCI_THROW(InternalError("Illegal FaceType in NCVariable::fillFaceNormal"));
  }
}

template <> void fillFaceNormal<Vector>(CCVariable<SCIRun::Vector>& var, 
					const Patch*,
					Patch::FaceType face,
					IntVector offset)
{
  IntVector low,hi;
  low = var.getLowIndex() + offset;
  hi = var.getHighIndex() - offset;
  switch (face) {
  case Patch::xplus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
        var[IntVector(hi.x()-1,j,k)] =
          Vector(-var[IntVector(hi.x()-2,j,k)].x(),
                 var[IntVector(hi.x()-2,j,k)].y(),
                   var[IntVector(hi.x()-2,j,k)].z());
         //cout<<"fillFaceNORMAL Xplus "<<IntVector(hi.x()-1,j,k)<<endl;

      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
        var[IntVector(low.x(),j,k)] = 
          Vector(-var[IntVector(low.x()+1,j,k)].x(),
                 var[IntVector(low.x()+1,j,k)].y(),
                   var[IntVector(low.x()+1,j,k)].z());
         //cout<<"fillFaceNORMAL Xminus "<<IntVector(low.x(),j,k)<<endl;
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
        var[IntVector(i,hi.y()-1,k)] =
          Vector( var[IntVector(i,hi.y()-2,k)].x(),
                -var[IntVector(i,hi.y()-2,k)].y(),
                   var[IntVector(i,hi.y()-2,k)].z());
         //cout<<"fillFaceNORMAL Yplus "<<IntVector(i,hi.y()-1,k)<<endl;
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
        var[IntVector(i,low.y(),k)] =
          Vector( var[IntVector(i,low.y()+1,k)].x(),
                -var[IntVector(i,low.y()+1,k)].y(),
                   var[IntVector(i,low.y()+1,k)].z());
         //cout<<"fillFaceNORMAL Yminus "<<IntVector(i,low.y(),k)<<endl;
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
        var[IntVector(i,j,hi.z()-1)] =
          Vector( var[IntVector(i,j,hi.z()-2)].x(),
                   var[IntVector(i,j,hi.z()-2)].y(),
                -var[IntVector(i,j,hi.z()-2)].z());
         //cout<<"fillFaceNORMAL Zplus "<<IntVector(i,j,hi.z()-1)<<endl;
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
        var[IntVector(i,j,low.z())] =
          Vector( var[IntVector(i,j,low.z()+1)].x(),
                   var[IntVector(i,j,low.z()+1)].y(),
                -var[IntVector(i,j,low.z()+1)].z());
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
}
