#ifndef Packages_Uintah_CCA_Components_MPM_BoundaryCond_h
#define Packages_Uintah_CCA_Components_MPM_BoundaryCond_h
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <typeinfo>

namespace Uintah {
using namespace SCIRun;
using namespace Uintah;

  // CCVariables
  template<class T> void fillFace(CCVariable<T>& var,Patch::FaceType face, 
				  const T& value,
				  IntVector offset = IntVector(0,0,0))
  { 
    IntVector low,hi;
    low = var.getLowIndex() + offset;
    hi = var.getHighIndex() - offset;
    switch (face) {
    case Patch::xplus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(hi.x()-1,j,k)] = value;
	}
      }
      break;
    case Patch::xminus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(low.x(),j,k)] = value;
	}
      }
      break;
    case Patch::yplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(i,hi.y()-1,k)] = value;
	}
      }
      break;
    case Patch::yminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(i,low.y(),k)] = value;
	}
      }
      break;
    case Patch::zplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  var[IntVector(i,j,hi.z()-1)] = value;
	}
      }
      break;
    case Patch::zminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  var[IntVector(i,j,low.z())] = value;
	}
      }
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break;
    }
    
  }

  
  // Replace the values on the indicated face with value
  // using a 1st order difference formula for a Neumann BC condition
  // The plus_minus_one variable allows for negative interior BC, which is
  // simply the (-1)* interior value.
  
  template<class T> void fillFaceFlux(CCVariable<T>& var,Patch::FaceType face, 
				      const T& value,const Vector& dx,
				      const double& plus_minus_one=1.0,
				      IntVector offset = IntVector(0,0,0))
  { 
    IntVector low,hi;
    low = var.getLowIndex() + offset;
    hi = var.getHighIndex() - offset;
    
    switch (face) {
    case Patch::xplus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(hi.x()-1,j,k)] = 
	    (var[IntVector(hi.x()-2,j,k)])*plus_minus_one - 
	    value*dx.x();
	}
      }
      break;
    case Patch::xminus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(low.x(),j,k)] = 
	    (var[IntVector(low.x()+1,j,k)])*plus_minus_one - 
	    value * dx.x();
	}
      }
      break;
    case Patch::yplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(i,hi.y()-1,k)] = 
	    (var[IntVector(i,hi.y()-2,k)])*plus_minus_one - 
	    value * dx.y();
	}
      }
      break;
    case Patch::yminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(i,low.y(),k)] = 
	    (var[IntVector(i,low.y()+1,k)])*plus_minus_one - 
	    value * dx.y();
	}
      }
      break;
    case Patch::zplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  var[IntVector(i,j,hi.z()-1)] = 
	    (var[IntVector(i,j,hi.z()-2)])*plus_minus_one - 
	    value * dx.z();
	}
      }
      break;
    case Patch::zminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  var[IntVector(i,j,low.z())] =
	    (var[IntVector(i,j,low.z()+1)])*plus_minus_one - 
	    value * dx.z();
	}
      }
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break;
    }
  }

  // Only implemented for T==Vector, below
  template <class T> void fillFaceNormal(CCVariable<T>& var,const Patch* patch,
					 Patch::FaceType, 
					 IntVector);
  template <> void fillFaceNormal<Vector>(CCVariable<Vector>& var, 
					  const Patch* patch,
					  Patch::FaceType face,
					  IntVector offset);
  // NCVariables
  template<class T> void fillFace(NCVariable<T>& var,const Patch* patch, 
				  Patch::FaceType face, const T& value, 
				  IntVector offset = IntVector(0,0,0)) { 
    //    cout <<"NCVariable.h: fillFace face "<<face<<endl;
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
	  var[IntVector(hi.x()-1,j,k)] = value;
	  //  cout<<"fillFace xPlus "<<"patch "<<patch->getID()<<" "<<
	  //IntVector(hi.x()-1,j,k)<<endl;
	}
      }
      break;
    case Patch::xminus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(low.x(),j,k)] = value;
	  //cout<<"fillFace xMinus "<<"patch "<<patch->getID()<<" "<<
	  //	    IntVector(low.x(),j,k)<<endl;
	}
      }
      break;
    case Patch::yplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(i,hi.y()-1,k)] = value;
	  //cout<<"fillFace yplus "<<"patch "<<patch->getID()<<" "<<
	  //IntVector(i,hi.y()-1,k)<<endl;
	}
      }
      break;
    case Patch::yminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  var[IntVector(i,low.y(),k)] = value;
	  //cout<<"fillFace yminus "<<"patch "<<patch->getID()<<" "<<
	  // IntVector(i,low.y(),k)<<endl;
	}
      }
      break;
    case Patch::zplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  var[IntVector(i,j,hi.z()-1)] = value;
	  //cout<<"fillFace zplus "<<"patch "<<patch->getID()<<" "<<
	  //IntVector(i,j,hi.z()-1)<<endl;
	}
      }
      break;
    case Patch::zminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  var[IntVector(i,j,low.z())] = value;
	  //cout<<"fillFace zminus "<<"patch "<<patch->getID()<<" "<<
	  // IntVector(i,j,low.z())<<endl;
	}
      }
      break;
    default:
      SCI_THROW(InternalError("Illegal FaceType in NCVariable::fillFace"));
    }
  }
  

  // Use to apply symmetry boundary conditions.  On the
  // indicated face, replace the component of the vector
  // normal to the face with 0.0
  // Only implemented for T==Vector, below
  template<class T> void fillFaceNormal(NCVariable<T>& var, const Patch* patch, 
					Patch::FaceType face, 
					IntVector offset = IntVector(0,0,0));

  template<> void fillFaceNormal<Vector>(NCVariable<Vector>& var, const Patch* patch, 
					 Patch::FaceType face, 
					 IntVector offset);

  // Any variables
  template <class V, class T> void fillFace(V& var, const Patch* patch, 
					    Patch::FaceType face,
					    const T& value, 
					 IntVector offset = IntVector(0,0,0))
  {
    //__________________________________
    // Add (1,0,0) to low index when no 
    // neighbor patches are present
    IntVector low,hi,adjust; 
    int numGC = 0;
    low = patch->getCellLowIndex();
    
    if (typeid(V) == typeid(SFCXVariable<double>))
      adjust=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:1,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:0);
    
    if (typeid(V) == typeid(SFCYVariable<double>))
      adjust=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:1,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:0);
    
    if (typeid(V) == typeid(SFCZVariable<double>))
      adjust=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:1);
    
    
    low+= adjust;
    low-= offset;
    hi  = patch->getCellHighIndex();
    hi += IntVector(patch->getBCType(Patch::xplus) ==Patch::Neighbor?numGC:0,
		    patch->getBCType(Patch::yplus) ==Patch::Neighbor?numGC:0,
		    patch->getBCType(Patch::zplus) ==Patch::Neighbor?numGC:0);
    hi += offset;
    // cout<< "fillFace: SFCXVariable.h"<<endl;
    // cout<< "low: "<<low<<endl;
    // cout<< "hi:  "<<hi <<endl;
    
    switch (face) {
    case Patch::xplus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  // cout << "Using iterator = " << IntVector(hi.x()-1,j,k) << endl;
	  var[IntVector(hi.x()-1,j,k)] = value;
	}
      }
      break;
    case Patch::xminus:
      for (int j = low.y(); j<hi.y(); j++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  // cout << "Using iterator = " << IntVector(low.x(),j,k) << endl;
	  var[IntVector(low.x(),j,k)] = value;
	}
      }
      break;
    case Patch::yplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  // cout << "Using iterator = " << IntVector(i,hi.y()-1,k) << endl;
	  var[IntVector(i,hi.y()-1,k)] = value;
	}
      }
      break;
    case Patch::yminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int k = low.z(); k<hi.z(); k++) {
	  // cout << "Using iterator = " << IntVector(i,low.y(),k) << endl;
	  var[IntVector(i,low.y(),k)] = value;
	}
      }
      break;
    case Patch::zplus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  // cout << "Using iterator = " << IntVector(i,j,hi.z()-1) << endl;
	  var[IntVector(i,j,hi.z()-1)] = value;
	}
      }
      break;
    case Patch::zminus:
      for (int i = low.x(); i<hi.x(); i++) {
	for (int j = low.y(); j<hi.y(); j++) {
	  // cout << "Using iterator = " << IntVector(i,j,low.z()) << endl;
	  var[IntVector(i,j,low.z())] = value;
	}
      }
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break;
    }
    
  }
  
} // End namespace Uintah
#endif
