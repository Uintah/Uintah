#ifndef Packages_Uintah_CCA_Components_MPM_BoundaryCond_h
#define Packages_Uintah_CCA_Components_MPM_BoundaryCond_h
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {
using namespace SCIRun;
using namespace Uintah;

 template<class T> void fillFace(NCVariable<T>& var,const Patch* patch, 
				 Patch::FaceType face, const T& value, 
				 IntVector offset = IntVector(0,0,0));

 template<class T> void fillFaceNormal(NCVariable<T>& var, const Patch* patch, 
				       Patch::FaceType face, 
				       IntVector offset = IntVector(0,0,0));

 template<> void fillFaceNormal(NCVariable<Vector>& var, const Patch* patch,
				Patch::FaceType face, 
				IntVector offset);

} // End namespace Uintah
#endif
