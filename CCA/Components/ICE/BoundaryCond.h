#ifndef Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#define Packages_Uintah_CCA_Components_Ice_BoundaryCond_h

#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>

namespace Uintah {
  class DataWarehouse;
  void setHydrostaticPressureBC(CCVariable<double>& press,
				Patch::FaceType face, Vector& gravity,
				const CCVariable<double>& rho,
				const Vector& dx,
				IntVector offset = IntVector(0,0,0));
    
  void determineSpacingAndGravity(Patch::FaceType face, Vector& dx,
				  SimulationStateP& sharedState,
				  double& spacing, double& gravity);

  void setBC(CCVariable<double>& variable,const std::string& type, 
	     const Patch* p,  SimulationStateP& sharedState,
	     const int mat_id);
  
  void setBC(CCVariable<double>& press_CC, const CCVariable<double>& rho,
             const std::string& whichVar, const std::string& type, 
             const Patch* p, SimulationStateP& sharedState,
             const int mat_id, DataWarehouse*);
  
  void setBC(CCVariable<Vector>& variable,const std::string& type,
             const Patch* p, const int mat_id);
  
  template<class T> void Neuman_SFC(T& var, const Patch* patch,
				    Patch::FaceType face,
				    const double value, const Vector& dx,
				    IntVector offset = IntVector(0,0,0));

  void determineSpacingAndSign(Patch::FaceType face, Vector& dx,
			       double& spacing, double& sign);  

  void setBC(SFCXVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);

  void setBC(SFCYVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);  

  void setBC(SFCZVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);   
  
  void setBC(SFCXVariable<Vector>& variable,const std::string& type,
             const Patch* p, const int mat_id);
  
  void checkValveBC(CCVariable<Vector>& var, const Patch* patch,
		    Patch::FaceType face); 
  
  void ImplicitMatrixBC(CCVariable<Stencil7>& var, const Patch* patch);
  
} // End namespace Uintah
#endif
