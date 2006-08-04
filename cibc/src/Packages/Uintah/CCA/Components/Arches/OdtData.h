#ifndef Uintah_Component_Arches_OdtData_h
#define Uintah_Component_Arches_OdtData_h

#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

namespace Uintah {
  struct odtData { 
	  	double x_x[10]; //x coordinates along x direction line
		double y_y[10]; //y coordinates along x direction line
		double z_z[10]; //z coordinates along x direction line
		double x_u[10]; //u velocity along x direction line
  		double x_v[10]; //v velocity along x direction line
		double x_w[10]; //w velocity along x direction line
		double x_rho[10]; //density along x direction line 
		double x_T[10]; //temperature along x direction line
		double x_Phi[10]; //scalar  along x direction line
		double y_u[10]; //u velocity along y direction line
  		double y_v[10]; //v velocity along y direction line
		double y_w[10]; //w velocity along y direction line
		double y_rho[10]; //density along y direction line 
		double y_T[10]; //temperature along y direction line
		double y_Phi[10]; //scalar  along y direction line
		double z_u[10]; //u velocity along z direction line
  		double z_v[10]; //v velocity along z direction line
		double z_w[10]; //w velocity along z direction line
		double z_rho[10]; //density along z direction line 
		double z_T[10]; //temperature along z direction line
		double z_Phi[10]; //scalar  along z direction line

  };    


//________________________________________________________________________
  const TypeDescription* fun_getTypeDescription(odtData*);    

} // End namespace Uintah
  
namespace SCIRun {
  void swapbytes( Uintah::odtData& d);
}       

#endif
