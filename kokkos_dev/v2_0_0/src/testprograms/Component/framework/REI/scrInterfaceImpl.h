
#ifndef ScrInterfaceImpl_h
#define ScrInterfaceImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/SSIDL/array.h>

namespace sci_cca {

using SSIDL::array1;

class scrInterfaceImpl : virtual public scrInterface {

public:
  scrInterfaceImpl();
  virtual ~scrInterfaceImpl();

  virtual void exec( int cells,
		     double pressure,
		     double kgcat,
		     double NH3ratio,
		     double NH3,
		     double& flow,
		     double temp,
		     double NO,
		     double N2,
		     double H2O,
		     double O2,
		     double sum_of_all_others,
		     double heat_loss,
		     ::SSIDL::array1< double>& kmol_s,
		     double& noreduction,
		     double& new_temp,
		     double& kmol_s_tot);
};

} // namespace sci_cca

#endif 

