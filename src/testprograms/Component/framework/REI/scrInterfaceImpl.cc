
#include <testprograms/Component/framework/scrInterfaceImpl.h>

#include <sstream>
#include <iostream>

namespace sci_cca {

using std::cerr;

scrInterfaceImpl::scrInterfaceImpl()
{
}

scrInterfaceImpl::~scrInterfaceImpl()
{
}

void
scrInterfaceImpl::exec( int cells,
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
			::CIA::array1< double>& kmol_s,
			double& noreduction,
			double& new_temp,
			double& kmol_s_tot)
{
  cout << "scrInterfaceImpl: exec()\n";
}

} // namespace sci_cca

