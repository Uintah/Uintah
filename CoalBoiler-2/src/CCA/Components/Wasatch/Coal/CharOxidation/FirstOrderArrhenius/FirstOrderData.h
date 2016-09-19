/*
 * FirstOrderData.h
 *
 *  Created on: May 25, 2015
 *      Author: josh
 */

#ifndef FirstOrderData_h
#define FirstOrderData_h

#include <CCA/Components/Wasatch/Coal/CoalData.h>

namespace FOA{

class FirstOrderData {
public:

  FirstOrderData( const Coal::CoalType coalType );

  inline const double get_a_h2o()  const{ return aH2o_; }
  inline const double get_a_co2()  const{ return aCo2_; }
  inline const double get_ea()     const{ return ea_;   }

protected:
  double aH2o_, aCo2_, ea_;

private:
  const Coal::CoalComposition coalComp_;
};
} // namespace FOA

#endif /* FirstOrderData_h */
