#ifndef c0_fun_h
#define c0_fun_h

namespace CPD{

  /**
   *  \ingroup CPD
   *  \fn double c0_fun( const double C, const double O )
   *  This function calculate $c_{0}$ that represent
   *  initial amount of char which enter to the devolatilization process from [1]
   *  
   *  [1] "Validation and Uncertainty Quantifcation of Coal Devolatization Models", 
   *     Julien Pedel, Jeremy Thornock, Philip Smith, Univesity of Utah
   * 
   */
  inline double c0_fun( const double C, const double O )
  {
//    if (C > 0.859 && C < 0.889265) {
//      return 11.83 * C - 10.16;
//    }
//    
//    if (O > 0.125 && O < 0.2321) {
//      return 1.4 * O - 0.175;
//    }
//    return 0.0;
   double c0=0;
   if( C > 0.859 ) c0 = 11.83 * C - 10.16;
   if( O > 0.125 ) c0 = 1.4 *  O - 0.175;
   return c0;
  }


} // namespace CPD

#endif // c0_fun_h
