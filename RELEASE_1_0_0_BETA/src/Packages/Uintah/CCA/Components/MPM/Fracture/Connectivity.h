#ifndef Uintah_Connectivity
#define Uintah_Connectivity

#include <Core/Geometry/Vector.h>

namespace Uintah {

using namespace SCIRun;

/*  
    0 directly unconnected
    1 directly connected
    2 contact
*/

class Connectivity {
public:
  enum Cond {unconnect,connect,contact};
  
                     Connectivity();
                     Connectivity(int v);
                     Connectivity(const int info[8]);
         void        operator=(int v);

         void        getInfo(int info[8]) const;
         void        setInfo(const int info[8]);

  static void        modifyWeights(const int connectivity[8],
                                   double S[8],
				   Cond cond);
  static void        modifyShapeDerivatives(const int connectivity[8],
                                            Vector d_S[8],
					    Cond cond);

         int         flag() const {return d_flag;}
   
private:
         int         d_flag;
};

} // End namespace Uintah

#endif
