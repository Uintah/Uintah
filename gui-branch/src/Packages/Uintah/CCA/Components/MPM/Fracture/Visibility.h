#ifndef UINTAH_MPM_VISIBILITY
#define UINTAH_MPM_VISIBILITY

#include <Core/Geometry/Vector.h>

namespace Uintah {

using namespace SCIRun;

class Visibility {
public:
               Visibility();
               Visibility(int v);
   void        operator=(int v);

   void        setVisible(const int i);
   void        setUnvisible(const int i);
   bool        visible(const int i) const;

   void        modifyWeights(double S[8]) const;
   void        modifyShapeDerivatives(Vector d_S[8]) const;
   
   int         flag() const {return d_flag;}
   
private:
   int         d_flag;
};
} // End namespace Uintah


#endif
