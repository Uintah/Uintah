#ifndef Uintah_MPM_Visibility
#define Uintah_MPM_Visibility

#include <SCICore/Geometry/Vector.h>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;

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

}} //namespace

#endif

// $Log$
// Revision 1.2  2000/09/09 19:34:16  tan
// Added MPMLabel::pVisibilityLabel and SerialMPM::computerNodesVisibility().
//
// Revision 1.1  2000/09/09 18:12:05  tan
// Added Visibility class to handle the relationship between a particle
// and the related nodes.
//
