#include "FieldDiags.h"

#include <string>
#include <map>
#include <list>
#include <numeric>
#include <limits>

using namespace std;
using namespace SCIRun;
using namespace Uintah;
  
FieldDiag::~FieldDiag() {}
  
bool 
FieldDiag::has_mass(DataArchive * da, const Patch * patch, 
                    Uintah::TypeDescription::Type fieldtype,
                    int imat, int index, const IntVector & pt) const
{
  switch(fieldtype) 
    {
    case TypeDescription::NCVariable: {
      NCVariable<double> Mvalue;
      da->query(Mvalue, "g.mass", imat, patch, index);
      return (Mvalue[pt]>numeric_limits<float>::epsilon());
    }
    case TypeDescription::CCVariable: {
      CCVariable<double> Mvalue;
      da->query(Mvalue, "g.mass", imat, patch, index);
      return (Mvalue[pt]>numeric_limits<float>::epsilon());
    }
    default: 
      throw InternalError("Bad type is has_mass()", __FILE__, __LINE__);
    } // end switch

} // end has_mass()

