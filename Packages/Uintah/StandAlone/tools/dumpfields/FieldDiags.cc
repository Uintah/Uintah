#include "FieldDiags.h"

#include <string>
#include <map>
#include <list>

using namespace std;

namespace Uintah {
  using namespace SCIRun;
  
  FieldDiag::~FieldDiag() {}
  
  bool 
  FieldDiag::has_mass(DataArchive * da, const Patch * patch, 
                      TypeDescription::Type fieldtype,
                      int imat, double time, const IntVector & pt) const
  {
    switch(fieldtype) {
    case TypeDescription::NCVariable: {
      NCVariable<double> Mvalue;
      da->query(Mvalue, "g.mass", imat, patch, time);
      return (Mvalue[pt]>numeric_limits<float>::epsilon());
    }
    case TypeDescription::CCVariable: {
      CCVariable<double> Mvalue;
      da->query(Mvalue, "g.mass", imat, patch, time);
      return (Mvalue[pt]>numeric_limits<float>::epsilon());
    }
    }
  }
}
