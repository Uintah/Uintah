#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactMaterialSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

ContactMaterialSpec::ContactMaterialSpec(ProblemSpecP & ps)
{
  if(ps) {
    vector<int> materials;
    if(ps->get("materials", materials)) {
      for(vector<int>::const_iterator mit(materials.begin());mit!=materials.end();mit++) {
        if(*mit<0)
          throw ProblemSetupException(" Invalid material index in contact block");
        this->add(*mit);
      }
    }
  }
  
}

void
ContactMaterialSpec::add(unsigned int matlIndex)
{
  // we only add things once at the start, but want 
  // quick lookup, so keep logical for each material
  // rather than searching a list every time
  if(d_matls.size()==0)
    {
      d_matls.resize(matlIndex+1);
      for(size_t i=0;i<matlIndex+1;i++) d_matls[i] = false;
        
    }
  if(matlIndex>=d_matls.size())
    {
      vector<bool> copy(d_matls);
      d_matls.resize(matlIndex+1);
      for(size_t i=0;i<copy.size();i++) d_matls[i] = copy[i];
      for(size_t i=copy.size();i<matlIndex+1;i++) d_matls[i] = false;
    }
  
  d_matls[matlIndex] = true;
}

