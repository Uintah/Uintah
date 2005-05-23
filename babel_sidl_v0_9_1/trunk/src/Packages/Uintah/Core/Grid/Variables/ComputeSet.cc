
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <iostream>

using namespace std;
using namespace Uintah;

std::ostream& operator<<(std::ostream& out, const Uintah::PatchSet& ps)
{
  if(&ps == 0)
    out << "(null PatchSet)";
  else {
    out << "Patches: {";
    for(int i=0;i<ps.size();i++){
      const PatchSubset* pss = ps.getSubset(i);
      if(i != 0)
	out << ", ";
      out << *pss;
    }
    out << "}";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Uintah::MaterialSet& ms)
{
  if(&ms == 0)
    out << "(null Materials)";
  else {
    out << "Matls: {";
    for(int i=0;i< ms.size();i++){
      const MaterialSubset* mss = ms.getSubset(i);
      if(i != 0)
	out << ", ";
      out << *mss;
    }
    out << "}";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Uintah::PatchSubset& pss)
{
  out << "{";
  for(int j=0;j<pss.size();j++){
    if(j != 0)
      out << ",";
    const Patch* patch = pss.get(j);
    out << patch->getID();
  }
  out << "}";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Uintah::MaterialSubset& mss)
{
  out << "{";
  for(int j=0;j<mss.size();j++){
    if(j != 0)
      out << ",";
    out << mss.get(j);
  }
  out << "}";
  return out;
}

