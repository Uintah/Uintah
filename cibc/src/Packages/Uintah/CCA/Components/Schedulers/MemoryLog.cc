
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <iostream>

using namespace std;

namespace Uintah {
  void logMemory(std::ostream& out, unsigned long& total,
		 const std::string& label, const std::string& name,
		 const std::string& type, const Patch* patch,
		 int material, const std::string& nelems,
		 unsigned long size, void* ptr, int dwid)
  {
    out << label;
    if(dwid != -1)
      out << ":" << dwid;
    char tab = '\t';
    out << tab << name << tab << type << tab;
    if(patch)
      out << patch->getID();
    else
      out << "-";
    out << tab << material << tab << nelems << tab << size << tab << ptr << '\n';
    total += size;
  }
}

