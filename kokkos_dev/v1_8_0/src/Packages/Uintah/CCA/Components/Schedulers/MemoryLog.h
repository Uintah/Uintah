
#ifndef Uintah_MemoryLog_h
#define Uintah_MemoryLog_h

#include <iosfwd>
#include <string>

namespace Uintah {
  class Patch;
  void logMemory(std::ostream& out, unsigned long& total,
		 const std::string& label, const std::string& name,
		 const std::string& type, const Patch* patch,
		 int material, const std::string& elems,
		 unsigned long size, void* ptr, int dwid=-1);
}

#endif
