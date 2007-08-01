#ifndef Uintah_Component_Arches_ExtraScalarSrcFactory_h
#define Uintah_Component_Arches_ExtraScalarSrcFactory_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class ExtraScalarSrc;
  class ArchesLabel;
  class MPMArchesLabel;
  class VarLabel;

  class ExtraScalarSrcFactory {
  public:
    static ExtraScalarSrcFactory& self();
    
    
    ExtraScalarSrc* create(const ArchesLabel* label, 
			   const MPMArchesLabel* MAlb,
                           const VarLabel* d_src_label,
                           const std::string d_src_name);
                              
  private:
    ExtraScalarSrcFactory();
    ~ExtraScalarSrcFactory();
    ExtraScalarSrcFactory(const ExtraScalarSrcFactory&); // no copying
    ExtraScalarSrcFactory& operator=(const ExtraScalarSrcFactory&); // no assignment
  };

} // End namespace Uintah

#endif 
