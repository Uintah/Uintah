#ifndef Uintah_Component_Arches_ExtraScalarSrcFactory_h
#define Uintah_Component_Arches_ExtraScalarSrcFactory_h

#include <string>

namespace Uintah {
  class ExtraScalarSrc;
  class ArchesLabel;
  class MPMArchesLabel;
  class VarLabel;

  class ExtraScalarSrcFactory {
  public:
  ExtraScalarSrcFactory();
  ~ExtraScalarSrcFactory();
    
    
  static ExtraScalarSrc* create(const ArchesLabel* label, 
                                const MPMArchesLabel* MAlb,
                                const VarLabel* d_src_label,
                                const std::string d_src_name);
                            
  };
} // End namespace Uintah

#endif 
