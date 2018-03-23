#ifndef UT_TransportHelper_h
#define UT_TransportHelper_h

#include <string>

namespace Uintah{ namespace ArchesCore{

  enum EQUATION_CLASS {DENSITY_WEIGHTED, DQMOM, MOMENTUM, NO_PREMULT};

  EQUATION_CLASS assign_eqn_class_enum( std::string my_class );

  std::string get_premultiplier_name(ArchesCore::EQUATION_CLASS eqn_class);

  std::string get_postmultiplier_name(ArchesCore::EQUATION_CLASS eqn_class);

} } //namespace Uintah::ArchesCore

#endif
