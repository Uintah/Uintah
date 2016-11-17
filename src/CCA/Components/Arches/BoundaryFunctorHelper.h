#ifndef Uintah_Component_Arches_BOUNDARYFUNCTORHELPER_h
#define Uintah_Component_Arches_BOUNDARYFUNCTORHELPER_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah { namespace ArchesCore {

  //NOTICE:
  // The ordering of the BCFunctorType below must match the enum ordering here
  enum BC_FUNCTOR_ENUM
  {
    INVALID_TYPE,
    SWIRL,
    FILE,
    MASSFLOW,
    STABLE,
    TABLELOOKUP
  };

  struct SwirlType{};
  struct FileType{};
  struct MassFlowType{};
  struct StableType{};
  struct TableLookupType{};

  /**
  * @struct General BC Information
  * @author Jeremy T.
  * @date whatever today is
  *
  * @brief This struct will hold a min. set of information that can be passed into a
  *        boundary functor to enable ANY of them to actually work.
  **/
  struct BoundaryFunctorInformation{

    //Inlets:
    Uintah::Vector velocity;
    double mdot;
    double swirl_no;
    Uintah::Vector swirl_cent;

  };

  typedef std::map<std::string, BoundaryFunctorInformation > BCIMap;
  typedef BoundaryFunctorInformation BFI;

  Uintah::ProblemSpecP get_uintah_bc_problem_spec( Uintah::ProblemSpecP& db, std::string var_name, std::string face_name );

}} // namespace Uintah::ArchesCore

#endif
