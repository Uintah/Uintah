#ifndef Uintah_Component_Arches_BOUNDARYFUNCTORHELPER_h
#define Uintah_Component_Arches_BOUNDARYFUNCTORHELPER_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>

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

  static Uintah::ProblemSpecP get_uintah_bc_problem_spec( Uintah::ProblemSpecP& db, std::string var_name, std::string face_name ){

    ProblemSpecP db_root = db->getRootNode();
    ProblemSpecP db_bc = db_root->findBlock("Grid")->findBlock("BoundaryConditions");

    if ( db_bc ){

      for ( Uintah::ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != 0;
            db_face = db_face->findNextBlock("Face") ){

        std::string fname, vname;

        db_face->getAttribute("name",fname);
        if ( fname == face_name ){

          for ( Uintah::ProblemSpecP db_bc_type = db_face->findBlock("BCType"); db_bc_type != 0;
                db_bc_type = db_bc_type->findNextBlock("BCType") ){
            db_bc_type->getAttribute("label", vname );

            if ( vname == var_name ){
              return db_bc_type;
            }

          }
        }
      }

      //if you made it this far, this means that we can't find a matching spec for the parameters given
      std::stringstream msg;
      msg << "Error: Cannot find a matching bc in <Grid><BoundaryConditions> for [varname, facename]: [" << var_name << ", " << face_name << "]" << std::endl;
      throw Uintah::ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }

}} // namespace Uintah::ArchesCore

#endif
