#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctorHelper.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah { namespace ArchesCore{
Uintah::ProblemSpecP get_uintah_bc_problem_spec( Uintah::ProblemSpecP& db, std::string var_name, std::string face_name ){

  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc = db_root->findBlock("Grid")->findBlock("BoundaryConditions");

  if ( db_bc ){

    for ( Uintah::ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ){

      std::string fname, vname;

      db_face->getAttribute("name",fname);
      if ( fname == face_name ){

        for ( Uintah::ProblemSpecP db_bc_type = db_face->findBlock("BCType"); db_bc_type != nullptr; db_bc_type = db_bc_type->findNextBlock("BCType") ){
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

  return NULL;

}

} } //namespace Uintah::ArchesCore
