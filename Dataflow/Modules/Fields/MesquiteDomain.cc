
#include <Dataflow/Modules/Fields/MesquiteDomain.h>

#include <Vector3D.hpp>
#include <MsqError.hpp>
#include <iostream>

namespace SCIRun {

using namespace std;

//! Modifies "coordinate" so that it lies on the
//! domain to which "entity_handle" is constrained.
//! The handle determines the domain.  The coordinate
//! is the proposed new position on that domain.
void MesquiteDomain::snap_to(
  Mesquite::Mesh::EntityHandle entity_handle,
  Mesquite::Vector3D &coordinate) const
{
  cout << "ERROR: MesquiteDomain::snap_to is not currently implemented." << endl;
  
//     // Get the domain for the entity_handle
//   MRefEntity* owner = reinterpret_cast<MeshEntity*>(entity_handle)->owner();
  
//     // If it's a surface, curve, or vertex, snap back to owner
//   switch (owner->dimension())
//   {
//     case 0:
//         //PRINT_INFO("\nCASE 0");
//       *reinterpret_cast<CubitVector*>(&coordinate) =
//         dynamic_cast<RefVertex*>(owner)->coordinates();
//       break;
//     case 1:
//         //PRINT_INFO("\nCASE 1");
//       dynamic_cast<RefEdge*>(owner)->move_to_curve(
//         *reinterpret_cast<CubitVector*>(&coordinate));
//       break;
//     case 2:
//         //PRINT_INFO("\nCASE 2");
//       dynamic_cast<RefFace*>(owner)->move_to_surface(
//         *reinterpret_cast<CubitVector*>(&coordinate));
//       break;
//   }
}

//! Returns the normal of the domain to which
//! "entity_handle" is constrained.  For non-planar surfaces,
//! the normal is calculated at the point on the domain that
//! is closest to the passed in value of "coordinate".  If the
//! domain does not have a normal, or the normal cannot
//! be determined, "coordinate" is set to (0,0,0).  Otherwise,
//! "coordinate" is set to the domain's normal at the
//! appropriate point.
//! In summary, the handle determines the domain.  The coordinate
//! determines the point of interest on that domain.
void MesquiteDomain::normal_at(
  Mesquite::Mesh::EntityHandle entity_handle,
  Mesquite::Vector3D &coordinate) const
{
  cout << "ERROR: MesquiteDomain::normal_at is not currently implemented." << endl;
  
//       // Get the domain for the entity_handle
//   MRefEntity* owner = reinterpret_cast<MeshEntity*>(entity_handle)->owner();

//     // Only get normal if it's a surface owner
//   if (owner && owner->dimension() == 2)
//     *reinterpret_cast<CubitVector*>(&coordinate) =
//       dynamic_cast<RefFace*>(owner)->normal_at(
//         *reinterpret_cast<CubitVector*>(&coordinate));
//   else
//     coordinate.set(0.0,0.0,0.0);
  
}

void MesquiteDomain::normal_at(
  const Mesquite::Mesh::EntityHandle* entity_handles,
  Mesquite::Vector3D coordinates[],
  unsigned count,
  Mesquite::MsqError &/*err*/) const
{  
  cout << "ERROR: MesquiteDomain::normal_at(b) is not currently implemented." << endl;  

//   int i;
//   for (i=0; i<count; ++i){
//     normal_at(entity_handles[i], coordinates[i]);
//   }
}
 
void MesquiteDomain::closest_point(
  Mesquite::Mesh::EntityHandle handle,
  const Mesquite::Vector3D& position,
  Mesquite::Vector3D& closest,
  Mesquite::Vector3D& normal,
  Mesquite::MsqError& /*err*/ ) const
{
  cout << "ERROR: MesquiteDomain::closest_point is not currently implemented." << endl;
  
   // Get the domain for the entity_handle
//   MeshEntity* temp_ent = reinterpret_cast<MeshEntity*>(handle);
  
//   MRefEntity* owner = temp_ent->owner();
  
//   CubitVector cubit_closest(0.0,0.0,0.0);
//   CubitVector cubit_normal(0.0,0.0,0.0);
//   CubitVector cubit_position(position[0], position[1], position[2]);
  
//     // Only get normal if it's a surface owner
//   if (owner && owner->dimension() == 2){
//     RefFace* temp_rface = dynamic_cast<RefFace*>(owner);
//     if(temp_rface){
//       temp_rface->find_closest_point_trimmed(cubit_position,
//                                              cubit_closest);
//       temp_rface->get_point_normal(cubit_closest, cubit_normal);
//     }
//   }
//   closest.set(cubit_closest.x(),cubit_closest.y(),cubit_closest.z());
  
//   normal.set(cubit_normal.x(),cubit_normal.y(),cubit_normal.z());
}

void MesquiteDomain::domain_DoF(
  const Mesquite::Mesh::EntityHandle* handle_array,
  unsigned short* dof_array,
  size_t num_handles,
  Mesquite::MsqError& err ) const
{
  cout << "ERROR: MesquiteDomain::domain_DoF is not currently implemented." << endl;
//   int i;
//   MRefEntity* owner=NULL;
  
//   for(i=0;i<num_handles;++i){
//       // Get the domain for the entity_handle
//     owner = reinterpret_cast<MeshEntity*>(handle_array[i])->owner();
  
//       // If it's a surface, curve, or vertex, snap back to owner
//     int temp_int = owner->dimension();
    
//     switch (temp_int)
//     {
//       case 0:
//       case 1:
//       case 2:
//       case 3:
//         dof_array[i] = (short) temp_int;
//         break;
//       default:
//         MSQ_SETERR(err)("Unexpected dimension.",
//                         Mesquite::MsqError::INVALID_STATE);
//         return;
//     };
//   }  
}

} //namespace SCIRun
