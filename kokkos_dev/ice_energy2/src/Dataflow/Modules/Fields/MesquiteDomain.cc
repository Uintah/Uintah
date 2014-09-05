
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
  Point n_p, new_result;
  n_p.x( coordinate[0] );
  n_p.y( coordinate[1] );
  n_p.z( coordinate[2] );
  TriSurfMesh<TriLinearLgn<Point> >::Face::index_type face_id;
  domain_mesh_->find_closest_face( new_result, face_id, n_p );    
  coordinate.set( new_result.x(), new_result.y(), new_result.z() );
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
  Point n_p, new_result;
  n_p.x( coordinate[0] );
  n_p.y( coordinate[1] );
  n_p.z( coordinate[2] );
  TriSurfMesh<TriLinearLgn<Point> >::Face::index_type face_id;
  domain_mesh_->find_closest_face( new_result, face_id, n_p );  
  Vector result;
  vector<double> bogus;
  bogus.push_back( new_result.x() );
  bogus.push_back( new_result.y() );
  bogus.push_back( new_result.z() );
  domain_mesh_->get_normal( result, bogus, face_id, 0);
  coordinate.set( result.x(), result.y(), result.z() );
//    coordinate.set(0.0,0.0,0.0);

//  cout << "WARNING: MesquiteDomain::normal_at = <" << result.x() << "," << result.y() << "," << result.z() << ">" << endl;
  

//       // Get the domain for the entity_handle
//   MRefEntity* owner = reinterpret_cast<MeshEntity*>(entity_handle)->owner();

//     // Only get normal if it's a surface owner
//   if (owner && owner->dimension() == 2)
//   {
//     *reinterpret_cast<Vector*>(&coordinate) =
//       dynamic_cast<RefFace*>(owner)->normal_at(
//         *reinterpret_cast<Vector*>(&coordinate));
//   }  
//   else
//   {
//  coordinate.set(0.0,0.0,0.0);
//   }

}

void MesquiteDomain::normal_at(
  const Mesquite::Mesh::EntityHandle* entity_handles,
  Mesquite::Vector3D coordinates[],
  unsigned count,
  Mesquite::MsqError &/*err*/) const
{  
  unsigned int i;
  for( i = 0; i < count; ++i )
  {
    normal_at( entity_handles[i], coordinates[i] );
  }
}
 
void MesquiteDomain::closest_point(
  Mesquite::Mesh::EntityHandle handle,
  const Mesquite::Vector3D& position,
  Mesquite::Vector3D& closest,
  Mesquite::Vector3D& normal,
  Mesquite::MsqError& /*err*/ ) const
{
  Point p, close_pt;
  p.x( position[0] );
  p.y( position[1] );
  p.z( position[2] );
  TriSurfMesh<TriLinearLgn<Point> >::Face::index_type face_id;
  domain_mesh_->find_closest_face( close_pt, face_id, p );    
  closest.set( close_pt.x(), close_pt.y(), close_pt.z() );
 
    //Need to set the normal...
  Vector result;
  vector<double> bogus;
  bogus.push_back( close_pt.x() );
  bogus.push_back( close_pt.y() );
  bogus.push_back( close_pt.z() );
  domain_mesh_->get_normal( result, bogus, face_id, 0);
  normal.set( result.x(), result.y(), result.z() );
//  normal.set( 0., 0., 0. );
  
//  cout << "WARNING: MesquiteDomain::closest_point <" << p.x() << "," << p.y() << "," << p.z() << "> to <" << close_pt.x() << "," << close_pt.y() << "," << close_pt.z() << ">" << endl;

   // Get the domain for the entity_handle
//   MeshEntity* temp_ent = reinterpret_cast<MeshEntity*>(handle);
  
//   MRefEntity* owner = temp_ent->owner();
  
//   Vector closest(0.0,0.0,0.0);
//   Vector normal(0.0,0.0,0.0);
//   Vector position(position[0], position[1], position[2]);
  
//     // Only get normal if it's a surface owner
//   if (owner && owner->dimension() == 2)
//   {
//     RefFace* temp_rface = dynamic_cast<RefFace*>(owner);
//     if(temp_rface){
//       temp_rface->find_closest_point_trimmed(position,
//                                              closest);
//       temp_rface->get_point_normal(closest, normal);
//     }
//   }
//   closest.set(closest.x(), closest.y(), closest.z());
  
//   normal.set(normal.x(),normal.y(),normal.z());
}

void MesquiteDomain::domain_DoF(
  const Mesquite::Mesh::EntityHandle* handle_array,
  unsigned short* dof_array,
  size_t num_handles,
  Mesquite::MsqError& err ) const
{
//  cout << "WARNING: MesquiteDomain::domain_DoF was called." << endl;
    //Since we are only supporting TriSurfMeshes for now, simply fill
    // the array with 2...
//   MRefEntity* owner=NULL;
  
  size_t i;
  for( i = 0; i < num_handles; ++i )
  {
//       // Get the domain for the entity_handle
//     owner = reinterpret_cast<MeshEntity*>(handle_array[i])->owner();
  
//       // If it's a surface, curve, or vertex, snap back to owner
//     int temp_int = owner->dimension();
    
//     switch( temp_int )
//     {
//       case 0:
//       case 1:
//       case 2:
//       case 3:
//         dof_array[i] = (short) temp_int;
    dof_array[i] = (short)2;
//         break;
//       default:
//         MSQ_SETERR(err)("Unexpected dimension.",
//                         Mesquite::MsqError::INVALID_STATE);
//         return;
//     };
  }  
}

} //namespace SCIRun
