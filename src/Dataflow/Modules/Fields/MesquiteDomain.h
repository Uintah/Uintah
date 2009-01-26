/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef MESQUITE_DOMAIN_HPP
#define MESQUITE_DOMAIN_HPP

#include <MeshInterface.hpp>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Basis/TriLinearLgn.h>

namespace SCIRun {

class MesquiteDomain : public Mesquite::MeshDomain
{
public:

  MesquiteDomain( TriSurfMesh<TriLinearLgn<Point> > *domain_mesh )
      {domain_mesh_ = domain_mesh;}
  virtual ~MesquiteDomain()
    {}
  
    //! Modifies "coordinate" so that it lies on the
    //! domain to which "entity_handle" is constrained.
    //! The handle determines the domain.  The coordinate
    //! is the proposed new position on that domain.
  virtual void snap_to(
    Mesquite::Mesh::EntityHandle entity_handle,
    Mesquite::Vector3D &coordinate) const;
  
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
  virtual void normal_at(
    Mesquite::Mesh::EntityHandle entity_handle,
    Mesquite::Vector3D &coordinate) const;
  
  virtual void normal_at(
    const Mesquite::Mesh::EntityHandle* entity_handles,
    Mesquite::Vector3D coordinates[],
    unsigned count,
    Mesquite::MsqError &err) const;
  
  virtual void closest_point(
    Mesquite::Mesh::EntityHandle handle,
    const Mesquite::Vector3D& position,
    Mesquite::Vector3D& closest,
    Mesquite::Vector3D& normal,
    Mesquite::MsqError& err
    ) const;
  
  virtual void domain_DoF(
    const Mesquite::Mesh::EntityHandle* handle_array,
    unsigned short* dof_array,
    size_t num_handles,
    Mesquite::MsqError& err ) const;

private:
  TriSurfMesh<TriLinearLgn<Point> > *domain_mesh_;
  
};
    
} //namespace SCIRun

#endif
