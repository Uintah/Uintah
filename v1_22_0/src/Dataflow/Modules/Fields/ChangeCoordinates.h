#ifndef SCIRun_ChangeCoordinates_H
#define SCIRun_ChangeCoordinates_H
/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {

class ChangeCoordinatesAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src, const string &o, const string &n) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class ChangeCoordinatesAlgoT : public ChangeCoordinatesAlgo
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, const string &o, const string &n);
};

template <class MESH>
void 
ChangeCoordinatesAlgoT<MESH>::execute(MeshHandle mesh_h,
				      const string &oldsystem,
				      const string &newsystem)
{
  typedef typename MESH::Node::iterator node_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());
  double theta, phi, r, x, y, z;
  node_iter_type ni; mesh->begin(ni);
  node_iter_type nie; mesh->end(nie);
  while (ni != nie)
  {
    Point pO, pE, pN; // point in "Old", "Cartesian", and "New" coordinates
    mesh->get_point(pO, *ni);

    // transform from old system to Cartesian
    if (oldsystem=="Cartesian") {
      pE=pO;
    } else if (oldsystem=="Spherical") {
      // pO was in Spherical coordinates -- transform it to Cartesian
      theta=pO.x();
      phi=pO.y();
      r=pO.z();
      pE.x(r*sin(phi)*sin(theta));
      pE.y(r*sin(phi)*cos(theta));
      pE.z(r*cos(phi));
    } else if (oldsystem=="Polar") {
      // pO was in Polar coordinates -- transform to Cartesian
      theta=pO.x();
      r=pO.y();
      z=pO.z();
      pE.x(r*sin(theta));
      pE.y(r*cos(theta));
      pE.z(z);
    } else if (oldsystem=="Range") {
      // pO was in Range coordinates -- transform to Cartesian
          // first convert range-to-spherical, then spherical-to-cartesian
      theta=pO.x();
      phi=pO.y();
      r=pO.z();
//      phi=phi*sin(theta); // make it wedge-shaped
      phi=atan(tan(phi)*sin(theta));
      phi=M_PI/2.-phi;    // phi went + to -, but we need it to be just pos
      pE.x(-r*sin(phi)*cos(theta)); // theta sweeps across x
      pE.y(r*cos(phi)); // phi (tilt of scanner) is y
      pE.z(r*sin(phi)*sin(theta)); // depth is z
    }

    x=pE.x();
    y=pE.y();
    z=pE.z();

    // transform from Cartesian to new system
    if (newsystem=="Cartesian") {
      pN=pE;
    } else if (newsystem=="Spherical") {
      theta=atan2(y,x);
      r=sqrt(x*x+y*y+z*z);
      phi=acos(z/r);
      pN.x(theta);
      pN.y(phi);
      pN.z(r);
    } else if (newsystem=="Polar") {
      theta=atan2(x,y);
      r=sqrt(x*x+y*y);
      pN.x(theta);
      pN.y(r);
      pN.z(z);
    } else if (newsystem=="Range") {
      // first convert cartesian-to-spherical, then spherical-to-range
      theta=atan2(z,x);    // theta's in the x/z plane (y is up)
      r=sqrt(x*x+y*y+z*z);
      phi=acos(y/r);       // y is up
      phi=M_PI/2.-phi; // zero at equator
//      phi=phi/sin(theta);      
      phi=atan(tan(phi)/sin(theta));  // wedge shaped
      if (phi>M_PI) phi=M_PI; else if (phi<-M_PI) phi=-M_PI;
      pN.x(-theta);
      pN.y(phi);
      pN.z(r);
    }

    mesh->set_point(pN, *ni);
    ++ni;
  }
}

} // End namespace SCIRun

#endif
