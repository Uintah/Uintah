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


/*
 *  DipoleInSphere: Calculation of potential on 
 *                  conducting sphere due to the dipole sources
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Modified by:
 *   Samsonov Alexei
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/GuiInterface/GuiVar.h>


#include <iostream>
#include <stdio.h>
#include <math.h>

namespace BioPSE {
using namespace SCIRun;

class DipoleInSphere : public Module {
  //! Private Methods
  // -- fills in the surface with potentials for single sphere uniform model
  void fillOneSphere(DenseMatrix&, TriSurfField<double>*, TriSurfField<Vector>*);

public:
  
  DipoleInSphere(GuiContext *context);
  virtual ~DipoleInSphere();
  virtual void execute();
};

DECLARE_MAKER(DipoleInSphere)


DipoleInSphere::DipoleInSphere(GuiContext *context)
  : Module("DipoleInSphere", context, Filter, "Forward", "BioPSE")
{
}

DipoleInSphere::~DipoleInSphere()
{
}

void DipoleInSphere::execute() {
  update_state(NeedData);
  
  FieldIPort *iportGeom_ = (FieldIPort *)get_iport("Sphere");
  FieldIPort *iportDip_ = (FieldIPort *)get_iport("Dipole Sources");
  FieldOPort *oportPot_ = (FieldOPort *)get_oport("SphereWithPots");
  FieldOPort *oportMag_ = (FieldOPort *)get_oport("SphereWithMagneticField");
  FieldHandle field_handle;

  if (!iportGeom_->get(field_handle)){
    error("Can't get input mesh data.");
    return;
  }
  
  if (!field_handle.get_rep()) {
    error("Empty input mesh.");
    return;
  }
 
  if (field_handle->get_type_name(0) == "TriSurfField") {
    TriSurfMeshHandle hMesh = 
      dynamic_cast<TriSurfMesh*>(field_handle->mesh().get_rep());
    TriSurfField<double>* hNewSurf = 
      new TriSurfField<double>(hMesh, 1);
    TriSurfField<Vector>* hBSurf = 
      new TriSurfField<Vector>(hMesh, 1);
    
    FieldHandle dip_handle;
    
    if (iportDip_->get(dip_handle) 
	&& dip_handle.get_rep() &&
	dip_handle->get_type_name(0) == "PointCloudField" 
	&& dip_handle->get_type_name(1) == "Vector"){
      
      PointCloudField<Vector>*  pDips = dynamic_cast<PointCloudField<Vector>*>(dip_handle.get_rep());
      PointCloudMeshHandle hMesh = pDips->get_typed_mesh();
      
      PointCloudMesh::Node::iterator ii;
      PointCloudMesh::Node::iterator ii_end;
      Point p;
      Vector qdip;
      vector<Vector> dips;
      vector<Point>  pos;
      hMesh->begin(ii);
      hMesh->end(ii_end);
      for (; ii != ii_end; ++ii) {
	if (pDips->value(qdip, *ii)){
	  dips.push_back(qdip);
	  hMesh->get_point(p, *ii);
	  pos.push_back(p);
	}
      }
      
      DenseMatrix dip_mtrx((int)pos.size(), 6);
      unsigned int i;
      msgStream_ << "Dipoles: " << endl;
      for (i=0; i<pos.size(); ++i){
	qdip = dips[i];
	p = pos[i];
	dip_mtrx[i][0] = p.x(); dip_mtrx[i][1] = p.y();  dip_mtrx[i][2] = p.z();
	dip_mtrx[i][3] = qdip.x(); dip_mtrx[i][4] = qdip.y();  dip_mtrx[i][5] = qdip.z();
	msgStream_ << "Pos: " << p << ", moment: " << qdip << endl;
      }
      
      update_state(JustStarted);
      fillOneSphere(dip_mtrx, hNewSurf, hBSurf);
      oportPot_->send(FieldHandle(hNewSurf));
      oportMag_->send(FieldHandle(hBSurf));
    }
    else {
      warning("No dipole info found in the mesh supplied or supplied field is not of type PointCloudField<Vector>.");
    }
  }
  else {
    error("The supplied field is not of type TriSurfField<double>.");
    return;
  }
}

void DipoleInSphere::fillOneSphere(DenseMatrix& dips, TriSurfField<double>* hSurf, TriSurfField<Vector>* hBSurf) {
  
  TriSurfMeshHandle hMesh = hSurf->get_typed_mesh();
  vector<double>& data = hSurf->fdata();
  vector<Vector>& bdata = hBSurf->fdata();
  TriSurfMesh::Node::size_type nsize; hMesh->size(nsize);

  BBox bbox = hMesh->get_bounding_box();
  
  if (!bbox.valid()){
    error("No valid input mesh.");
    return;
  }

  double R = 0.5*bbox.longest_edge();
  
  double gamma=1;
  double E[3];
  msgStream_ << "Radius of the sphere is " << R << endl;
  Point p;

  TriSurfMesh::Node::iterator niter; hMesh->begin(niter);
  TriSurfMesh::Node::iterator niter_end; hMesh->end(niter_end);

  // -- for every point
  while (niter != niter_end) {
    data[*niter]=0;
    bdata[*niter]=Vector(0,0,0);
    hMesh->get_point(p, *niter);
      
    // -- for every dipole
    int id;
    for (id = 0; id < dips.nrows(); ++id){
	
      double V = 0.0;
      E[0] = p.x();
      E[1] = p.y();
      E[2] = p.z();

      double rho = sqrt( pow((E[0] - dips[id][0]),2) + pow((E[1] - dips[id][1]),2) + pow((E[2] - dips[id][2]),2));
      double S = E[0]*dips[id][0] + E[1]*dips[id][1] + E[2]*dips[id][2];
	
      for(int k=0;k<3;k++) {
	double F[3];
	F[k] = (1/(4*M_PI*gamma*rho)) * (2*(E[k]-dips[id][k])/pow(rho,2) +
					 (1/pow(R,2)) * (E[k] + (E[k]*S/R - R*dips[id][k])/(rho+R-S/R)));
	V += F[k]*dips[id][k+3];
	  
      }
	
      data[*niter] += V;


      // magnetic
      Vector r_vec = p.asVector()*1.01;
      Vector r0_vec = Vector(dips[id][0], dips[id][1], dips[id][2]);
      double r_mag = r_vec.length();
      Vector a_vec = r_vec-r0_vec;
      double a_mag = a_vec.length();
      Vector Q_vec(dips[id][3], dips[id][4], dips[id][5]);
      double F_mag = a_mag*(r_mag*a_mag + r_mag*r_mag - Dot(r0_vec, r_vec));
      Vector gradF_vec((a_mag*a_mag/r_mag + Dot(a_vec,r_vec)/a_mag + 2*a_mag +
			2*r_mag)*r_vec -
		       (a_mag + 2*r_mag + Dot(a_vec,r_vec)/a_mag)*r0_vec);
      bdata[*niter] += (F_mag*(Cross(Q_vec,r0_vec)) - 
			Dot(Cross(Q_vec,r0_vec),r_vec)*gradF_vec)/
	(4*M_PI*F_mag*F_mag);
    }
    ++niter;
  }
}
} // End namespace BioPSE
