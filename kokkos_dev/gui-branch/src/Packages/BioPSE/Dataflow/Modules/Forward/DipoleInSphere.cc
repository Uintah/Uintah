/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>


#include <iostream>
using std::cerr;
#include <stdio.h>
#include <math.h>

namespace BioPSE {
using namespace SCIRun;

typedef LockingHandle<TriSurf<double> > TriSurfHandle;
typedef LockingHandle<TriSurfMesh > TriSurfMeshHandle;

class DipoleInSphere : public Module {
  
  //! Private Data

  //! input ports
  FieldIPort*  iportGeom_;
  FieldIPort*  iportDip_;

  //! output port
  FieldOPort*  oportPot_;

  //! Private Methods
  // -- fills in the surface with potentials for single sphere uniform model
  void fillOneSpherePotentials(DenseMatrix&, TriSurfHandle);

public:
  
  DipoleInSphere(const string& id);  
  virtual ~DipoleInSphere();
  virtual void execute();
};

extern "C" Module* make_DipoleInSphere(const string& id)
{
  return new DipoleInSphere(id);
}

DipoleInSphere::DipoleInSphere(const string& id)
: Module("DipoleInSphere", id, Filter, "Forward", "BioPSE")
{
}

DipoleInSphere::~DipoleInSphere()
{
}

void DipoleInSphere::execute() {
  update_state(NeedData);
  
  iportGeom_ = (FieldIPort *)get_iport("Sphere");
  iportDip_ = (FieldIPort *)get_iport("Dipole Sources");
  oportPot_ = (FieldOPort *)get_oport("SphereWithPots");
  FieldHandle field_handle;

  if (!iportGeom_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!iportDip_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!oportPot_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  
  if (!iportGeom_->get(field_handle)){
    msgStream_ << "Cann't get mesh data" << endl;
    return;
  }
  
  if (!field_handle.get_rep()) {
    msgStream_ << "Error: empty mesh" << endl;
    return;
  }
 
  if (field_handle->get_type_name(0) == "TriSurf" && field_handle->get_type_name(1) == "double"){
    
    TriSurf<double>* pSurf = dynamic_cast<TriSurf<double>*>(field_handle.get_rep());
    TriSurfMeshHandle hMesh = new TriSurfMesh(*(pSurf->get_typed_mesh().get_rep()));
    TriSurfHandle hNewSurf = new TriSurf<double>(hMesh, Field::NODE);
    
    FieldHandle dip_handle;
    
    if (iportDip_->get(dip_handle) 
	&& dip_handle.get_rep() &&
	dip_handle->get_type_name(0) == "PointCloud" 
	&& dip_handle->get_type_name(1) == "Vector"){
      
      PointCloud<Vector>*  pDips = dynamic_cast<PointCloud<Vector>*>(dip_handle.get_rep());
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
      
      DenseMatrix dip_mtrx(pos.size(), 6);
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
      fillOneSpherePotentials(dip_mtrx, hNewSurf);
      oportPot_->send(FieldHandle(hNewSurf.get_rep()));
    }
    else {
      msgStream_ << "No dipole info found in the mesh supplied or supplied field is not of type PointCloud<Vector>" << endl;
    }
   
  }
  else {
    msgStream_ << "Error: the supplied field is not of type TriSurf<double>" << endl;
    return;
  }
}

void DipoleInSphere::fillOneSpherePotentials(DenseMatrix& dips, TriSurfHandle hSurf) {
  
  TriSurfMeshHandle hMesh = hSurf->get_typed_mesh();
  vector<double>& data = hSurf->fdata();
  TriSurfMesh::Node::size_type nsize; hMesh->size(nsize);
  data.resize(nsize, 0);
  BBox bbox = hMesh->get_bounding_box();
  
  if (!bbox.valid()){
    msgStream_ << "No valid mesh" << endl;
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
    }
    ++niter;
  }
}

} // End namespace BioPSE
