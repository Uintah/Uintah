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
 *  EITAnalyticSolution.cc:
 *
 *  Written by:
 *   Lorena Kreda
 *   June 17, 2003
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>


namespace BioPSE {

using namespace SCIRun;
typedef LockingHandle<TetVolField<int> >    CondTetFieldHandle;
typedef LockingHandle<TriSurfField<int> >    CondTriFieldHandle;

class EITAnalyticSolution : public Module {
  //! Private data

  //! Private methods
  double pot2DHomogeneous(int numElectrodes, float r, float theta, 
               float outerRadius, int k, float sigma0);

  double pot2DTwoConcentric(int numElectrodes, float r, float theta, 
               float outerRadius, float innerRadius, int k, 
               float sigma0, float sigma1);

public:
  GuiDouble outerRadiusTCL_;
  GuiDouble innerRadiusTCL_;
  GuiString bodyGeomTCL_; //"Homogeneous disk" or "Concentric disks" 

  //! Constructor/Destructor
  EITAnalyticSolution(GuiContext*);
  virtual ~EITAnalyticSolution();

  //! Public methods
  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(EITAnalyticSolution)
EITAnalyticSolution::EITAnalyticSolution(GuiContext* context)
  : Module("EITAnalyticSolution", context, Source, "Forward", "BioPSE"),
    outerRadiusTCL_(context->subVar("outerRadiusTCL")),
    innerRadiusTCL_(context->subVar("innerRadiusTCL")),
    bodyGeomTCL_(context->subVar("bodyGeomTCL"))
{
}

EITAnalyticSolution::~EITAnalyticSolution(){
}

double
EITAnalyticSolution::pot2DHomogeneous(int L, float r, float theta, 
               float r0, int k, float sigma0) 
{
  double u; // computed potential value

  // L is the number of electrodes, must be an even number
  // r, theta is the point at which to compute solution
  // r0 is the outer radius
  // k is the current pattern index
  // sigma0 is the conductivity in the entire region, if homogeneous

  // Formula for the solution depends on k and the total number of electrodes
 
  if (k<((L/2)+1))
  {
    u=(r0/(sigma0*k))*pow(r/r0,k)*cos(k*theta);
  }
  else
  {
    int kNew = k-(L/2);
    u=(r0/(sigma0*kNew))*pow(r/r0,kNew)*sin(kNew*theta);
  }
  return u;
}

double
EITAnalyticSolution::pot2DTwoConcentric(int L, float r, float theta, 
               float r0, float r1, int k, 
               float sigma0, float sigma1) 
{
  double u; // computed potential value

  // L is the number of electrodes, must be an even number
  // r, theta is the point at which to compute solution
  // r0 is the outer radius, r1 is the inner radius
  // k is the current pattern index
  // sigma0 is the conductivity in the outer annular region
  // sigma1 is the conductivity in the center circular region

  // Formula for the solution depends on k and the total number of electrodes

  double mu = (sigma0 - sigma1)/(sigma0 + sigma1);
  double gamma = r1/r0;

  if (r>=r1)
  { 
    double gamma2 = r1/r;
    if (k<((L/2)+1))
    {
      u=(r0/(sigma0*k))*pow(r/r0,k)*((1+mu*pow(gamma2,2*k))/(1-mu*pow(gamma,2*k)))*cos(k*theta);
    }
    else
    {   
      int kNew = k-(L/2);
      u=(r0/(sigma0*kNew))*pow(r/r0,kNew)*((1+mu*pow(gamma2,2*kNew))/(1-mu*pow(gamma,2*kNew)))*sin(kNew*theta);
    }
  }
  else
  {
    if (k<((L/2)+1))
    {
      u=(r0/(sigma0*k))*pow(r/r0,k)*((1+mu)/(1-mu*pow(gamma,2*k)))*cos(k*theta);
    }
    else
    {
      int kNew = k-(L/2);
      u=(r0/(sigma0*kNew))*pow(r/r0,kNew)*((1+mu)/(1-mu*pow(gamma,2*kNew)))*sin(kNew*theta);
    }
  }
  return u;
}

void
EITAnalyticSolution::execute(){

  //! Input ports
  MatrixIPort*  iportCurrentPatternIndex_;
  FieldIPort*   iportField_;
  MatrixIPort*  iportElectrodeParams_;
  //! Output ports
  MatrixOPort*  oportPotentialVector_;

  // This module is currently able to process only TetVols and
  // TriSurfs - the following flag used is to indicate which
  bool tet;

  iportCurrentPatternIndex_ = (MatrixIPort *)get_iport("CurrentPatternIndex");
  iportField_ = (FieldIPort *)get_iport("Mesh");
  iportElectrodeParams_ = (MatrixIPort *)get_iport("Electrode Parameters");

  oportPotentialVector_ = (MatrixOPort *)get_oport("PotentialVector");

  //! Obtaining handles to computation objects
  FieldHandle hField;
  
  if (!iportField_->get(hField) || !hField.get_rep()) {
    error("Can't get handle to input mesh.");
    return;
  }

  CondTetFieldHandle hCondTetField;
  CondTriFieldHandle hCondTriField;

  TetVolMeshHandle hTetMesh;
  TriSurfMeshHandle hTriMesh;

  if (hField->get_type_name(0)=="TetVolField" && hField->get_type_name(1)=="int"){
    tet = true;
    hCondTetField = dynamic_cast<TetVolField<int>*> (hField.get_rep());
    hTetMesh = hCondTetField->get_typed_mesh();
  }
  else if (hField->get_type_name(0)=="TriSurfField" && hField->get_type_name(1)=="int"){
    tet = false;
    hCondTriField = dynamic_cast<TriSurfField<int>*> (hField.get_rep());
    hTriMesh = hCondTriField->get_typed_mesh();
  }
  else {
    error("Supplied field is not of type TetVolField<int> or TriSurfField<int>.");
    return;
  }

  // Get the current pattern index  
  MatrixHandle  hCurrentPatternIndex;
  ColumnMatrix* currPatIdx;
  int           k;

  // -- copy the input current index into local variable, k 
  if (iportCurrentPatternIndex_->get(hCurrentPatternIndex) && 
      (currPatIdx=dynamic_cast<ColumnMatrix*>(hCurrentPatternIndex.get_rep())) && 
      (currPatIdx->nrows() == 1))
  {
    k=static_cast<int>((*currPatIdx)[0]);
  }
  else{
    msgStream_ << "The supplied current pattern index is not a 1x1 matrix" << endl;
  }

  // Find the number of electrodes from the input electrode parameters

  int numParams=4;

  // Get the electrode parameters input vector
  // -----------------------------------------
  MatrixHandle  hElectrodeParams;

  if (!iportElectrodeParams_->get(hElectrodeParams) || !hElectrodeParams.get_rep()) 
  {
      error("Can't get handle to electrode parameters matrix.");
      return;
  }

  ColumnMatrix* electrodeParams = scinew ColumnMatrix(numParams);
  electrodeParams=dynamic_cast<ColumnMatrix*>(hElectrodeParams.get_rep());

  //unsigned int electrodeModel = (unsigned int)((*electrodeParams)[0]);
  int numElectrodes           = (int) ( (*electrodeParams)[1]);
  //double electrodeLen         = (*electrodeParams)[2];
  //int startNodeIndex          = (int) (*electrodeParams)[3];


  cout << "number of electrodes = " << numElectrodes << endl;

  float r, theta;
  float sigma0, sigma1;
  vector<pair<string, Tensor> > tens;

  // Allocate space for the output vector of potentials
  TetVolMesh::Node::size_type nsizeTet; 
  TriSurfMesh::Node::size_type nsizeTri;
  ColumnMatrix* potential;
  
  if (tet) {
    hTetMesh->size(nsizeTet);
    potential = scinew ColumnMatrix(nsizeTet);
  } else {
    hTriMesh->size(nsizeTri);
    potential = scinew ColumnMatrix(nsizeTri);
  }

  // Find the conductivities from the mesh data. Then for each
  // point in the mesh, convert to (r,theta) and compute potential
  // value according to the appropriate equation.


  if (tet)
  {
    // get conductivity tensors - assume homogeneity (diag terms equal, off diag zero)
    // assume conductivity for the entire body or annular region, if concentric 
    // disks, is the 0th tensor. assume conductivity for center region, if conc. disks,
    // is the 1st tensor
    if (hCondTetField->get_property("conductivity_table", tens))
    {
      double (&el_cond)[3][3] = tens[0].second.mat_;
      sigma0 = el_cond[0][0];
//      el_cond = tens[1].second.mat_;
      double (&el_cond2)[3][3] = tens[1].second.mat_;
      sigma1 = el_cond2[0][0];
    }
    else
    {
      sigma0 = 1.0;
      sigma1 = 1.0;
    }
    string bodyGeom;
    bodyGeom = bodyGeomTCL_.get();
    TetVolMesh::Node::iterator ii;
    TetVolMesh::Node::iterator ii_end;
    hTetMesh->begin(ii);
    hTetMesh->end(ii_end);
    int i=0;
    for (; ii != ii_end; ++ii) 
    {
      Point p;
      hTetMesh->get_point(p, *ii);
      r = sqrt( p.x()*p.x() + p.y()*p.y() );
      theta = atan(p.y()/p.x());
      if ((p.x()<0 && p.y()<0)||(p.x()<0 && p.y()>0)) theta = theta + M_PI;
      if (bodyGeom == "Homogeneous disk") 
      {
        (*potential)[i] = pot2DHomogeneous(numElectrodes, r, theta, outerRadiusTCL_.get(), 
                          k, sigma0);
      }
      else
      { 
        (*potential)[i] = pot2DTwoConcentric(numElectrodes, r, theta, outerRadiusTCL_.get(), 
                         innerRadiusTCL_.get(), k, sigma0, sigma1);
      
      }
      i++;
    }
  }
  else
  {
    // get conductivity tensors - assume homogeneity (diag terms equal, off diag zero)
    // assume conductivity for the entire body or annular region, if concentric 
    // disks, is the 0th tensor. assume conductivity for center region, if conc. disks,
    // is the 1st tensor
    if (hCondTriField->get_property("conductivity_table", tens))
    {
      double (&el_cond)[3][3] = tens[0].second.mat_;
      sigma0 = el_cond[0][0];
//      el_cond = tens[1].second.mat_;
      double (&el_cond2)[3][3] = tens[1].second.mat_;
      sigma1 = el_cond2[0][0];
    }
    else
    {
      sigma0 = 1.0;
      sigma1 = 1.0;
    }
    string bodyGeom;
    bodyGeom = bodyGeomTCL_.get();
    TriSurfMesh::Node::iterator ii;
    TriSurfMesh::Node::iterator ii_end;
    hTriMesh->begin(ii);
    hTriMesh->end(ii_end);
    int i=0;
    for (; ii != ii_end; ++ii) 
    {
      Point p;
      hTriMesh->get_point(p, *ii);
      r = sqrt(p.x()*p.x() + p.y()*p.y());
      theta = atan(p.y()/p.x());      
      if ((p.x()<0 && p.y()<0)||(p.x()<0 && p.y()>0)) theta = theta + M_PI;
      if (bodyGeom == "Homogeneous disk") 
      {
        (*potential)[i] = pot2DHomogeneous(numElectrodes, r, theta, outerRadiusTCL_.get(), k, sigma0);
      }
      else
      { 
        (*potential)[i] = pot2DTwoConcentric(numElectrodes, r, theta, outerRadiusTCL_.get(), 
                         innerRadiusTCL_.get(), k, sigma0, sigma1);
      }
      i++;
    }
  } 

  //! Sending result
  oportPotentialVector_->send(MatrixHandle(potential));       

}

void
 EITAnalyticSolution::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace BioPSE


