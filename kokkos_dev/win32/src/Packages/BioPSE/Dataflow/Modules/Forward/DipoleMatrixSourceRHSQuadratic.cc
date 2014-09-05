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

//    File   : DipoleMatrixSourceRHSQuadratic.cc
//    Author : Robert L. Van Uitert, Martin Cole
//    Date   : Thu Mar 14 19:25:02 2002

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/Point.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Barrier.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>


namespace BioPSE {

using namespace SCIRun;

class DipoleMatrixSourceRHSQuadratic : public Module {
public:
  DipoleMatrixSourceRHSQuadratic(GuiContext *context);

  virtual ~DipoleMatrixSourceRHSQuadratic();

  virtual void execute();

private:
  FieldIPort               *ifld_;
  FieldIPort              *ifld2_;
  MatrixIPort              *irhs;
  MatrixOPort              *orhs;
  TetVolMesh::Cell::index_type loc;
};

DECLARE_MAKER(DipoleMatrixSourceRHSQuadratic)


DipoleMatrixSourceRHSQuadratic::DipoleMatrixSourceRHSQuadratic(GuiContext *gc)
  : Module("DipoleMatrixSourceRHSQuadratic", gc, Source, "Forward", "BioPSE")
{
}

DipoleMatrixSourceRHSQuadratic::~DipoleMatrixSourceRHSQuadratic(){
}

void
DipoleMatrixSourceRHSQuadratic::execute()
{
  ifld_ = (FieldIPort *)get_iport("QuadTetVolField");
  FieldHandle mesh;
  ifld2_ = (FieldIPort *)get_iport("DipoleSource");
  FieldHandle mp; //dipoleSource
  MatrixHandle mat_handle2; //inputRHS
  irhs = (MatrixIPort *)get_iport("Input RHS");
  ifld_->get(mesh);
  ifld2_->get(mp);
  irhs->get(mat_handle2);
    
  if(!mesh.get_rep()){
    error("No Data in port 1 field.");
    return;
  }
  else if (mesh->get_type_name(-1) != "QuadraticTetVolField<int> ")
  {
    error("Input must be a TetVol type, not a '"+mesh->get_type_name(-1)+"'.");
    return;
  }

  if(!mp.get_rep()){
    error("No Data in port 2 field.");
    return;
  }

  QuadraticTetVolField<int>* qtv =
    dynamic_cast<QuadraticTetVolField<int>*>(mesh.get_rep());
  if (!qtv) {
    error("Failed dynamic cast to QuadraticTetVolField<int>.");
      
    return;
  }
    
  QuadraticTetVolMeshHandle mesh_handle;
  QuadraticTetVolMeshHandle qtvm_ = qtv->get_typed_mesh();
  
  qtvm_->synchronize(Mesh::LOCATE_E | Mesh::NODE_NEIGHBORS_E);

  QuadraticTetVolMesh::Node::size_type nnodes;
  qtvm_->size(nnodes);
    
  MatrixHandle rhsh;
  ColumnMatrix* rhs = scinew ColumnMatrix(nnodes);
  rhsh = rhs;
  ColumnMatrix* rhsIn;

  // if the user passed in a vector the right size, copy it into ours
  rhsIn = dynamic_cast<ColumnMatrix*>(mat_handle2.get_rep());
  if (mat_handle2.get_rep() && rhsIn &&
      ((unsigned int)(rhsIn->nrows()) == nnodes))
  {
    for (unsigned int i=0; i<nnodes; i++) (*rhs)[i]=(*rhsIn)[i];
  }
  else
  {
    rhs->zero();
  }

  LockingHandle<PointCloudField<Vector> > hDipField;
     
  if (mp->get_type_name(0)!="PointCloudField" ||
      mp->get_type_name(1)!="Vector")
  {
    error("Supplied field is not of type PointCloudField<Vector>.");
    return;
  }
  else {
    hDipField = dynamic_cast<PointCloudField<Vector>*> (mp.get_rep());
  }

  //go over all dipoles
  PointCloudMesh::Node::iterator ii;
  PointCloudMesh::Node::iterator ii_end;
  hDipField->get_typed_mesh()->begin(ii);
  hDipField->get_typed_mesh()->end(ii_end);
  for (; ii != ii_end; ++ii) {

    Vector dir = hDipField->value(*ii);
    Point p;
    hDipField->get_typed_mesh()->get_point(p, *ii);

    if (qtvm_->locate(loc,p)) {

      msgStream_ << "Source p="<<p<<" dir="<<dir<<
	" found in elem "<<loc<<endl;

      // use these next six lines if we're using a dipole
      vector<Vector> g(10);
      qtvm_->get_gradient_basis(loc,p,g[0],g[1],g[2],g[3],g[4],
				g[5],g[6],g[7],g[8],g[9]);
      vector<int> s(10);
      TetVolMesh::Node::array_type cell_nodes;
      qtvm_->get_nodes(cell_nodes, loc);

      for (int k = 0; k < 10; k++)
	(*rhs)[cell_nodes[k]] += Dot(dir,g[k]);

#if 0
      msgStream_ << "DipoleMatrixSourceRHS :: Here's the RHS vector: ";
      for (int ii=0; ii<nnodes; ii++) 
	msgStream_ << (*rhs)[ii]<<" ";
      msgStream_ << "\n";
      msgStream_ << "DipoleMatrixSourceRHS :: Here's the dipole: ";
      for (int ii=0; ii<6; ii++) 
	msgStream_ << col[ii]<<" ";
      msgStream_ << "\n";
#endif
 	 
    } 
    else 
    {
      msgStream_ << "Dipole: "<<p<<" not located within mesh!\n";
    }
  }

  orhs = (MatrixOPort *)get_oport("OutPut RHS");
  orhs->send_and_dereference(rhsh);
}


} // End namespace BioPSE


