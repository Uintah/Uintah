/*
 *  DipoleMatrixSourceRHSQuadratic.cc:
 *
 *  Written by:
 *   vanuiter
 *   TODAY'S DATE HERE
 *
 */

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


#include <Packages/RobV/share/share.h>

namespace RobV {

using namespace SCIRun;

class RobVSHARE DipoleMatrixSourceRHSQuadratic : public Module {
public:
  DipoleMatrixSourceRHSQuadratic(const string& id);

  virtual ~DipoleMatrixSourceRHSQuadratic();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
private:
  FieldIPort               *ifld_;
  FieldIPort              *ifld2_;
  MatrixIPort              *irhs;
  MatrixOPort              *orhs;
  TetVolMesh::Cell::index_type loc;
  int gen;
};

extern "C" RobVSHARE Module* make_DipoleMatrixSourceRHSQuadratic(const string& id) {
  return scinew DipoleMatrixSourceRHSQuadratic(id);
}

DipoleMatrixSourceRHSQuadratic::DipoleMatrixSourceRHSQuadratic(const string& id)
  : Module("DipoleMatrixSourceRHSQuadratic", id, Source, "Quadratic", "RobV")
{
}

DipoleMatrixSourceRHSQuadratic::~DipoleMatrixSourceRHSQuadratic(){
}

void DipoleMatrixSourceRHSQuadratic::execute(){
  

    ifld_ = (FieldIPort *)get_iport("QuadTetVolField");
    FieldHandle mesh;
    ifld2_ = (FieldIPort *)get_iport("DipoleSource");
    FieldHandle mp; //dipoleSource
    MatrixHandle mat_handle2; //inputRHS
    
    ifld_->get(mesh);
    ifld2_->get(mp);
    irhs->get(mat_handle2);
    
    if(!mesh.get_rep()){
      warning("No Data in port 1 field.");
      return;
    } else if (mesh->get_type_name(-1) != "QuadraticTetVolField<int>") {
      error("input must be a TetVol type, not a "+mesh->get_type_name(-1));
      return;
    }

    if(!mp.get_rep()){
      warning("No Data in port 2 field.");
      return;
    }

    QuadraticTetVolField<int>* qtv = dynamic_cast<QuadraticTetVolField<int>*>(mesh.get_rep());
    if (!qtv) {
      error("failed dynamic cast to QuadraticTetVolField<int>*");
      
      return;
    }
    
    QuadraticTetVolMeshHandle mesh_handle;
    QuadraticTetVolMeshHandle qtvm_ = qtv->get_typed_mesh();

    QuadraticTetVolMesh::Node::size_type nnodes;
    qtvm_->size(nnodes);
    
     MatrixHandle rhsh;
     ColumnMatrix* rhs = scinew ColumnMatrix(nnodes);
     rhsh=rhs;
     ColumnMatrix* rhsIn;

     // if the user passed in a vector the right size, copy it into ours

     rhsIn = dynamic_cast<ColumnMatrix*>(mat_handle2.get_rep());
     if (mat_handle2.get_rep() && rhsIn && (rhsIn->nrows()==nnodes))
	 for (int i=0; i<nnodes; i++) (*rhs)[i]=(*rhsIn)[i];
     else
	 rhs->zero();

     LockingHandle<PointCloudField<Vector> > hDipField;
     
     if (mp->get_type_name(0)!="PointCloudField" || mp->get_type_name(1)!="Vector"){
       msgStream_ << "Supplied field is not of type PointCloudField<Vector>. Returning..." << endl;
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

	 msgStream_ << "Source p="<<p<<" dir="<<dir<<" found in elem "<<loc<<endl;
	 
	 cerr << "DipoleMatrixSourceRHS: Found Dipole in element "<<loc<<"\n";

	 double s1, s2, s3, s4,s5,s6,s7,s8,s9,s10;
	 
	 // use these next six lines if we're using a dipole
	 Vector g1, g2, g3, g4,g5,g6,g7,g8,g9,g10;

	 qtvm_->get_gradient_basis(loc,p,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10);
	 
	 //	 Point centroid = mesh->elems[loc]->centroid();
//	 cerr << centroid << "\n";
//	 mesh->get_gradQuad(loc,centroid,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10);
	
	 //	 	 cerr << "DipoleMatrixSourceRHS :: p="<<p<<"  dir="<<dir<<"\n";
	 //	 cerr << "Dir="<<dir<<"  g1="<<g1<<"  g2="<<g2<<"\n";
	 //	 cerr << "g3="<<g3<<"  g4="<<g4<<"  g5="<<g5<<"  g6="<<g6<<"  g7="<<g7<<"  g8="<<g8<<"  g9="<<g9<<"  g10="<<g10<<"\n";
	 //


	 s1=Dot(dir,g1);
	 s2=Dot(dir,g2);
	 s3=Dot(dir,g3);
	 s4=Dot(dir,g4);
	 s5=Dot(dir,g5);
	 s6=Dot(dir,g6);
	 s7=Dot(dir,g7);
	 s8=Dot(dir,g8);
	 s9=Dot(dir,g9);
	 s10=Dot(dir,g10);

	 //	 cerr << s1 << " " << s2 << " " << s3 << " " << s4 << " " << s5 << " " << s6 << " " << s7 << " " << s8 << " " << s9 << " " << s10 << "\n";
	  
	 int i1, i2, i3, i4,i5,i6,i7,i8,i9,i10;

	 TetVolMesh::Node::array_type cell_nodes(10);
	 qtvm_->get_nodes(cell_nodes, loc);
	 i1=cell_nodes[0];
	 i2=cell_nodes[1];
	 i3=cell_nodes[2];
	 i4=cell_nodes[3];
	 i5=cell_nodes[4];
	 i6=cell_nodes[5];
	 i7=cell_nodes[6];
	 i8=cell_nodes[7];
	 i9=cell_nodes[8];
	 i10=cell_nodes[9];	
	 
	 (*rhs)[i1]+=s1;
	 (*rhs)[i2]+=s2;
	 (*rhs)[i3]+=s3;
	 (*rhs)[i4]+=s4;
	 (*rhs)[i5]+=s5;
	 (*rhs)[i6]+=s6;
	 (*rhs)[i7]+=s7;
	 (*rhs)[i8]+=s8;
	 (*rhs)[i9]+=s9;
	 (*rhs)[i10]+=s10;
	
#if 0
	 cerr << "DipoleMatrixSourceRHS :: Here's the RHS vector: ";
	 for (int ii=0; ii<nnodes; ii++) 
	   cerr << (*rhs)[ii]<<" ";
	 cerr << "\n";
	 cerr << "DipoleMatrixSourceRHS :: Here's the dipole: ";
	 for (int ii=0; ii<6; ii++) 
	   cerr << col[ii]<<" ";
	 cerr << "\n";
#endif
 	 
       } else {
	 cerr << "Dipole: "<<p<<" not located within mesh!\n";
       }

       gen=rhsh->generation;
     }
     //     cerr << "DipoleMatrixSourceRHS: about to send result...\n";
     orhs = (MatrixOPort *)get_oport("OutPut RHS");
     orhs->send(rhsh);
     //     cerr << "DipoleMatrixSourceRHS: sent result!\n";


}

void DipoleMatrixSourceRHSQuadratic::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


