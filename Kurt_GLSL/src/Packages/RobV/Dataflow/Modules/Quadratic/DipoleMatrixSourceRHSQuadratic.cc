/*
 *  DipoleMatrixSourceRHSQuadratic.cc:
 *
 *  Written by:
 *   vanuiter
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/RobV/share/share.h>

namespace RobV {

using namespace SCIRun;

class RobVSHARE DipoleMatrixSourceRHSQuadratic : public Module {
public:
  DipoleMatrixSourceRHSQuadratic(const string& id);

  virtual ~DipoleMatrixSourceRHSQuadratic();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
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
  /*
     MeshHandle mesh;
//     cerr << "DipoleMatrixSourceRHS: about to read inputs...\n";
     if (!inmesh->get(mesh) || !mesh.get_rep()) return;
     MatrixHandle mh;
     Matrix *mp;
     if (!isource->get(mh) || !(mp=mh.get_rep())) return;
//     cerr << "DipoleMatrixSourceRHS: got inputs!\n";
     ColumnMatrixHandle rhsh;
     ColumnMatrix* rhs = scinew ColumnMatrix(mesh->nodes.size());
     rhsh=rhs;
     ColumnMatrixHandle rhshIn;
     ColumnMatrix* rhsIn;
     // if the user passed in a vector the right size, copy it into ours
     if (irhs->get(rhshIn) && (rhsIn=rhshIn.get_rep()) && 
	 (rhsIn->nrows()==mesh->nodes.size()))
	 for (int i=0; i<mesh->nodes.size(); i++) (*rhs)[i]=(*rhsIn)[i];
     else
	 rhs->zero();


     //go over all dipoles
     for (int i=0; i<mp->nrows(); i++) {
       ColumnMatrix col = ColumnMatrix(6);
       col[0] = mp->get(i,0);
       col[1] = mp->get(i,1);
       col[2] = mp->get(i,2);
       col[3] = mp->get(i,3);
       col[4] = mp->get(i,4);
       col[5] = mp->get(i,5);

       Vector dir(col[3], col[4], col[5]);
       Point p(col[0], col[1], col[2]);

       if (mesh->locate(p, loc)) {

	 cerr << "DipoleMatrixSourceRHS: Found Dipole in element "<<loc<<"\n";

	 double s1, s2, s3, s4,s5,s6,s7,s8,s9,s10;
	 
	 // use these next six lines if we're using a dipole
	 Vector g1, g2, g3, g4,g5,g6,g7,g8,g9,g10;

	 mesh->get_gradQuad(mesh->elems[loc],p,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10);
	 
	 //	 Point centroid = mesh->elems[loc]->centroid();
//	 cerr << centroid << "\n";
//	 mesh->get_gradQuad(mesh->elems[loc],centroid,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10);
	
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
	 i1=mesh->elems[loc]->n[0];
	 i2=mesh->elems[loc]->n[1];
	 i3=mesh->elems[loc]->n[2];
	 i4=mesh->elems[loc]->n[3];
	 i5=mesh->elems[loc]->xtrpts[0];
	 i6=mesh->elems[loc]->xtrpts[1];
	 i7=mesh->elems[loc]->xtrpts[2];
	 i8=mesh->elems[loc]->xtrpts[3];
	 i9=mesh->elems[loc]->xtrpts[4];
	 i10=mesh->elems[loc]->xtrpts[5];	
	 
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
	 for (int ii=0; ii<mesh->nodes.size(); ii++) 
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
     orhs->send(rhsh);
     //     cerr << "DipoleMatrixSourceRHS: sent result!\n";
*/

}

void DipoleMatrixSourceRHSQuadratic::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


