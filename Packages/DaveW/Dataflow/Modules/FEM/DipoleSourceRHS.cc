/*
 *  DipoleSourceRHS.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;

class DipoleSourceRHS : public Module {
    MeshIPort* inmesh;
    ColumnMatrixIPort * isource;
    MatrixIPort * isrcmat;
    ColumnMatrixIPort* irhs;
    ColumnMatrixOPort* orhs;
    ColumnMatrixOPort* oidx;
    int loc;
    int gen;
public:
    DipoleSourceRHS(const clString& id);
    virtual ~DipoleSourceRHS();
    virtual void execute();
};

extern "C" Module* make_DipoleSourceRHS(const clString& id)
{
    return scinew DipoleSourceRHS(id);
}

DipoleSourceRHS::DipoleSourceRHS(const clString& id)
: Module("DipoleSourceRHS", id, Filter)
{
    // Create the input port
    inmesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inmesh);
    isource=scinew ColumnMatrixIPort(this, "Source", ColumnMatrixIPort::Atomic);
    add_iport(isource);
    irhs=scinew ColumnMatrixIPort(this, "Input RHS",ColumnMatrixIPort::Atomic);
    add_iport(irhs);
    isrcmat=scinew MatrixIPort(this, "SourceMat", MatrixIPort::Atomic);
    add_iport(isrcmat);

    // Create the output ports
    orhs=scinew ColumnMatrixOPort(this,"Output RHS",ColumnMatrixIPort::Atomic);
    add_oport(orhs);
    oidx=scinew ColumnMatrixOPort(this,"Elem Index",ColumnMatrixIPort::Atomic);
    add_oport(oidx);
}

DipoleSourceRHS::~DipoleSourceRHS()
{
}

void DipoleSourceRHS::execute()
{
     MeshHandle mesh;
//     cerr << "DipoleSourceRHS: about to read inputs...\n";
     if (!inmesh->get(mesh) || !mesh.get_rep()) return;

     ColumnMatrixHandle mh=0;
     ColumnMatrix *mp=0;

     MatrixHandle mmh=0;
     Matrix* mmp=0;
     
     if ((!isource->get(mh) || !(mp=mh.get_rep())) &&
	 (!isrcmat->get(mmh) || !(mmp=mmh.get_rep()))) return; 
     
     if ((mp && mp->nrows()<6) || (mmp && mmp->nrows()<6)) {
       cerr << "Error - dipole source must have at least six rows\n";
       return;
     }

//     cerr << "DipoleSourceRHS: got inputs!\n";
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

     Vector dir;
     Point p;
     int nsrcs=1;
     if (mmp) nsrcs=mmp->nrows();
     
     for (int ii=0; ii<nsrcs; ii++) {
       if (mp) {
	 dir=Vector((*mp)[3], (*mp)[4], (*mp)[5]);
	 p=Point((*mp)[0], (*mp)[1], (*mp)[2]);
	 if (mp->nrows() == 7) loc=(int)((*mp)[6]);
       } else {
	 dir=Vector((*mmp)[ii][3], (*mmp)[ii][4], (*mmp)[ii][5]);
	 p=Point((*mmp)[ii][0], (*mmp)[ii][1], (*mmp)[ii][2]);
	 if (mmp->nrows() == 7) loc=(int)((*mmp)[ii][6]);
       }
       if (mesh->locate(p, loc)) {
//	 cerr << "DipoleSourceRHS: Found Dipole in element "<<loc<<"\n";
	 double s1, s2, s3, s4;

	 // use these next six lines if we're using a dipole
	 Vector g1, g2, g3, g4;
	 mesh->get_grad(mesh->elems[loc], p, g1, g2, g3, g4);

//	 cerr << "DipoleSourceRHS :: p="<<p<<"  dir="<<dir<<"\n";
//	 cerr << "Dir="<<dir<<"  g1="<<g1<<"  g2="<<g2<<"\n";
//	 cerr << "g3="<<g3<<"  g4="<<g4<<"\n";
	 s1=Dot(g1,dir);
	 s2=Dot(g2,dir);
	 s3=Dot(g3,dir);
	 s4=Dot(g4,dir);
	     
	 // use this next line if we're using a monopole
	 // mesh->get_interp(mesh->elems[loc], p, s1, s2, s3, s4);

	 // s1*=src; s2*=src; s3*=src; s4*=src;
	 int i1, i2, i3, i4;
	 i1=mesh->elems[loc]->n[0];
	 i2=mesh->elems[loc]->n[1];
	 i3=mesh->elems[loc]->n[2];
	 i4=mesh->elems[loc]->n[3];
	 (*rhs)[i1]+=s1;
	 (*rhs)[i2]+=s2;
	 (*rhs)[i3]+=s3;
	 (*rhs)[i4]+=s4;

#if 0
	 cerr << "DipoleSourceRHS :: Here's the RHS vector: ";
	 for (int jj=0; jj<mesh->nodes.size(); jj++) 
	     cerr << (*rhs)[jj]<<" ";
	 cerr << "\n";
	 cerr << "DipoleSourceRHS :: Here's the dipole: ";
	 if (mmp)
	   for (jj=0; jj<5; jj++) 
	     cerr << (*mmp)[ii][jj]<<" ";
	 else
	   for (jj=0; jj<5; jj++) 
	     cerr << (*mp)[jj]<<" ";
	 cerr << "\n";
#endif

     } else {
	 loc=0;
	 dir=Vector(0,0,0);
	 cerr << "Dipole: "<<p<<" not located within mesh!\n";
     }
     gen=rhsh->generation;
//     cerr << "DipoleSourceRHS: about to send result...\n";
     orhs->send(rhsh);
//     cerr << "DipoleSourceRHS: sent result!\n";
     
     ColumnMatrix *idxvec = new ColumnMatrix(6);
     idxvec->zero();
     (*idxvec)[0]=loc*3;
     (*idxvec)[1]=dir.x();
     (*idxvec)[2]=loc*3+1;
     (*idxvec)[3]=dir.y();
     (*idxvec)[4]=loc*3+2;
     (*idxvec)[5]=dir.z();
     oidx->send(ColumnMatrixHandle(idxvec));
     }
}

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.7  2000/10/17 15:38:37  dmw
// Can now take a matrix of dipoles
//
// Revision 1.5  2000/03/17 09:25:43  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  1999/10/07 02:06:34  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/16 00:36:55  dmw
// added new Module that Chris Butson will work on (DipoleInSphere) and fixed SRCDIR references in DaveW makefiles
//
// Revision 1.2  1999/09/08 02:26:27  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/02 04:49:24  dmw
// more of Dave's modules
//
//
