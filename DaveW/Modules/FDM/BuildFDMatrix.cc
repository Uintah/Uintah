//static char *id="@(#) $Id$";

/*
 *  BuildFDMatrix.cc:  Builds the global finite difference matrix
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <DaveW/Datatypes/General/SigmaSetPort.h>
#include <DaveW/Datatypes/General/SigmaSet.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Datatypes/SparseRowMatrix.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Util/NotFinished.h>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using SCICore::Math::Max;

class BuildFDMatrix : public Module {
    ScalarFieldIPort* sfiport;
    SigmaSetIPort* ssiport;
    SurfaceIPort* surfiport;
    double dx;
    double dy;
    double dz;
    ColumnMatrixOPort* rhsoport;
    MatrixOPort* moport;

    void build_local_matrix(const Array2<double>& sigs,
			    ScalarFieldRGchar* sf,
			    Array1<double> &lcl,
			    int i, int j, int k,
			    const Array1<char>& nmn);
    void add_lcl_gbl(Matrix& gbl_a, const Array1<double>& lcl,
		     int i, int j, int k, int ny, int nz,
		     HashTable<int, double>& hash, ColumnMatrix& rhs);
public:
    BuildFDMatrix(const clString& id);
    virtual ~BuildFDMatrix();
    virtual void execute();
};


Module* make_BuildFDMatrix(const clString& id)
{
    return scinew BuildFDMatrix(id);
}

BuildFDMatrix::BuildFDMatrix(const clString& id)
: Module("BuildFDMatrix", id, Filter)
{
    // Create the input ports
    sfiport = scinew ScalarFieldIPort(this, "SF", ScalarFieldIPort::Atomic);
    add_iport(sfiport);
    ssiport = scinew SigmaSetIPort(this, "Sigmas", SigmaSetIPort::Atomic);
    add_iport(ssiport);
    surfiport = scinew SurfaceIPort(this, "Surfs", SurfaceIPort::Atomic);
    add_iport(surfiport);

    // Create the output ports
    moport = scinew MatrixOPort(this, "FDM Matrix", MatrixIPort::Atomic);
    add_oport(moport);
    rhsoport = scinew ColumnMatrixOPort(this,"RHS", ColumnMatrixIPort::Atomic);
    add_oport(rhsoport);
}

BuildFDMatrix::~BuildFDMatrix()
{
}

void BuildFDMatrix::execute()
{
//    cerr << "BuildFDMatrix - grabbing data from ports.\n";
    ScalarFieldHandle sfh;
    if(!sfiport->get(sfh)) { 
//	cerr << "No sfH.\n"; 
	return; 
    }
    ScalarFieldRGBase *sfrgb;
    if (!(sfrgb = sfh->getRGBase())) { 
	cerr << "Not an rg base.\n"; 
	return; 
    } 
    ScalarFieldRGchar *sf;
    if (!(sf = sfrgb->getRGChar())) { 
	cerr << "Not a char sf.\n"; 
	return; 
    }
    SigmaSetHandle ssh;
    if (!ssiport->get(ssh)) { 
//	cerr << "No sigma set.\n"; 
	return; 
    }
    if (ssh->vals.dim2() != 6) { 
	cerr << "SigmaSet must be nx6.\n"; 
	return; 
    }
    Array2<double> *sigs=&(ssh->vals);
    SurfaceHandle surfh;
    if (!surfiport->get(surfh)) { 
//	cerr << "No surf.\n"; 
	return; 
    }
    
    int i,j,k;
    int num_nmn, num_nonnmn;
    num_nmn=num_nonnmn=0;
    Array1<char> nmn(sigs->dim1());	// is this material a Neumann b.c.?
    for (i=0; i<sigs->dim1(); i++) {
	int nm=1;
	for (j=0; j<sigs->dim2(); j++) {
	    if ((*sigs)(i,j) != 0) nm=0;
	}
	nmn[i]=nm;
    }
    for (i=0; i<sf->nx; i++)
	for (j=0; j<sf->ny; j++)
	    for (k=0; k<sf->nz; k++)
		sf->grid(i,j,k)-='0';

    int nnodes=sf->nx*sf->ny*sf->nz;
    int ndof=nnodes;
    int r=0;
    long curr=0;
    Array1<int> rows(ndof+1);
    Array1<int> cols;

    int nl, nr, nu, nd, nb, nf;
    nl=nr=nu=nd=nb=nf=0;

    // Figure out size for matrix
//    cerr << "Non-zero rows: ";
    for (i=0; i<sf->nx; i++) {
	for (j=0; j<sf->ny; j++) {
	    for (k=0; k<sf->nz; k++, curr++) {
		if(curr%500 == 0)
		    update_progress(curr, 2*nnodes);
		rows[r++]=cols.size();
// if (r>1 && (rows[r-1] != rows[r-2]+1)) cerr << r-1 <<" ";		
		if (!nmn[sf->grid(i,j,k)]) { 		// not a Neumann voxel
		    num_nonnmn++;
		    if ((i!=0) && (!nmn[sf->grid(i-1,j,k)])){        // 011111
			cols.add(curr-sf->ny*sf->nz); nl++;}
		    if ((j!=0) && (!nmn[sf->grid(i,j-1,k)])){        // 110111
			cols.add(curr-sf->nz); nd++;}
		    if ((k!=0) && (!nmn[sf->grid(i,j,k-1)])){        // 111101
			cols.add(curr-1); nb++;}
		    cols.add(curr);
		    if ((k!=sf->nz-1) && (!nmn[sf->grid(i,j,k+1)])){ // 111110
			cols.add(curr+1); nf++;}
		    if ((j!=sf->ny-1) && (!nmn[sf->grid(i,j+1,k)])){ // 111011
			cols.add(curr+sf->nz); nu++;}
		    if ((i!=sf->nx-1) && (!nmn[sf->grid(i+1,j,k)])){ // 101111
			cols.add(curr+sf->ny*sf->nz); nr++;}
		} else {
		    cols.add(curr);
		    num_nmn++;
		}
	    }
	}
    }
    cerr << "\n";
    rows[r]=cols.size();
cerr << "Number of non-conducting voxels: "<<num_nmn;
cerr << "\nNumber of conducting voxels: "<<num_nonnmn;
cerr << "\nNumber of left, right, up, down, fwd and bk nbrs: "<<nl<<" "<<nr<<" "<<nu<<" "<<nd<<" "<<nf<<" "<<nb<<"\n";
    HashTable<int, double> hash;
    Array1<NodeHandle> nodes;
    surfh->get_surfnodes(nodes);
    for (i=0; i<nodes.size(); i++) {
	int x, y, z;
	sf->locate(nodes[i]->p, x, y, z);
cerr << "point is at: ("<<nodes[i]->p.x()<<", "<<nodes[i]->p.y()<<", "<<nodes[i]->p.z()<<")\n";
	int row=z+(sf->nz*(y+x*sf->ny));
cerr << "surf node at field point: ("<<x<<", "<<y<<", "<<z<<"), row: "<<row<<"\n";
	hash.insert(row, nodes[i]->bc->value);
    }

    Matrix* gbl_matrix=scinew SparseRowMatrix(ndof, ndof, rows, cols);
//    Matrix* gbl_matrix=scinew DenseMatrix(ndof, ndof);
    gbl_matrix->zero();
    ColumnMatrix* rhs=scinew ColumnMatrix(ndof);
    rhs->zero();
    curr=0;
    Point min;
    Point max;
    sf->get_bounds(min, max);
    Vector v(max-min);
    dx=v.x()/(sf->nx-1);
    dy=v.y()/(sf->ny-1);
    dz=v.z()/(sf->nz-1);
    double m=Max(dx,dy,dz);
    dx/=m;
    dy/=m;
    dz/=m;

    for (i=0; i<sf->nx; i++) {
	for (j=0; j<sf->ny; j++) {
	    for (k=0; k<sf->nz; k++, curr++) {
		if(curr%500 == 0)
		    update_progress(nnodes+curr, 2*nnodes);

		// Build local matrix
		Array1<double> lcl(27);
		for (int ii=0; ii<27; ii++) lcl[ii]=0;
		double xx;

		// if it's Dirichlet, just set the diagonal term
		if (!hash.lookup(curr, xx) && !nmn[sf->grid(i,j,k)]) {
		    build_local_matrix(*sigs, sf, lcl, i, j, k, nmn);
		} else {
		    if (hash.lookup(curr,xx)) cerr << "Dirichlet: "<<curr<<"\n";
		    lcl[13]=1;
		}
		add_lcl_gbl(*gbl_matrix, lcl, i, j, k, sf->ny, sf->nz,
			    hash, *rhs);

	    }
	}
    }
    moport->send(MatrixHandle(gbl_matrix));
    rhsoport->send(ColumnMatrixHandle(rhs));
}

void BuildFDMatrix::build_local_matrix(const Array2<double>& sigs,
				       ScalarFieldRGchar* sf,
				       Array1<double> &lcl,
				       int i, int j, int k, 
				       const Array1<char>& nmn) {

    int currSig=(int) (sf->grid(i,j,k));
    double A = sigs(currSig,0)/(dx*dx);
    double B = sigs(currSig,3)/(dy*dy);
    double C = sigs(currSig,5)/(dz*dz);
    double D = 2*sigs(currSig,1)/(dx*dy);
    double E = 2*sigs(currSig,4)/(dy*dz);
    double F = 2*sigs(currSig,2)/(dx*dz);

    lcl[1] = D; lcl[3] = F; lcl[4] = A; lcl[5] =-F; lcl[7] =-D;
    lcl[19]= D; lcl[21]= F; lcl[22]= A; lcl[23]=-F; lcl[25]=-D;
    lcl[9] = E; lcl[10]= B; lcl[11]=-E; lcl[12]= C;
    lcl[13]=-2*(A+B+C);
    lcl[14]= C; lcl[15]=-E; lcl[16]= B; lcl[17]= E;

    char empty=0;
    if ((i==0) || nmn[sf->grid(i-1,j,k)])        		// 1-----
	empty |= 32;
    if ((i==sf->nx-1) || nmn[sf->grid(i+1,j,k)]) 		// -1----
	empty |= 16;
    if ((j==0) || nmn[sf->grid(i,j-1,k)])        		// --1---
	empty |= 8;
    if ((j==sf->ny-1) || nmn[sf->grid(i,j+1,k)]) 		// ---1--
	empty |= 4;
    if ((k==0) || nmn[sf->grid(i,j,k-1)])        		// ----1- 
	empty |= 2;
    if ((k==sf->nz-1) || nmn[sf->grid(i,j,k+1)]) 		// -----1 
	empty |= 1;

    if ((empty & 51)==51) lcl[3]=lcl[5]=lcl[21]=lcl[23]=0;	// 11--11
    if ((empty & 60)==60) lcl[1]=lcl[7]=lcl[19]=lcl[25]=0;      // 1111--
    if ((empty & 15)==15) lcl[9]=lcl[11]=lcl[15]=lcl[17]=0;	// --1111
    if ((empty & 48)==48) {lcl[13]-=2*A; lcl[4]=lcl[22]=0; }    // 11----
    if ((empty & 12)==12) {lcl[13]-=2*B; lcl[10]=lcl[16]=0;}    // --11--
    if ((empty & 3)==3)  {lcl[13]-=2*C; lcl[12]=lcl[14]=0;}     // ----11
    if ((empty & 63)==63) lcl[13]=1; 				// 111111
    if (empty & 32) {lcl[22]+=lcl[4]; lcl[4]=0; } 		// 1-----
    if (empty & 16) {lcl[4]+=lcl[22]; lcl[22]=0; } 		// -1----
    if (empty & 8)  {lcl[16]+=lcl[10]; lcl[10]=0; } 		// --1---
    if (empty & 4)  {lcl[10]+=lcl[16]; lcl[16]=0; } 		// ---1--
    if (empty & 2)  {lcl[14]+=lcl[12]; lcl[12]=0; } 		// ----1-
    if (empty & 1)  {lcl[12]+=lcl[14]; lcl[14]=0; } 		// -----1

#if 0
    cerr << i << "," << j << "," << k << "   ";
    for (int y=2; y>=0; y--) {
	for (int x=0; x<3; x++) {
	    for (int z=0; z<3; z++) {
		cerr << lcl[x*9+y*3+z] << " ";
	    }
    cerr << "       ";
	}
	if (y==2)
	    cerr << "\n" << (empty&32)/32 << (empty&16)/16 << (empty&8)/8 << (empty&4)/4 << (empty&2)/2 << (empty&1)/1 <<"  ";
	else       
	    cerr << "\n        ";
   }
    cerr << "\n";
#endif
}

void BuildFDMatrix::add_lcl_gbl(Matrix& gbl_a, 
				const Array1<double>& lcl,
				int i, int j, int k, int ny, int nz,
				HashTable<int, double>& hash, 
				ColumnMatrix& rhs)
{
    int ii, jj, kk, ci, cj, ck;
    long idx=k+nz*(j+i*ny);
    for (ii=0, ci=i-1; ii<3; ii++, ci++)
	for (jj=0, cj=j-1; jj<3; jj++, cj++)
	    for (kk=0, ck=k-1; kk<3; kk++, ck++) {
		long c_idx=idx+(nz*ny*(ii-1))+(nz*(jj-1))+kk-1;
		if (Abs(lcl[ii*9+jj*3+kk])>.0000000001) {
		    double val;	
//cerr << "node("<<idx<<") - lcl["<<ii*9+jj*3+kk<<"],global["<<c_idx<<"] = "<<lcl[ii*9+jj*3+kk]<<"   ";
		    if (hash.lookup(c_idx, val)) {	// Dirichlet node
			cerr << "Dirichlet: "<<c_idx<<"\n";
			if (ii!=1 || jj!=1 || kk!=1) {
			    rhs[idx]-=val*lcl[ii*9+jj*3+kk];
//			    lcl[ii*9+jj*3+kk]=0;
			} else {
			    rhs[idx]=val;
			}
//cerr << "RHS["<<idx<<"]="<<rhs[idx]<<"\n";
		    }
		    gbl_a[idx][c_idx] += lcl[ii*9+jj*3+kk];
//cerr << "gbl_a["<<idx<<"]["<<c_idx<<"] = "<<gbl_a[idx][c_idx]<<"\n";
		}
	    }
//    cerr << "\n";
}

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/08/24 06:23:05  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:16  dmw
// Added and updated DaveW Datatypes/Modules
//
// Revision 1.1  1999/04/27 23:44:08  dav
// moved FDM to DaveW
//
// Revision 1.2  1999/04/27 22:57:47  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
