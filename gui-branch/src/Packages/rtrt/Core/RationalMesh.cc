#include "RationalMesh.h"

using namespace rtrt;

RationalMesh::RationalMesh (int m, int n)
{
    msize = m;
    nsize = n;
    mesh = new Point4D *[m];

    init_comb_table();


    for (int i=0; i<m; i++)
    {
        mesh[i] = new Point4D[n];
    }
}

RationalMesh::~RationalMesh()
{
    int i;
    
    for (i=0; i<msize; i++) {
        delete mesh[i];
    }
    delete mesh;
}

RationalMesh * RationalMesh::Copy() {
    
    RationalMesh *m = new RationalMesh(msize,nsize);
    
    for (int i=0; i<msize; i++) 
        for (int j=0; j<nsize; j++) {
            m->mesh[i][j] = mesh[i][j];
        }
    return m;
}
    
void RationalMesh::Out(int i, int j) {
    for (int m=0; m<msize-1; m++) {
        for (int n=0; n<nsize-1; n++) {
            printf("Polygon\tpoly%d,%d,%d,%d,%d\n%d\n",i,j,m,n,0,3);
            mesh[m][n].Print();
            mesh[m+1][n].Print();
            mesh[m][n+1].Print();
            printf("poly%d,%d,%d,%d,%d\tsurface\tdefault\n",i,j,m,n,0);
            printf("Polygon\tpoly%d,%d,%d,%d,%d\n%d\n",i,j,m,n,1,3);
            mesh[m][n+1].Print();
            mesh[m+1][n+1].Print();
            mesh[m+1][n].Print();
            printf("poly%d,%d,%d,%d,%d\tsurface\tdefault\n",i,j,m,n,1);
        }
    }
}

void RationalMesh::Print() {
    for (int i=0; i<msize; i++) {
        for (int j=0; j<nsize; j++) {
            mesh[i][j].Print();
        }
    }
}

/*    void RationalMesh::calc_axes(Vector &u, Vector &v, Vector &w) {

	double x,y,z;
	double xsum=0, ysum=0, zsum=0;
	double xsqsum=0, ysqsum=0, zsqsum=0;
	double xysum=0, xzsum=0, yzsum=0;
	double mx, my, mz;
	double normfac = 1./(nsize*msize);
	double **C;
	double eigenvals[3];
	double **eigenvecs;
	int nrot;
	
	for (int i=0; i<msize; i++)
	    for (int j=0; j<nsize; j++) {
		xsum += x = mesh[i][j].coord[0];
		xsqsum += x*x;
		ysum += y = mesh[i][j].coord[1];
		ysqsum += y*y;
		zsum += z = mesh[i][j].coord[2];
		zsqsum += z*z;
		xysum += x*y;
		xzsum += x*z;
		yzsum += y*z;
	    }
	mx = xsum*normfac;
	my = ysum*normfac;
	mz = zsum*normfac;
	C = new double *[3];
	C[0] = new double[3]; C[1] = new double[3]; C[2] = new double[3];
	
	C[0][0] = xsqsum*normfac - mx*mx;
	C[1][1] = ysqsum*normfac - my*my;
	C[2][2] = zsqsum*normfac - mz*mz;
	C[0][1] = C[1][0] = xysum*normfac - mx*my;
	C[0][2] = C[2][0] = xzsum*normfac - mx*mz;
	C[1][2] = C[2][1] = yzsum*normfac - my*mz;

        eigenvecs = new double *[3];
        eigenvecs[0] = new double[3]; eigenvecs[1] = new double[3];
        eigenvecs[2] = new double[3];
	// get the eigenvectors for the C [covariance] matrix
	jacobi(C,3,eigenvals,eigenvecs,&nrot);

	u = Vector(eigenvecs[0][0],eigenvecs[1][0],eigenvecs[2][0]);
	v = Vector(eigenvecs[0][1],eigenvecs[1][1],eigenvecs[2][1]);
	w = Vector(eigenvecs[0][2],eigenvecs[1][2],eigenvecs[2][2]);

	u.normalize();
	v.normalize();
	w.normalize();
    }
*/

Point4D **RationalMesh::getPts(Vector &u, Vector &v, Vector &w)
{
    Point4D **P;
    Vector vec;
    
    P = new Point4D *[msize];
    for (int i=0; i<msize; i++) {
        P[i] = new Point4D[nsize];
        for (int j=0; j<nsize; j++)
        {
            vec = (Vector)(mesh[i][j]);
            P[i][j] = Point4D(vec.dot(u),vec.dot(v),vec.dot(w));
        }
    }
    return P;
}

