#include <Packages/rtrt/Core/Mesh.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* mesh_maker() {
  return new Mesh();
}
// initialize the static member type_id
PersistentTypeID Mesh::type_id("Mesh", "Persistent", mesh_maker);

Mesh::Mesh (int m, int n)
{
    msize = m;
    nsize = n;
    mesh = new Point *[m];

    init_comb_table();

    for (int i=0; i<m; i++)
    {
        mesh[i] = new Point[n];
    }
}

Mesh::~Mesh()
{
    int i;
    
    for (i=0; i<msize; i++) {
        delete mesh[i];
    }
    delete mesh;
}

Mesh * Mesh::Copy() {
    
    Mesh *m = new Mesh(msize,nsize);
    
    for (int i=0; i<msize; i++) 
        for (int j=0; j<nsize; j++) {
            m->mesh[i][j] = mesh[i][j];
        }
    return m;
}
    
/*    void Mesh::calc_axes(Vector &u, Vector &v, Vector &w) {

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

Point **Mesh::getPts(Vector &u, Vector &v, Vector &w)
{
    Point **P;
    Vector vec;
    
    P = new Point *[msize];
    for (int i=0; i<msize; i++) {
        P[i] = new Point[nsize];
        for (int j=0; j<nsize; j++)
        {
            vec = Vector(mesh[i][j].x(),mesh[i][j].y(),mesh[i][j].z());
            P[i][j] = Point(Dot(vec, u), Dot(vec, v), Dot(vec, w));
        }
    }
    return P;
}

const int MESH_VERSION = 1;
void 
Mesh::io(SCIRun::Piostream &str)
{
  str.begin_class("rtrtMesh", MESH_VERSION);

  Pio(str, msize);
  Pio(str, nsize);
  
  if (str.reading()) {
    mesh = new Point *[msize];
    
    init_comb_table();
    
    for (int i=0; i<msize; i++)  {
      mesh[i] = new Point[nsize];
    }
  }
  for (int i=0; i<msize; i++) 
    for (int j=0; j<nsize; j++) {
      Pio(str, mesh[i][j]);
    }
  
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Mesh*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Mesh::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Mesh*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
