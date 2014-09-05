#include <gfx/std.h>
#include <gfx/math/Mat3.h>
#include <gfx/math/Mat4.h>

// Adapted from VTK source code (see vtkMath.cxx)
// which seems to have been adapted directly from Numerical Recipes in C.


#define ROT(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau);

#define MAX_ROTATIONS 60



// Description:
// Jacobi iteration for the solution of eigenvectors/eigenvalues of a 3x3
// real symmetric matrix. Square 3x3 matrix a; output eigenvalues in w;
// and output eigenvectors in v. Resulting eigenvalues/vectors are sorted
// in decreasing order; eigenvectors are normalized.
//
static
bool internal_jacobi(real a[3][3], real w[3], real v[3][3])
{
    int i, j, k, iq, ip;
    real tresh, theta, tau, t, sm, s, h, g, c;
    real b[3], z[3], tmp;

    // initialize
    for (ip=0; ip<3; ip++) 
    {
	for (iq=0; iq<3; iq++) v[ip][iq] = 0.0;
	v[ip][ip] = 1.0;
    }
    for (ip=0; ip<3; ip++) 
    {
	b[ip] = w[ip] = a[ip][ip];
	z[ip] = 0.0;
    }

    // begin rotation sequence
    for (i=0; i<MAX_ROTATIONS; i++) 
    {
	sm = 0.0;
	for (ip=0; ip<2; ip++) 
	{
	    for (iq=ip+1; iq<3; iq++) sm += fabs(a[ip][iq]);
	}
	if (sm == 0.0) break;

	if (i < 4) tresh = 0.2*sm/(9);
	else tresh = 0.0;

	for (ip=0; ip<2; ip++) 
	{
	    for (iq=ip+1; iq<3; iq++) 
	    {
		g = 100.0*fabs(a[ip][iq]);
		if (i > 4 && (fabs(w[ip])+g) == fabs(w[ip])
		    && (fabs(w[iq])+g) == fabs(w[iq]))
		{
		    a[ip][iq] = 0.0;
		}
		else if (fabs(a[ip][iq]) > tresh) 
		{
		    h = w[iq] - w[ip];
		    if ( (fabs(h)+g) == fabs(h)) t = (a[ip][iq]) / h;
		    else 
		    {
			theta = 0.5*h / (a[ip][iq]);
			t = 1.0 / (fabs(theta)+sqrt(1.0+theta*theta));
			if (theta < 0.0) t = -t;
		    }
		    c = 1.0 / sqrt(1+t*t);
		    s = t*c;
		    tau = s/(1.0+c);
		    h = t*a[ip][iq];
		    z[ip] -= h;
		    z[iq] += h;
		    w[ip] -= h;
		    w[iq] += h;
		    a[ip][iq]=0.0;
		    for (j=0;j<ip-1;j++) 
		    {
			ROT(a,j,ip,j,iq)
			    }
		    for (j=ip+1;j<iq-1;j++) 
		    {
			ROT(a,ip,j,j,iq)
			    }
		    for (j=iq+1; j<3; j++) 
		    {
			ROT(a,ip,j,iq,j)
			    }
		    for (j=0; j<3; j++) 
		    {
			ROT(v,j,ip,j,iq)
			    }
		}
	    }
	}

	for (ip=0; ip<3; ip++) 
	{
	    b[ip] += z[ip];
	    w[ip] = b[ip];
	    z[ip] = 0.0;
	}
    }

    if ( i >= MAX_ROTATIONS )
    {
	cerr << "WARNING -- jacobi() -- Error computing eigenvalues." << endl;
	return false;
    }

    // sort eigenfunctions
    for (j=0; j<3; j++) 
    {
	k = j;
	tmp = w[k];
	for (i=j; i<3; i++)
	{
	    if (w[i] >= tmp) 
	    {
		k = i;
		tmp = w[k];
	    }
	}
	if (k != j) 
	{
	    w[k] = w[j];
	    w[j] = tmp;
	    for (i=0; i<3; i++) 
	    {
		tmp = v[i][j];
		v[i][j] = v[i][k];
		v[i][k] = tmp;
	    }
	}
    }
    // insure eigenvector consistency (i.e., Jacobi can compute vectors that
    // are negative of one another (.707,.707,0) and (-.707,-.707,0). This can
    // reek havoc in hyperstreamline/other stuff. We will select the most
    // positive eigenvector.
    int numPos;
    for (j=0; j<3; j++)
    {
	for (numPos=0, i=0; i<3; i++) if ( v[i][j] >= 0.0 ) numPos++;
	if ( numPos < 2 ) for(i=0; i<3; i++) v[i][j] *= -1.0;
    }

    return true;
}

static
bool internal_jacobi4(real a[4][4], real w[4], real v[4][4])
{
    int i, j, k, iq, ip;
    real tresh, theta, tau, t, sm, s, h, g, c;
    real b[4], z[4], tmp;

    // initialize
    for (ip=0; ip<4; ip++) 
    {
	for (iq=0; iq<4; iq++) v[ip][iq] = 0.0;
	v[ip][ip] = 1.0;
    }
    for (ip=0; ip<4; ip++) 
    {
	b[ip] = w[ip] = a[ip][ip];
	z[ip] = 0.0;
    }

    // begin rotation sequence
    for (i=0; i<MAX_ROTATIONS; i++) 
    {
	sm = 0.0;
	for (ip=0; ip<3; ip++) 
	{
	    for (iq=ip+1; iq<4; iq++) sm += fabs(a[ip][iq]);
	}
	if (sm == 0.0) break;

	if (i < 4) tresh = 0.2*sm/(16);
	else tresh = 0.0;

	for (ip=0; ip<3; ip++) 
	{
	    for (iq=ip+1; iq<4; iq++) 
	    {
		g = 100.0*fabs(a[ip][iq]);
		if (i > 4 && (fabs(w[ip])+g) == fabs(w[ip])
		    && (fabs(w[iq])+g) == fabs(w[iq]))
		{
		    a[ip][iq] = 0.0;
		}
		else if (fabs(a[ip][iq]) > tresh) 
		{
		    h = w[iq] - w[ip];
		    if ( (fabs(h)+g) == fabs(h)) t = (a[ip][iq]) / h;
		    else 
		    {
			theta = 0.5*h / (a[ip][iq]);
			t = 1.0 / (fabs(theta)+sqrt(1.0+theta*theta));
			if (theta < 0.0) t = -t;
		    }
		    c = 1.0 / sqrt(1+t*t);
		    s = t*c;
		    tau = s/(1.0+c);
		    h = t*a[ip][iq];
		    z[ip] -= h;
		    z[iq] += h;
		    w[ip] -= h;
		    w[iq] += h;
		    a[ip][iq]=0.0;
		    for (j=0;j<ip-1;j++) 
		    {
			ROT(a,j,ip,j,iq)
			    }
		    for (j=ip+1;j<iq-1;j++) 
		    {
			ROT(a,ip,j,j,iq)
			    }
		    for (j=iq+1; j<4; j++) 
		    {
			ROT(a,ip,j,iq,j)
			    }
		    for (j=0; j<4; j++) 
		    {
			ROT(v,j,ip,j,iq)
			    }
		}
	    }
	}

	for (ip=0; ip<4; ip++) 
	{
	    b[ip] += z[ip];
	    w[ip] = b[ip];
	    z[ip] = 0.0;
	}
    }

    if ( i >= MAX_ROTATIONS )
    {
	cerr << "WARNING -- jacobi() -- Error computing eigenvalues." << endl;
	return false;
    }

    // sort eigenfunctions
    for (j=0; j<4; j++) 
    {
	k = j;
	tmp = w[k];
	for (i=j; i<4; i++)
	{
	    if (w[i] >= tmp) 
	    {
		k = i;
		tmp = w[k];
	    }
	}
	if (k != j) 
	{
	    w[k] = w[j];
	    w[j] = tmp;
	    for (i=0; i<4; i++) 
	    {
		tmp = v[i][j];
		v[i][j] = v[i][k];
		v[i][k] = tmp;
	    }
	}
    }
    // insure eigenvector consistency (i.e., Jacobi can compute vectors that
    // are negative of one another (.707,.707,0) and (-.707,-.707,0). This can
    // reek havoc in hyperstreamline/other stuff. We will select the most
    // positive eigenvector.
    int numPos;
    for (j=0; j<4; j++)
    {
	for (numPos=0, i=0; i<4; i++) if ( v[i][j] >= 0.0 ) numPos++;
	if ( numPos < 3 ) for(i=0; i<4; i++) v[i][j] *= -1.0;
    }

    return true;
}


#undef ROT
#undef MAX_ROTATIONS


bool jacobi(const Mat3& m, Vec3& eig_vals, Vec3 eig_vecs[3])
{
    real a[3][3], w[3], v[3][3];
    int i,j;

    for(i=0;i<3;i++) for(j=0;j<3;j++) a[i][j] = m(i,j);

    bool result = internal_jacobi(a, w, v);
    if( result )
    {
	for(i=0;i<3;i++) eig_vals[i] = w[i];

	for(i=0;i<3;i++) for(j=0;j<3;j++) eig_vecs[i][j] = v[j][i];
    }

    return result;
}

bool jacobi(const Mat4& m, Vec4& eig_vals, Vec4 eig_vecs[4])
{
    real a[4][4], w[4], v[4][4];
    int i,j;

    for(i=0;i<4;i++) for(j=0;j<4;j++) a[i][j] = m(i,j);

    bool result = internal_jacobi4(a, w, v);
    if( result )
    {
	for(i=0;i<4;i++) eig_vals[i] = w[i];

	for(i=0;i<4;i++) for(j=0;j<4;j++) eig_vecs[i][j] = v[j][i];
    }

    return result;
}
