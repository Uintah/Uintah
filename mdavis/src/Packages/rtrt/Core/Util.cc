#include <Packages/rtrt/Core/Util.h>

unsigned int fact_table[NFACT] =
    { 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
      3628800, 39916800, 479001600 };

unsigned int comb_table[NCOMB][NCOMB];
int comb_table_inited=0;
#define ROTATE(a,i,j,k,l) g = a[i][j]; h = a[k][l]; a[i][j] = g-s*(h+g*tau);\
a[k][l] = h+s*(g-h*tau);

void jacobi (double **a, int n, double d[], double **v, int *nrot)
/* Computes all of the eigenvalues and eigenvectors of a real
   symmetric matrix a[1..n][1..n].  On ouput, elements of a above
   the diagonal are destroyed.  d[1..n] returns the eigenvalues of
   a.  v[1..n][1..n] is a matrix whose columns contain, on output,
   the normalized eigenvectors of a.  nrot returns the number of
   Jacobi retotations that were required. */
{
    int j,iq,ip,i;
    double tresh,theta,tau,t,sm,s,h,g,c,*b,*z;

    b = new double[n];
    z = new double[n];
    for (ip=0; ip<n; ip++) {
	for (iq=0; iq<n; iq++)
	    v[ip][iq] = 0.;
	v[ip][ip] = 1.;
    }
    for (ip=0; ip<n; ip++) {
	b[ip] = d[ip] = a[ip][ip];
	z[ip] = 0.;
    }
    *nrot = 0;
    for (i=0; i<50; i++) {
	sm=0.;
	for (ip=0; ip<n-1; ip++)
	    for (iq=ip+1; iq<n; iq++)
		sm += fabs(a[ip][iq]);
	if (sm == 0.) {
	    delete b;
	    delete z;
	    return;
	}
	if (i < 3)
	    tresh = .2*sm/(n*n);
	else
	    tresh = 0.;
	for (ip=0; ip<n-1; ip++) {
	    for (iq=ip+1; iq<n; iq++) {
		g = 100.*fabs(a[ip][iq]);
		/* After four sweeps, skip the rotation if the
		   off-diagonal element is small. */
		if (i > 3 && (float)(fabs(d[ip])+g) == (float)fabs(d[ip])
		    && (float)(fabs(d[iq])+g) == (float)fabs(d[iq]))
		    a[ip][iq]=0.;
		else if (fabs(a[ip][iq]) > tresh) {
		    h = d[iq]-d[ip];
		    if ((float)(fabs(h)+g) == (float)fabs(h))
			t=(a[ip][iq])/h;
		    else {
			theta = .5*h/(a[ip][iq]);
			t = 1./(fabs(theta)+sqrt(1.+theta*theta));
			if (theta<0.) t = -t;
		    }
		    c = 1./sqrt(1+t*t);
		    s = t*c;
		    tau = s/(1.+c);
		    h = t*a[ip][iq];
		    z[ip] -= h;
		    z[iq] += h;
		    d[ip] -= h;
		    d[iq] += h;
		    a[ip][iq]=0.;
		    for (j=0; j<ip-1; j++) {
			ROTATE(a,j,ip,j,iq);
		    }
		    for (j=ip+1; j<iq; j++) {
			ROTATE(a,ip,j,j,iq);
		    }
		    for (j=iq+1; j<n; j++) {
			ROTATE(a,ip,j,iq,j);
		    }
		    for (j=0; j<n; j++) {
			ROTATE(v,j,ip,j,iq);
		    }
		    ++(*nrot);
		}
	    }
	}
	for (ip=0; ip<n; ip++) {
	    b[ip] += z[ip];
	    d[ip] = b[ip];
	    z[ip] = 0.;
	}
    }
    printf("Too many iterations in routine jacobi");
}

int solvecubic(double coef[4], double roots[3])
{
    int i, num;
    double sub;
    double A,B,C;
    double sq_A, p, q;
    double cb_p, D;
    
    /* Dive by the highest order coeffcient */
    
    A = coef[2]/coef[3];
    B = coef[1]/coef[3];
    C = coef[0]/coef[3];
    
    /* substitute x=y-A/3 to eliminate quadric term --
       yielding x^3+px+q=0 */
    
    sq_A = A*A;
    p = 1./3.*(-1./3.*sq_A+B);
    q = 1./2.*(2./27.*A*sq_A-1./3.*A*B+C);
    cb_p = p*p*p;
    D = q*q+cb_p;
    
    if (IsZero(D)) {
	if (IsZero(q)) {
	    roots[0] = 0;
	    num = 1;
	} else {
	    double u = cbrt(-q);
	    roots[0] = 2*u;
	    roots[1] = -u;
	    num = 2;
	}
    } else if (D < 0) {
	double phi = 1./3. * acos(-q/sqrt(-cb_p));
	double t = 2*sqrt(-p);
	
	roots[0] = t * cos(phi);
	roots[1] = -t*cos(phi+M_PI/3.);
	roots[2] = -t*cos(phi-M_PI/3.);
	num = 3;
    } else {
	double sqrt_D = sqrt(D);
	double u = cbrt(sqrt_D - q);
	double v = - cbrt(sqrt_D + q);
	
	roots[0] = u+v;
	num = 1;
    }
    
    sub = 1./3.*A;
    
    for (i=0; i<num; i++)
	roots[i] -= sub;
    
    return num;
}
