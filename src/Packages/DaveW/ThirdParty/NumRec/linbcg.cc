#include <stdio.h>
#include <math.h>
#define NRANSI
#include "nrutil.h"
#define EPS 1.0e-14

#include <math.h>

double snrm(unsigned long n, double sx[], int itol)
{
        unsigned long i, isamax;
        double ans;

        if (itol <= 3) {
                ans = 0.0;
                for (i=1;i<=n;i++) ans += sx[i]*sx[i];
                return sqrt(ans);
        } else {
                isamax=1;
                for (i=1;i<=n;i++) {
                        if (fabs(sx[i]) > fabs(sx[isamax])) isamax=i;
                }
                return fabs(sx[isamax]);
        }
}

void atimes(int n, double x[], double r[], double **a)
{
    for (int i=1; i<=n; i++) {
	r[i]=0;
	for (int j=1; j<=n; j++) {
	    r[i] +=  a[i][j]*x[j];
	}
    }
}

void asolve(unsigned long n, double b[], double x[], int itrnsp) {
    for (unsigned long i=1; i<=n; i++) x[i]=b[i];
}

void linbcg(unsigned long n, double b[], double x[], int itol, double tol,
	int itmax, int *iter, double *err, double **a)
{
	void asolve(unsigned long n, double b[], double x[], int itrnsp);
	void atimes(int n, double x[], double r[], double **a);
	double snrm(unsigned long n, double sx[], int itol);
	unsigned long j;
	double ak,akden,bk,bkden,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm;
	double *p,*pp,*r,*rr,*z,*zz;

	p=dvector(1,n);
	pp=dvector(1,n);
	r=dvector(1,n);
	rr=dvector(1,n);
	z=dvector(1,n);
	zz=dvector(1,n);

	*iter=0;
	atimes(n,x,r,a);
	for (j=1;j<=n;j++) {
		r[j]=b[j]-r[j];
		rr[j]=r[j];
	}
	znrm=1.0;
	if (itol == 1) bnrm=snrm(n,b,itol);
	else if (itol == 2) {
		asolve(n,b,z,0);
		bnrm=snrm(n,z,itol);
	}
	else if (itol == 3 || itol == 4) {
		asolve(n,b,z,0);
		bnrm=snrm(n,z,itol);
		asolve(n,r,z,0);
		znrm=snrm(n,z,itol);
	} else nrerror("illegal itol in linbcg");
	asolve(n,r,z,0);
	while (*iter <= itmax) {
		++(*iter);
		zm1nrm=znrm;
		asolve(n,rr,zz,1);
		for (bknum=0.0,j=1;j<=n;j++) bknum += z[j]*rr[j];
		if (*iter == 1) {
			for (j=1;j<=n;j++) {
				p[j]=z[j];
				pp[j]=zz[j];
			}
		}
		else {
			bk=bknum/bkden;
			for (j=1;j<=n;j++) {
				p[j]=bk*p[j]+z[j];
				pp[j]=bk*pp[j]+zz[j];
			}
		}
		bkden=bknum;
		atimes(n,p,z,a);
		for (akden=0.0,j=1;j<=n;j++) akden += z[j]*pp[j];
		ak=bknum/akden;
		atimes(n,pp,zz,a);
		for (j=1;j<=n;j++) {
			x[j] += ak*p[j];
			r[j] -= ak*z[j];
			rr[j] -= ak*zz[j];
		}
		asolve(n,r,z,0);
		if (itol == 1 || itol == 2) {
			znrm=1.0;
			*err=snrm(n,r,itol)/bnrm;
		} else if (itol == 3 || itol == 4) {
			znrm=snrm(n,z,itol);
			if (fabs(zm1nrm-znrm) > EPS*znrm) {
				dxnrm=fabs(ak)*snrm(n,p,itol);
				*err=znrm/fabs(zm1nrm-znrm)*dxnrm;
			} else {
				*err=znrm/bnrm;
				continue;
			}
			xnrm=snrm(n,x,itol);
			if (*err <= 0.5*xnrm) *err /= xnrm;
			else {
				*err=znrm/bnrm;
				continue;
			}
		}
		printf("iter=%4d err=%12.6f\n",*iter,*err);
	if (*err <= tol) break;
	}

	free_dvector(p,1,n);
	free_dvector(pp,1,n);
	free_dvector(r,1,n);
	free_dvector(rr,1,n);
	free_dvector(z,1,n);
	free_dvector(zz,1,n);
}
#undef EPS
#undef NRANSI
