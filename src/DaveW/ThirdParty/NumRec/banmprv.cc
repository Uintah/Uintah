#define NRANSI
#include "nrutil.h"
#include "banbks.h"
#include "vector.h"
#include <stdlib.h>
#include <iostream>
#include "matrix.h"

using namespace std;

void banmprv(double **a, double **al, double **au, int n, int m1, int m2, int indx[], double b[], double x[])
{
	int j,i;
	double sdp;
	double *r;
	r=makeVector(n);
	for (i=1;i<=n;i++) {
		sdp = -b[i];
		for (j=1;j<=m1+m2+1;j++) {
		    if (j+i>m1+1 && j+i<m2+n+2) {
			sdp += a[i][j]*x[j+i-m1-1];
//			cerr << i << " " << j << "\n";
		    }
		}
		r[i]=sdp;
	}
	banbks(al, n, m1, m2, au, indx, r);
	for (i=1;i<=n;i++) x[i] -= r[i];
	free(r);
}
#undef NRANSI
