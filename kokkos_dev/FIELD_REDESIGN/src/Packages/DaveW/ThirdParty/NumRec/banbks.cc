#define SWAP(a,b) {dum=(a);(a)=(b);(b)=dum;}

void banbks(double **a, int n, int m1, int m2, double **al,
	int indx[], double b[])
{
	int i,k,l;
	int mm;
	double dum;

	mm=m1+m2+1;
	l=m1;
	for (k=1;k<=n;k++) {
		i=indx[k];
		if (i != k) SWAP(b[k],b[i])
		if (l < n) l++;
		for (i=k+1;i<=l;i++) b[i] -= al[k][i-k]*b[k];
	}
	l=1;
	for (i=n;i>=1;i--) {
		dum=b[i];
		for (k=2;k<=l;k++) dum -= a[i][k]*b[k+i-1];
		b[i]=dum/a[i][1];
		if (l < mm) l++;
	}
}
#undef SWAP
