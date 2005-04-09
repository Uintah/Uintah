
#include <Classlib/Array1.h>
#include <Classlib/Timer.h>
#include <Datatypes/SymSparseRowMatrix.h>
#include <Datatypes/ColumnMatrix.h>
#include <Math/Expon.h>
#include <iostream.h>
#include <stdlib.h>
#include <string.h>

void usage(char* progname)
{
    cerr << progname << " vnorm|vvmult|vvdot|vvsmadd|ssmmult r n [k]\n";
    exit(-1);

}

int main(int argc, char** argv)
{
    if(argc<4){
	usage(argv[0]);
    }
    int r=atoi(argv[2]);
    int n=atoi(argv[3]);
    CPUTimer timer;
    int flops=0;
    int size=0;
    if(!strcmp(argv[1], "vnorm")){
	ColumnMatrix c1(n);
	size+=n*sizeof(double);
	double v1=Sqrt((double)n);
	for(int i=0;i<n;i++){
	    c1[i]=v1;
	}
	timer.start();
	double norm=0;
	for(int ii=0;ii<r;ii++){
	    norm=c1.vector_norm();
	}
	timer.stop();
	flops=r*2*n;
 	cout << "norm=" << norm << endl;
    } else if(!strcmp(argv[1], "vvmult")){
	ColumnMatrix res(n);
	ColumnMatrix a(n);
	ColumnMatrix b(n);
	double v1=Sqrt((double)n);
	double v2=Sqrt(v1);
	int i;
	for(i=0;i<n;i+=2){
	    a[i]=v1;
	    b[i]=1;
	}
	for(i=1;i<n;i+=2){
	    a[i]=v2;
	    b[i]=v2;
	}

	timer.start();
	for(int ii=0;ii<r;ii++){
	    Mult(res, a, b);
	}
	timer.stop();
	flops=r*1*n;
	size+=n*sizeof(double)*3;
	
	double norm=res.vector_norm();
 	cout << "norm=" << norm << endl;
    } else if(!strcmp(argv[1], "vvdot")){
	ColumnMatrix a(n);
	ColumnMatrix b(n);
	double v1=n;
	double v2=1/v1;
	int i;
	for(i=0;i<n;i+=2){
	    a[i]=1;
	    b[i]=1;
	}
	for(i=1;i<n;i+=2){
	    a[i]=v1;
	    b[i]=v2;
	}

	double norm=0;
	timer.start();
	for(int ii=0;ii<r;ii++){
	    norm=Dot(a, b);
	}
	timer.stop();
	flops=r*2*n;
	size+=n*sizeof(double)*2;

 	cout << "norm=" << norm << endl;
    } else if(!strcmp(argv[1], "vvsmadd")){
	ColumnMatrix res(n);
	ColumnMatrix a(n);
	ColumnMatrix b(n);
	double v1=1./2.;
	double v2=Sqrt((double)n)/2.;
	double s=Sqrt((double)n);
	for(int i=0;i<n;i++){
	    a[i]=v1;
	    b[i]=v2;
	}

	timer.start();
	for(int ii=0;ii<r;ii++){
	    ScMult_Add(res, s, a, b);
	}
	timer.stop();
	flops=r*2*n;
	size+=n*sizeof(double)*3;
	
	double norm=res.vector_norm();
 	cout << "norm=" << norm << endl;
    } else if(!strcmp(argv[1], "ssmmult")){
	if(argc<4)
	    usage(argv[0]);
	int k=atoi(argv[4]);
	if(k>n)k=n;

	Array1<int> rows(n+1);
	Array1<int> cols(n*k);
	int i;
	for(i=0;i<=n;i++){
	    rows[i]=i*k;
	}

	for(i=0;i<n;i++){
	    for(int j=0;j<k;j++){
		cols[i*k+j]=j;
	    }
	}

	SymSparseRowMatrix A(n, n, rows, cols);
	ColumnMatrix x(n);
	ColumnMatrix b(n);

	double v1=Sqrt((double)n);
	double v2=1./(double)k;

	for(i=0;i<n;i++){
	    x[i]=v2;
	    for(int j=0;j<k;j++){
		A.put(i, j, v1);
	    }
	    
	}

	timer.start();
	int s=0;
	for(int ii=0;ii<r;ii++){
	    A.mult(x, b, flops, s);
	}
	timer.stop();
	size+=n*sizeof(double)+n*k*(2*sizeof(double)+sizeof(int))+2*(n+1)*sizeof(int);
	double norm=b.vector_norm();
 	cout << "norm=" << norm << endl;
    } else {
	usage(argv[0]);
    }
    cout << flops << " floating point operations in " << timer.time() << " seconds\n";
    cout << (double)flops*1.e-6/timer.time() << " MFLOPS\n";
    cout << (double)size/1024/1024 << " Megabytes in dataset\n";
    cout << (double)size/1024/1024*r/timer.time() << " MB/sec\n";
}
