
#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Datatypes/SymSparseRowMatrix.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/ColumnMatrix.h>
#include <iostream.h>
#include <Math/Expon.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <stdio.h>

#define N 10

class FDMatrix : public Matrix {
    int size;
    int n;
    double eps;
public:
    FDMatrix(int n, double eps);
    ~FDMatrix();
    virtual double& get(int, int);
    virtual void put(int, int, const double&);
    virtual int nrows() const;
    virtual int ncols() const;
    virtual double minValue();
    virtual double maxValue();
    double density();
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
    virtual void zero();
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1);
    virtual void print();
};

void usage(char* progname)
{
    cerr << progname << " n\n";
    exit(-1);

}

void dump_frame(double* data, int n, int frame)
{
    char buf[100];
    sprintf(buf, "f%02d.raw", frame);
    FILE* fp=fopen(buf, "w");
    unsigned char* out=new unsigned char[n*3];
    for(int i=0;i<n;i++){
	for(int j=0;j<n;j++){
	    int idx=i*n+j;
	    double d=data[idx];
	    d=d<0?0:d>255?255:d;
	    out[j*3]=out[j*3+1]=out[j*3+2]=(unsigned char)d;
	}
	fwrite(out, sizeof(unsigned char), n*3, fp);
    }
    fclose(fp);
    delete[] out;
}

int main(int argc, char** argv)
{
    if(argc!=2){
	usage(argv[0]);
    }
    int n=atoi(argv[1]);
    Array2<double> from_obj(n,n);
    Array2<double> to_obj(n,n);
    // Make the from object
    // Make the to object
    FILE* fp1=fopen("p1.in", "r");
    FILE* fp2=fopen("p2.in", "r");
    unsigned char* in1=new unsigned char[n*3];
    unsigned char* in2=new unsigned char[n*3];
    for(int i=0;i<n;i++){
	int s=fread(in1, sizeof(unsigned char), n*3, fp1);
//	cerr << "s=" << s << endl;
	s=fread(in2, sizeof(unsigned char), n*3, fp2);
	for(int j=0;j<n;j++){
	    from_obj(i,j)=(double(in1[j*3])+double(in1[j*3+1])+double(in1[j*3+2]))/3;
	    to_obj(i,j)=(double(in2[j*3])+double(in2[j*3+1])+double(in2[j*3+2]))/3;
//	    cerr << "from_obj(" << i << ", " << j << ")=" << from_obj(i,j) << endl;
//	    cerr << "to_obj(" << i << ", " << j << ")=" << to_obj(i,j) << endl;
	}
    }    
    delete[] in1;
    delete[] in2;
    fclose(fp1);
    fclose(fp2);

    // Construct A
    FDMatrix A(n, .01);
    
    // Construct x
    ColumnMatrix tovec(n*n);
    for(i=0;i<n;i++){
	for(int j=0;j<n;j++){
	    int idx=i*n+j;
	    tovec[idx]=to_obj(i,j);
	    //cout << "x[" << idx << "]=" << to_obj(i,j) << endl;
	}
    }
    // Find b
    ColumnMatrix b(n*n);
    int f, m;
    A.mult(tovec, b, f, m);
    // Starting guess
    ColumnMatrix x(n*n);
    for(i=0;i<n;i++){
	for(int j=0;j<n;j++){
	    int idx=i*n+j;
	    x[idx]=from_obj(i,j);
	}
    }
    ColumnMatrix& lhs=x;
    ColumnMatrix& rhs=b;
    Matrix* matrix=&A;

    // CG iterations...
    int size=n*n;
    for(i=0;i<size;i++){
	//cout << "b[" << i << "]=" << b[i] << endl;
    }

    int flop=0;
    int memref=0;
    ColumnMatrix diag(size);
    // We should try to do a better job at preconditioning...
    for(i=0;i<size;i++){
	diag[i]=1./matrix->get(i,i);
    }
    flop+=size;
    memref=2*size*sizeof(double);

    ColumnMatrix R(size);
    matrix->mult(lhs, R, flop, memref);

    int frame=0;
    dump_frame(&lhs[0], n, frame++);

    Sub(R, rhs, R, flop, memref);
    double bnorm=rhs.vector_norm(flop, memref);

    ColumnMatrix Z(size);
    matrix->mult(R, Z, flop, memref);

    ColumnMatrix P(size);
    double bkden=0;
    double err=R.vector_norm(flop, memref)/bnorm;
    //cerr << "bnorn=" << bnorm << endl;
    cerr << "err=" << err << endl;
    if(err == 0){
	lhs=rhs;
	return 0;
    }

    int niter=0;
    int toomany=2*size;
    double max_error=1.e-1;

    while(niter < toomany){
	niter++;

	if(err < max_error)
	    break;

	// Simple Preconditioning...
	Mult(Z, R, diag, flop, memref);	

	// Calculate coefficient bk and direction vectors p and pp
	double bknum=Dot(Z, R, flop, memref);

	if(niter==1){
	    P=Z;
	    memref+=2*sizeof(double);
	} else {
	    double bk=bknum/bkden;
	    ScMult_Add(P, bk, P, Z, flop, memref);
	}
	bkden=bknum;

	// Calculate coefficient ak, new iterate x and new residuals r and rr
	matrix->mult(P, Z, flop, memref);

	double akden=Dot(Z, P, flop, memref);

	double ak=bknum/akden;
	ScMult_Add(lhs, ak, P, lhs, flop, memref);
	ScMult_Add(R, -ak, Z, R, flop, memref);
	dump_frame(&lhs[0], n, frame++);

	err=R.vector_norm(flop, memref)/bnorm;
	cerr << "err=" << err << endl;
    }
    dump_frame(&tovec[0], n, frame++);
}

FDMatrix::FDMatrix(int n, double eps)
: Matrix(Matrix::symmetric, Matrix::symsparse), n(n), size(n*n), eps(eps)
{
}

FDMatrix::~FDMatrix()
{
}

void FDMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs, int, int)
{
    mult(x, b, flops, memrefs);
}

void FDMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
		    int& flops, int& memrefs, int, int) const
{
    double eps2=eps+2;
    double eps3=eps+3;
    double eps4=eps+4;
    int idx=0;
    b[idx]=x[idx]*eps2-x[idx+1]-x[idx+n];
    for(int j=1;j<n-1;j++){
	int idx=j;
	b[idx]=x[idx]*eps3-x[idx-1]-x[idx+1]-x[idx+n];
    }
    idx=n-1;
    b[idx]=x[idx]*eps2-x[idx-1]-x[idx+n];
    for(int i=1;i<n-1;i++){
	int idx=i*n;
	b[idx]=x[idx]*eps3-x[idx-n]-x[idx+n]-x[idx+1];
	for(int j=1;j<n-1;j++){
	    int idx=i*n+j;
	    b[idx]=x[idx]*eps4-x[idx-n]-x[idx+n]-x[idx-1]-x[idx+1];
	}
	b[idx]=x[idx]*eps3-x[idx-n]-x[idx+n]-x[idx-1];
    }
    idx=(n-1)*n;
    b[idx]=x[idx]*eps2-x[idx+1]-x[idx-n];
    for(j=1;j<n-1;j++){
	int idx=(n-1)*n+j;
	b[idx]=x[idx]*eps3-x[idx-1]-x[idx+1]-x[idx-n];
    }
    idx=(n-1)*n+n-1;
    b[idx]=x[idx]*eps2-x[idx-1]-x[idx-n];
}

void FDMatrix::zero()
{
    NOT_FINISHED("FDMatrix::zero");
}

int FDMatrix::nrows() const
{
    return size;
}

int FDMatrix::ncols() const
{
    return size;
}

double& FDMatrix::get(int i, int j)
{
    static double dummy=0;
    if(i==j){
	if(i==0 || i==size-1)
	    dummy=eps+2;
	else if(i%n==0 || i%n==n-1)
	    dummy=eps+3;
	else
	    dummy=eps+4;
	return dummy;
    }
	
    NOT_FINISHED("FDMatrix::get");
    return dummy;
}

double FDMatrix::minValue()
{
    return Min(-1.,2+eps);
}

double FDMatrix::maxValue()
{
    return Max(-1.,4+eps);
}

void FDMatrix::print()
{
    NOT_FINISHED("FDMatrix::print");
}

void FDMatrix::put(int, int, const double&)
{
    NOT_FINISHED("FDMatrix::put");
}

void FDMatrix::getRowNonzeros(int, Array1<int>&, 
			      Array1<double>&)
{
    NOT_FINISHED("FDMatrix::getRowNonzeros");
}

