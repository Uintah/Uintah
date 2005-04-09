
#include <Classlib/Pstreams.h>
#include <Classlib/Timer.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/SymSparseRowMatrix.h>
#include <iostream.h>
#include <math.h>
#include <fstream.h>

void usage(char* progname)
{
    cerr << "usage: " << progname << " filebase" << endl;
    exit(1);
}


main(int argc, char** argv)
{
    if(argc != 2)
	usage(argv[0]);
    char buf[200];
    strcpy(buf, argv[1]);
    strcat(buf, ".mat");
    Piostream* stream=auto_istream(buf);
    if(!stream){
	cerr << "Error opening matrix: " << buf << endl;
	exit(1);
    }
    MatrixHandle mat;
    Pio(*stream, mat);
    if(!mat.get_rep()){
	cerr << "Error reading matrix: " << buf << endl;
	exit(1);
    }
    delete stream;

    strcpy(buf, argv[1]);
    strcat(buf, ".rhs");
    stream=auto_istream(buf);
    if(!stream){
	cerr << "Error opening rhs: " << buf << endl;
	exit(1);
    }
    ColumnMatrixHandle rhshandle;
    Pio(*stream, rhshandle);
    if(!rhshandle.get_rep()){
	cerr << "Error reading rhs: " << buf << endl;
	exit(1);
    }

    strcpy(buf, argv[1]);
    strcat(buf, ".sol");
    stream=auto_istream(buf);
    if(!stream){
	cerr << "Error opening solution: " << buf << endl;
	exit(1);
    }
    ColumnMatrixHandle correct_solhandle;
    Pio(*stream, correct_solhandle);
    if(!correct_solhandle.get_rep()){
	cerr << "Error reading solution: " << buf << endl;
	exit(1);
    }

    {
	strcpy(buf, argv[1]);
	strcat(buf, ".mat.text");
	ofstream omat(buf);
	omat.precision(17);
	SymSparseRowMatrix* ss=(SymSparseRowMatrix*)mat.get_rep();
	omat << ss->nrows() << " " << ss->ncols() << " " << ss->nnz << '\n';
	int nrows=ss->nrows();
	int i;
	for(i=0;i<=nrows;i++)
	    omat << ss->rows[i] << '\n';
	int nnz=ss->nnz;
	int* c=ss->columns;
	double* a=ss->a;
	for(i=0;i<nrows;i++){
	    omat << '\n';
	    int first=ss->rows[i];
	    int last=ss->rows[i+1];
	    for(int j=first;j<last;j++)
		omat << *c++ << " " << *a++ << '\n';
	}
    }

    ColumnMatrix& rhs=*rhshandle.get_rep();
    ColumnMatrix& sol=*correct_solhandle.get_rep();
    {
	strcpy(buf, argv[1]);
	strcat(buf, ".rhs.text");
	ofstream orhs(buf);
	orhs.precision(17);
	orhs << rhs.nrows() << '\n';
	for(int i=0;i<rhs.nrows();i++)
	    orhs << rhs[i] << '\n';
    }

    {
	strcpy(buf, argv[1]);
	strcat(buf, ".sol.text");
	ofstream osol(buf);
	osol.precision(17);
	osol << sol.nrows() << '\n';
	for(int i=0;i<sol.nrows();i++)
	    osol << sol[i] << '\n';
    }

}
