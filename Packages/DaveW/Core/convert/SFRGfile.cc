/*
 *  SFRGfile.cc:  Get info on and/or change the type of a ScalarFieldRG
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Containers/Array2.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldRGdouble.h>
#include <Datatypes/ScalarFieldRGfloat.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/ScalarFieldRGshort.h>
#include <Datatypes/ScalarFieldRGuchar.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/MinMax.h>
#include <Persistent/Pstreams.h>
#include <fstream.h>
#include <iostream.h>
#include <math.h>
#include <stdio.h>

extern "C" {
#include <nrrd.h>
}

typedef enum {DOUBLE, FLOAT, INT, SHORT, UCHAR, CHAR} VTYPE; // voxel type
typedef enum {SCI, PCGV, NRRD, RAW} FTYPE; // file type
typedef enum {BIN, ASCII} BTYPE; // binary / ascii type (for SCI)

using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

VTYPE inVoxel;
VTYPE outVoxel;
FTYPE inFile;
FTYPE outFile;
BTYPE inBin;
BTYPE outBin;

int haveOutBin=0;
int haveOutVoxel=0;
int haveOutFile=0;

char *inName;
char *outName;
int haveMinMax;
int nx, ny, nz;
Point minIn, minOut, maxIn, maxOut;
double Omin, Omax, Ospan, Nmin, Nmax, Nspan;
double Cmin, Cmax;
bool newBB;
bool PCGVHeader;
ScalarFieldRGdouble *ifd=0;
ScalarFieldRGfloat *iff=0;
ScalarFieldRGint *ifi=0;
ScalarFieldRGshort *ifs=0;
ScalarFieldRGchar *ifc=0;
ScalarFieldRGuchar *ifu=0;

ScalarFieldHandle ifh;
ScalarFieldRGBase *isf;


void setDefaultArgs() {
    outVoxel = DOUBLE;
    outFile = SCI;
    outBin = BIN;
    inName = NULL;
    outName = NULL;
    newBB = false;
    haveMinMax = 0;
}

BTYPE get_SCIbtype(char *filename) {    ifstream* inp=new ifstream(filename);
    ifstream& in=(*inp);
    if(!in){
        cerr << "file not found: " << filename << endl;
        exit(0);
    }
    char m1, m2, m3, m4;
    // >> Won't work here - it eats white space...
    in.get(m1); in.get(m2); in.get(m3); in.get(m4);
    if(!in || m1 != 'S' || m2 != 'C' || m3 != 'I' || m4 != '\n'){
        cerr << filename << " is not a valid SCI file! (magic=" << m1 << m2 << m3 << m4 << ")\n";
        exit(0);
    }
    in.get(m1); in.get(m2); in.get(m3); in.get(m4);
    if(!in){
        cerr << "Error reading file: " << filename << " (while readint type)" << endl;
        exit(0);
    }
    int version;
    in >> version;
    if(!in){
        cerr << "Error reading file: " << filename << " (while reading version)" << endl;
        exit(0);
    }
    char m;
    do 	{
        in.get(m);
        if(!in){
            cerr << "Error reading file: " << filename << " (while reading newline)" << endl;
            exit(0);
        }
    } while(m != '\n');
    if(m1 == 'B' && m2 == 'I' && m3 == 'N'){
	inp->close();
        return BIN;
    } else if(m1 == 'A' && m2 == 'S' && m3 == 'C'){
	inp->close();
        return ASCII;
    } else {
        cerr << filename << " is an unknown type!\n";
        exit(0);
    }	
    return BIN;	// never reached
}


int parseArgs(int argc, char *argv[]) {
    if (argc<2) return 0;
    inName = argv[1];
    int i=2;
    cerr << "argc="<<argc<<"\n";
    while (i<argc) {
	if (i+1>=argc) return 0;
        if (!strcmp(argv[i], "-oname")) {
	    outName=argv[i+1];
	    i+=2;
	} else if (!strcmp(argv[i], "-ifile")) {
	    if (!strcmp(argv[i+1], "SCI")) inFile = SCI;
	    else if (!strcmp(argv[i+1], "NRRD")) inFile = NRRD;
	    else if (!strcmp(argv[i+1], "PCGV")) inFile = PCGV;
	    else if (!strcmp(argv[i+1], "RAW")) inFile = RAW;
	    else {
		cerr << "Unknow input type: "<<argv[i+1]<<"\n";
		return 0;
	    }
	    i+=2;
	} else if (!strcmp(argv[i], "-range")) {
	    Nmin = atof(argv[i+1]);
	    Cmin = atof(argv[i+2]);
	    Cmax = atof(argv[i+3]);
	    Nmax = atof(argv[i+4]);
	    Nspan =  Nmax - Nmin;
	    haveMinMax=1;
	    i+=5;
	} else if (!strcmp(argv[i], "-ofile")) {
	    if (!strcmp(argv[i+1], "SCI")) outFile = SCI;
	    else if (!strcmp(argv[i+1], "NRRD")) outFile = NRRD;
	    else if (!strcmp(argv[i+1], "PCGV")) outFile = PCGV;
	    else if (!strcmp(argv[i+1], "RAW")) outFile = RAW;
	    else {
		cerr << "Unknow output type: "<<argv[i+1]<<"\n";
		return 0;
	    }
	    haveOutFile=1;
	    i+=2;
	} else if (!strcmp(argv[i], "-bbox")) {
	    newBB = true;
	    minOut.x(atof(argv[i+1]));
	    minOut.y(atof(argv[i+2]));
	    minOut.z(atof(argv[i+3]));
	    maxOut.x(atof(argv[i+4]));
	    maxOut.y(atof(argv[i+5]));
	    maxOut.z(atof(argv[i+6]));
	    i+=7;
	} else if (!strcmp(argv[i], "-obin")) {
	    if (!strcmp(argv[i+1], "BIN")) outBin = BIN;
	    else if (!strcmp(argv[i+1], "ASCII")) outBin = ASCII;
	    else {
		cerr << "Unknown write type: "<<argv[i+1]<<"\n";
		return 0;
	    }
	    haveOutBin=1;
	    i+=2;
	} else if (!strcmp(argv[i], "-ovoxel")) {
	    if (!strcmp(argv[i+1], "double")) outVoxel = DOUBLE;
	    else if (!strcmp(argv[i+1], "float")) outVoxel = FLOAT;
	    else if (!strcmp(argv[i+1], "int")) outVoxel = INT;
	    else if (!strcmp(argv[i+1], "short")) outVoxel = SHORT;
	    else if (!strcmp(argv[i+1], "char")) outVoxel = CHAR;
	    else if (!strcmp(argv[i+1], "uchar")) outVoxel = UCHAR;
	    else {
		cerr << "Unknown data type: "<<argv[i+1]<<"\n";
		return 0;
	    }
	    haveOutVoxel=1;
	    i+=2;
	} else {
	    cerr << "Unknown argument: "<<argv[i]<<"\n";
	    return 0;
	}
    }
    return 1;
}

void printInputStats() {
    cerr << "\n\n\nInput is a ";
    if (inFile == SCI) cerr << "SCI ";
    else if (inFile == NRRD) cerr << "NRRD ";
    else if (inFile == PCGV) cerr << "PCGV ";
    else cerr << "RAW ";
    cerr << "file of ";
    if (inBin == BIN) cerr << "binary ";
    else cerr << "ASCII ";
    if (inVoxel == DOUBLE) cerr << "doubles ";
    else if (inVoxel == FLOAT) cerr << "floats ";
    else if (inVoxel == INT) cerr << "ints ";
    else if (inVoxel == SHORT) cerr << "shorts ";
    else if (inVoxel == CHAR) cerr << "chars ";
    else cerr << "unsigned chars ";
    cerr << "with dimensions ["<<nx<<", "<<ny<< ", "<<nz<<"] ";
    cerr << "\n    minVal="<<Omin<<", maxVal="<<Omax;
    cerr << "\n    and bounds "<<minIn<<" to "<<maxIn<<"\n\n";
}

void printOutputStats() {
    cerr << "\nOutputing to "<<outName<<" -- ";
    if (outFile == SCI) cerr << "SCI ";
    else if (outFile == NRRD) cerr << "NRRD ";
    else if (outFile == PCGV) cerr << "PCGV ";
    else cerr << "RAW ";
    cerr << "file of ";
    if (outBin == BIN) cerr << "binary ";
    else cerr << "ASCII ";
    if (outVoxel == DOUBLE) cerr << "doubles ";
    else if (outVoxel == FLOAT) cerr << "floats ";
    else if (outVoxel == INT) cerr << "ints ";
    else if (outVoxel == SHORT) cerr << "shorts ";
    else if (outVoxel == CHAR) cerr << "char ";
    else cerr << "unsigned chars ";
    if (haveMinMax) cerr << "\n    with minVal="<<Max(Nmin,Cmin)<<", maxVal="<<Min(Nmax,Cmax);
    cerr << "\n    and bounds "<<minOut<<" to "<<maxOut<<"\n\n";
}

void readSCI() {
    inBin = get_SCIbtype(inName);
    
    Piostream* stream=auto_istream(inName);
    if (!stream) {
	cerr << "Error: couldn't open "<<inName<<".\n";
	return;
    }
    
    Pio(*stream, ifh);
    if (!ifh.get_rep()) {
	cerr << "Error reading field "<<inName<<".\n";
	return;
    }

    isf=ifh->getRGBase();
    if(!isf){
	cerr << "FieldFilter can't deal with unstructured grids.\n";
	return;
    }

    nx = isf->nx;
    ny = isf->ny;
    nz = isf->nz;

    ifh->get_bounds(minIn,maxIn);

    if (!newBB) {
	minOut = minIn;
	maxOut = maxIn;
    }

    isf->get_minmax(Omin, Omax);

    ifd=isf->getRGDouble();
    iff=isf->getRGFloat();
    ifi=isf->getRGInt();
    ifs=isf->getRGShort();
    ifu=isf->getRGUchar();
    ifc=isf->getRGChar();

    if (ifd) inVoxel = DOUBLE;
    if (iff) inVoxel = FLOAT;
    if (ifi) inVoxel = INT;
    if (ifs) inVoxel = SHORT;
    if (ifu) inVoxel = UCHAR;
    if (ifc) inVoxel = CHAR;
}

void readNRRD() {
    FILE *f;
    Nrrd *nin;
    if (!(f = fopen(inName, "rt"))) {
	cerr << "Error opening file "<<inName<<"\n";
	exit(0);
    }
    if (!(nin=nrrdNewRead(f))) {
	char *err = nrrdStrdupErr();
	cerr << "Error reading nrrd:"<<err<<"\n";
	exit(0);
    }
    fclose(f);
    if (nin->dim != 3) {
	cerr << "Can only deal with 3-dimensional scalar fields... sorry.\n";
	exit(0);
    }
    nx = nin->size[0];
    ny = nin->size[1];
    nz = nin->size[2];




    // HACK -- need two values in each dimension -- just gonna copy!
//    nz++;





    minIn.x(nin->axisMin[0]);
    if ((minIn.x()<1 || minIn.x()>-1)) {	// !NaN
	minIn.y(nin->axisMin[1]);
	minIn.z(nin->axisMin[2]);
	maxIn.x(nin->axisMax[0]);
	maxIn.y(nin->axisMax[1]);
	maxIn.z(nin->axisMax[2]);
    } else {
	minIn=Point(0,0,0);
	maxIn=Point(nx-1, ny-1, nz-1);
    }

    if (!newBB) {
	minOut = minIn;
	maxOut = maxIn;
    }

    int i, j, k;
    if (nin->type == nrrdTypeChar) {
	inVoxel = CHAR;
	ifc = new ScalarFieldRGchar;
	ifc->resize(nx,ny,nz);
	ifc->set_bounds(minIn, maxIn);
	char *p=(char *)nin->data;
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		for(i=0; i<nx; i++) {
		    ifc->grid(i,j,k) = *p++;
		}
	    }
	}
	ifc->get_minmax(Omin, Omax);
    } else if (nin->type == nrrdTypeUChar) {
	inVoxel = UCHAR;
	ifu = new ScalarFieldRGuchar;
	ifu->resize(nx,ny,nz);
	ifu->set_bounds(minIn, maxIn);
	uchar *p=(uchar *)nin->data;
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		for(i=0; i<nx; i++) {
		    ifu->grid(i,j,k) = *p++;
		}
	    }
	}
	ifu->get_minmax(Omin, Omax);
    } else if (nin->type == nrrdTypeShort) {
	inVoxel = SHORT;
	ifs = new ScalarFieldRGshort;
	ifs->resize(nx,ny,nz);
	ifs->set_bounds(minIn, maxIn);
	short int *p=(short int *)nin->data;
	for (k=0; k<nz; k++) {




	    // HACK -- here's how we get two slices!
//	    p=(short int *)nin->data;




	    for (j=0; j<ny; j++) {
		for(i=0; i<nx; i++) {
		    ifs->grid(i,j,k) = *p++;
		}
	    }
	}
	ifs->get_minmax(Omin, Omax);
    } else if (nin->type == nrrdTypeInt) {
	inVoxel = INT;
	ifi = new ScalarFieldRGint;
	ifi->resize(nx,ny,nz);
	ifi->set_bounds(minIn, maxIn);
	int *p=(int *)nin->data;
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		for(i=0; i<nx; i++) {
		    ifi->grid(i,j,k) = *p++;
		}
	    }
	}
	ifi->get_minmax(Omin, Omax);
    } else if (nin->type == nrrdTypeFloat) {
	inVoxel = FLOAT;
	iff = new ScalarFieldRGfloat;
	iff->resize(nx,ny,nz);
	iff->set_bounds(minIn, maxIn);
	float *p=(float *)nin->data;
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		for(i=0; i<nx; i++) {
		    iff->grid(i,j,k) = *p++;
		}
	    }
	}
	iff->get_minmax(Omin, Omax);
    } else if (nin->type == nrrdTypeDouble) {
	inVoxel = DOUBLE;
	ifd = new ScalarFieldRGdouble;
	ifd->resize(nx,ny,nz);
	ifd->set_bounds(minIn, maxIn);
	double *p=(double *)nin->data;
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		for(i=0; i<nx; i++) {
		    ifd->grid(i,j,k) = *p++;
		}
	    }
	}
	ifd->get_minmax(Omin, Omax);
    } else {
	cerr << "Unrecognized nrrd type.\n";
	exit(0);
    }
}

void readOther() {
    FILE *f = fopen(inName, "rt");
    if (inFile == PCGV) {
	char header[4096];
	char dummy[200];
	fgets(header,4096,f); /* eat magic number line */
	fgets(header,4096,f); /* has dimensions */
	sscanf(header,"Volume dimensions: %d %d %d",&nx,&ny,&nz);
	double dx, dy, dz;
	fgets(header,4096,f); /* has spacing */
	sscanf(header,"Voxel spacing: %lf %lf %lf",&dx,&dy,&dz);
	fgets(header,4096,f); /* has order */
	sscanf(header,"Voxel order: %s", dummy);
	fgets(header,4096,f); /* has type */
	sscanf(header,"Voxel type: %s", dummy);
	if (dummy[0] == 'U') inVoxel = UCHAR;
	else if (dummy[0] == 'C') inVoxel = CHAR;
	else if (dummy[0] == 'S') inVoxel = SHORT;
	else if (dummy[0] == 'I') inVoxel = INT;
	else if (dummy[0] == 'F') inVoxel = FLOAT;
	else if (dummy[0] == 'D') inVoxel = DOUBLE;
	else {
	    cerr << "Unknown type in PCGV file: "<<dummy<<"\n";
	    exit(0);
	}
	
	minIn=Point(0,0,0);
	maxIn=Point((nx-1)*dx,(ny-1)*dy,(nz-1)*dz);
	if (!newBB) {
	    minOut=minIn;
	    maxOut=maxIn;
	}
	int done=0;
	while(!done) {
	    fgets(header,4096,f); /* eat line */
	    if (header[0] == '\n')
		done = 1;
	}
    } else {	// it's a RAW file -- we have to get this info from the user...
	cout << "Please enter nz: ";
	cin >> nz;
	cout << "Please enter ny: ";
	cin >> ny;
	cout << "Please enter nx: ";
	cin >> nx;
	char dummy[200];
	cout << "Please enter data type (C, S, I, F or D): ";
	cin >> dummy;
	if (dummy[0] == 'U') inVoxel = UCHAR;
	else if (dummy[0] == 'C') inVoxel = CHAR;
	else if (dummy[0] == 'S') inVoxel = SHORT;
	else if (dummy[0] == 'I') inVoxel = INT;
	else if (dummy[0] == 'F') inVoxel = FLOAT;
	else if (dummy[0] == 'D') inVoxel = DOUBLE;
	else {
	    cerr << "Unknown type: "<<dummy<<"\n";
	    exit(0);
	}
	if (newBB) {
	    minIn=minOut;
	    maxIn=maxOut;
	} else {
	    double dx, dy, dz;
	    cout << "Please enter dx: ";
	    cin >> dx;
	    cout << "Please enter dy: ";
	    cin >> dy;
	    cout << "Please enter dz: ";
	    cin >> dz;
	    minOut=minIn=Point(0,0,0);
	    maxOut=maxIn=Point((nx-1)*dx,(ny-1)*dy,(nz-1)*dz);
	}
    }
    // SCIRun is stored z fastest, not .vol files...

    int j,k;
    if (inVoxel == UCHAR) {
	ifu = new ScalarFieldRGuchar();
	ifu->resize(nx,ny,nz);
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		unsigned char buf[4096]; // just read into buf...
		fread(buf, sizeof(unsigned char), nx, f);
		for(int i=0;i<nx; i++) {
		    ifu->grid(i,j,k) = buf[i];
		}
	    }
	}
	ifu->get_minmax(Omin, Omax);
    } else if (inVoxel == CHAR) {
	ifc = new ScalarFieldRGchar();
	ifc->resize(nx,ny,nz);
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		unsigned short int buf[4096]; // just read into buf...
		fread(buf, sizeof(char), nx, f);
		for(int i=0;i<nx; i++) {
		    ifc->grid(i,j,k) = buf[i];
		}
	    }	
	}
	ifc->get_minmax(Omin, Omax);
    } else if (inVoxel == SHORT) {
	ifs = new ScalarFieldRGshort();
	ifs->resize(nx,ny,nz);
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		unsigned short int buf[4096]; // just read into buf...
		fread(buf, sizeof(unsigned short int), nx, f);
		for(int i=0;i<nx; i++) {
		    ifs->grid(i,j,k) = buf[i];
		}
	    }	
	}
	ifs->get_minmax(Omin, Omax);
    } else if (inVoxel == INT) {
	ifi = new ScalarFieldRGint();
	ifi->resize(nx,ny,nz);
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		int buf[4096]; // just read into buf...
		fread(buf, sizeof(int), nx, f);
		for(int i=0;i<nx; i++) {
		    ifi->grid(i,j,k) = buf[i];
		}
	    }
	}
	ifi->get_minmax(Omin, Omax);
    } else if (inVoxel == FLOAT) {
	iff = new ScalarFieldRGfloat();
	iff->resize(nx,ny,nz);
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		float buf[4096]; // just read into buf...
		fread(buf, sizeof(float), nx, f);
		for(int i=0;i<nx; i++) {
		    iff->grid(i,j,k) = buf[i];
		}
	    }
	}
	iff->get_minmax(Omin, Omax);
    } else if (inVoxel == DOUBLE) {
	ifd = new ScalarFieldRGdouble();
	ifd->resize(nx,ny,nz);
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		double buf[4096]; // just read into buf...
		fread(buf, sizeof(double), nx, f);
		for(int i=0;i<nx; i++) {
		    ifd->grid(i,j,k) = buf[i];
		}
	    }
	}
	ifd->get_minmax(Omin, Omax);
    }
    fclose(f);
}

void setBounds() {
    if (outVoxel == UCHAR) {
	ifu->set_bounds(minOut, maxOut);
    } else if (outVoxel == CHAR) {
	ifc->set_bounds(minOut, maxOut);
    } else if (outVoxel == SHORT) {
	ifs->set_bounds(minOut, maxOut);
    } else if (outVoxel == INT) {
	ifi->set_bounds(minOut, maxOut);
    } else if (outVoxel == FLOAT) {
	iff->set_bounds(minOut, maxOut);
    } else if (outVoxel == DOUBLE) {
	ifd->set_bounds(minOut, maxOut);
    }
}

inline double SETVAL(double val) {
    double v;
    if (!haveMinMax) return val;
    else v=(val-Omin)*Nspan/Ospan+Nmin;
    if (v<Cmin) return Cmin; else if (v>Cmax) return Cmax; else return v;
}

void revoxelize() {
    if (inVoxel == outVoxel && !haveMinMax) return;

    int i,j,k;
    if (haveMinMax) {
	if (inVoxel == UCHAR) ifu->get_minmax(Omin, Omax);
	else if (inVoxel == CHAR) ifc->get_minmax(Omin, Omax);
	else if (inVoxel == SHORT) ifs->get_minmax(Omin, Omax);
	else if (inVoxel == INT) ifi->get_minmax(Omin, Omax);
	else if (inVoxel == FLOAT) iff->get_minmax(Omin, Omax);
	else ifd->get_minmax(Omin, Omax);
	Ospan = Omax-Omin;
    }

    cerr << "Cmin="<<Cmin<<"  Cmax="<<Cmax<<"\n";
    cerr << "Omin="<<Omin<<"  Ospan="<<Ospan<<"\n";
    cerr << "Nmin="<<Nmin<<"  Nspan="<<Nspan<<"\n";

    if (outVoxel == UCHAR) {
	ifu = new ScalarFieldRGuchar;
	ifu->resize(nx,ny,nz);
    } else if (outVoxel == CHAR) {
	ifc = new ScalarFieldRGchar;
	ifc->resize(nx,ny,nz);
    } else if (outVoxel == SHORT) {
	ifs = new ScalarFieldRGshort;
	ifs->resize(nx,ny,nz);
    } else if (outVoxel == INT) {
	ifi = new ScalarFieldRGint;
	ifi->resize(nx,ny,nz);
    } else if (outVoxel == FLOAT) {
	iff = new ScalarFieldRGfloat;
	iff->resize(nx,ny,nz);
    } else if (outVoxel == DOUBLE) {
	ifd = new ScalarFieldRGdouble;
	ifd->resize(nx,ny,nz);
    }

    if (inVoxel == UCHAR) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifu->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifu->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == SHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifu->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifu->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifu->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifu->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == CHAR) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifc->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifc->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == SHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifc->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifc->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifc->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifc->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == SHORT) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifs->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifs->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == SHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifs->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifs->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifs->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifs->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == INT) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifi->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifi->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == SHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifi->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifi->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifi->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifi->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == FLOAT) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(iff->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(iff->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == SHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(iff->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(iff->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(iff->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(iff->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == DOUBLE) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifd->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifd->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == SHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifd->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifd->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifd->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifd->grid(i,j,k));
		    }
		}
	    }
	}
    }
}

void writeSCI() {
    ScalarFieldHandle sfh;
    if (outVoxel == UCHAR) sfh=ifu;
    else if (outVoxel == CHAR) sfh=ifc;
    else if (outVoxel == SHORT) sfh=ifs;
    else if (outVoxel == INT) sfh=ifi;
    else if (outVoxel == FLOAT) sfh=iff;
    else if (outVoxel == DOUBLE) sfh=ifd;

    if (outBin == BIN) {
	BinaryPiostream stream(outName, Piostream::Write);
	Pio(stream, sfh);
    } else if (outBin == ASCII) {
	TextPiostream stream(outName, Piostream::Write);
	Pio(stream, sfh);
    }
}

void writeNRRD() {
    FILE *f;
    if (!(f = fopen(outName, "wt"))) {
	cerr << "Error opening file "<<outName<<"\n";
	exit(0);
    }
    ScalarFieldHandle sfh;
    if (outVoxel == UCHAR) sfh=ifu;
    else if (outVoxel == CHAR) sfh=ifc;
    else if (outVoxel == SHORT) sfh=ifs;
    else if (outVoxel == INT) sfh=ifi;
    else if (outVoxel == FLOAT) sfh=iff;
    else if (outVoxel == DOUBLE) sfh=ifd;

    ScalarFieldRGBase* sfb=sfh->getRGBase();
    int nx=sfb->nx;
    int ny=sfb->ny;
    int nz=sfb->nz;
    Nrrd *nout;

    if (outVoxel == UCHAR) {
	uchar *data=new uchar[nx*ny*nz];
	uchar *p=&(data[0]);
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    *p++=ifu->grid(i,j,k);
	nout=nrrdNewWrap(data, nx*ny*nz, nrrdTypeUChar, 3);
    } else if (outVoxel == CHAR) {
	char *data=new char[nx*ny*nz];
	char *p=&(data[0]);
	for (int k=0; k<nz; k++)
	    for (int j=0; j<ny; j++)
		for (int i=0; i<nx; i++)
		    *p++=ifc->grid(i,j,k);
	nout=nrrdNewWrap(data, nx*ny*nz, nrrdTypeChar, 3);
    } else if (outVoxel == SHORT) {
	short *data=new short[nx*ny*nz];
	short *p=&(data[0]);
	for (int k=0; k<nz; k++)
	    for (int j=0; j<ny; j++)
		for (int i=0; i<nx; i++)
		    *p++=ifs->grid(i,j,k);
	nout=nrrdNewWrap(data, nx*ny*nz, nrrdTypeShort, 3);
    } else if (outVoxel == INT) {
	int *data=new int[nx*ny*nz];
	int *p=&(data[0]);
	for (int k=0; k<nz; k++)
	    for (int j=0; j<ny; j++)
		for (int i=0; i<nx; i++)
		    *p++=ifi->grid(i,j,k);
	nout=nrrdNewWrap(data, nx*ny*nz, nrrdTypeInt, 3);
    } else if (outVoxel == FLOAT) {
	float *data=new float[nx*ny*nz];
	float *p=&(data[0]);
	for (int k=0; k<nz; k++)
	    for (int j=0; j<ny; j++)
		for (int i=0; i<nx; i++)
		    *p++=iff->grid(i,j,k);
	nout=nrrdNewWrap(data, nx*ny*nz, nrrdTypeFloat, 3);
    } else if (outVoxel == DOUBLE) {
	double *data=new double[nx*ny*nz];
	double *p=&(data[0]);
	for (int k=0; k<nz; k++)
	    for (int j=0; j<ny; j++)
		for (int i=0; i<nx; i++)
		    *p++=ifd->grid(i,j,k);
	nout=nrrdNewWrap(data, nx*ny*nz, nrrdTypeDouble, 3);
    } 

    double min, max;
    sfb->get_minmax(min, max);
    nout->min=min;
    nout->max=max;
    nout->encoding=nrrdEncodingRaw;
    nout->size[0]=nx;
    nout->size[1]=ny;
    nout->size[2]=nz;
    Point minP, maxP;
    sfb->get_bounds(minP, maxP);
    nout->axisMin[0]=minP.x();
    nout->axisMin[1]=minP.y();
    nout->axisMin[2]=minP.z();
    nout->axisMax[0]=maxP.x();
    nout->axisMax[1]=maxP.y();
    nout->axisMax[2]=maxP.z();
    nout->label[0][0]='x';
    nout->label[0][1]='\0';
    nout->label[1][0]='y';
    nout->label[1][1]='\0';
    nout->label[2][0]='z';
    nout->label[2][1]='\0';

    if (nrrdWrite(f, nout)) {
	char *err = nrrdStrdupErr();
	cerr << "Error writing nrrd:"<<err<<"\n";
	exit(0);
    }
    fclose(f);
}

void writeOther() {
    FILE *f = fopen(outName, "wt");

    double dx,dy,dz;
    Vector v(maxOut-minOut);
    dx = v.x()/(nx-1);
    dy = v.y()/(ny-1);
    dz = v.z()/(nz-1);

    if (outFile == PCGV) {	// otherwise it's RAW -- no header
	fprintf(f,"PCGV00.01HL\n");
	fprintf(f,"Volume dimensions: %d %d %d\n",nx,ny,nz);
	fprintf(f,"Voxel spacing: %lf %lf %lf\n",dx,dy,dz);
	fprintf(f,"Voxel order: ZYX\n");
	fprintf(f,"Voxel type: ");
	if (outVoxel == UCHAR) fprintf(f,"UChar\n");
	else if (outVoxel == CHAR) fprintf(f,"Char\n");
	else if (outVoxel == SHORT) fprintf(f, "Short\n");
	else if (outVoxel == INT) fprintf(f, "Int\n");
	else if (outVoxel == FLOAT) fprintf(f, "Float\n");
	else if (outVoxel == DOUBLE) fprintf(f, "Double\n");
	long int size=nx*ny*nz;
	if (outVoxel == UCHAR) size*=sizeof(unsigned char);
	else if (outVoxel == CHAR) size*=sizeof(char);
	else if (outVoxel == SHORT) size*=sizeof(short int);
	else if (outVoxel == INT) size*=sizeof(int);
	else if (outVoxel == FLOAT) size*=sizeof(float);
	else if (outVoxel == DOUBLE) size*=sizeof(double);
	fprintf(f,"Data size: %ld\n", size);
	fprintf(f,"Comment: just more data...\n\n");
    }
    int j,k;
    if (outVoxel == UCHAR) {
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		unsigned char buf[4096]; // just read into buf...
		for(int i=0;i<nx; i++) {
		    buf[i] = ifu->grid(i,j,k);
		}
		fwrite(buf, sizeof(unsigned char), nx, f);
	    }
	}
    } else if (outVoxel == CHAR) {
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		unsigned char buf[4096]; // just read into buf...
		for(int i=0;i<nx; i++) {
		    buf[i] = ifc->grid(i,j,k);
		}
		fwrite(buf, sizeof(unsigned char), nx, f);
	    }
	}
    } else if (outVoxel == SHORT) {
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		unsigned short int buf[4096]; // just read into buf...
		for(int i=0;i<nx; i++) {
		    buf[i] = ifs->grid(i,j,k);
		}
		fwrite(buf, sizeof(unsigned short int), nx, f);
	    }
	}
    } else if (outVoxel == INT) {
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		int buf[4096]; // just read into buf...
		for(int i=0;i<nx; i++) {
		    buf[i] = ifi->grid(i,j,k);
		}
		fwrite(buf, sizeof(int), nx, f);
	    }
	}
    } else if (outVoxel == FLOAT) {
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		float buf[4096]; // just read into buf...
		for(int i=0;i<nx; i++) {
		    buf[i] = iff->grid(i,j,k);
		}
		fwrite(buf, sizeof(float), nx, f);
	    }
	}
    } else if (outVoxel == DOUBLE) {
	for (k=0; k<nz; k++) {
	    for (j=0; j<ny; j++) {
		double buf[4096]; // just read into buf...
		for(int i=0;i<nx; i++) {
		    buf[i] = ifd->grid(i,j,k);
		}
		fwrite(buf, sizeof(double), nx, f);
	    }
	}
    }
    fclose(f);
}
    
void main(int argc, char *argv[]) {
    setDefaultArgs();
    if (!parseArgs(argc, argv)) {
	cerr << "Usage: "<<argv[0]<<" iname [-bbox xmin ymin zmin xmax ymax zmax] [-range RangeMin CropMin CropMax RangeMax] [-ifile { SCI | NRRD| PCGV | RAW }] [-oname name] [-ofile {SCI | NRRD | PCGV | RAW }] [-obin { BIN | ASCII }] [-ovoxel { double | float | int | short | char | uchar }]\n";
	return;
    }

    if (inFile == SCI) {
	readSCI();
    } else if (inFile == NRRD) {
	readNRRD();
    } else {	
	readOther();   // this works for PCGV or RAW
    }
    printInputStats();
    if (!outName) return;

    if (!haveOutBin) { outBin = inBin; cerr << "Copying bin type.\n"; }
    if (!haveOutVoxel) { outVoxel = inVoxel; cerr << "Copying voxel type.\n"; }
    if (!haveOutFile) { outFile = inFile; cerr << "Copying file type.\n"; }

    printOutputStats();
    if (outVoxel != inVoxel) revoxelize();
    setBounds();
    if (outFile == SCI) {
	writeSCI();
    } else if (outFile == NRRD) {
	writeNRRD();
    } else {	
	writeOther();  // this works for PCGV or RAW
    }
}
