/*
 *  SFRGfile.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <TCL/TCLvar.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/ScalarFieldPort.h>
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
#include <stdio.h>
#include <math.h>
#include <iostream.h>
#include <fstream.h>

typedef enum {DOUBLE, FLOAT, INT, USHORT, UCHAR, CHAR} VTYPE; // voxel type

class SFRGfile : public Module {
    ScalarFieldIPort* iField;
    ScalarFieldOPort* oField;

    VTYPE inVoxel;
    VTYPE outVoxel;

    char *inName;
    char *outName;
    int haveMinMax;
    int haveOutVoxel;
    int haveBBox;

    int nx, ny, nz;
    Point minIn, minOut, maxIn, maxOut;
    double Omin, Omax, Ospan, Nmin, Nmax, Nspan;
    double Cmin, Cmax;
    bool newBB;
    bool PCGVHeader;

    ScalarFieldRGdouble *ifd;
    ScalarFieldRGfloat *iff;
    ScalarFieldRGint *ifi;
    ScalarFieldRGshort *ifs;
    ScalarFieldRGchar *ifc;
    ScalarFieldRGuchar *ifu;
    
    ScalarFieldHandle ifh;
    ScalarFieldRGBase *isf;
    ScalarFieldHandle ofh;

    TCLint haveMinMaxTCL;
    TCLint haveOutVoxelTCL;
    TCLint haveBBoxTCL;
    TCLint outVoxelTCL;
    TCLstring NminTCL;
    TCLstring NmaxTCL;
    TCLstring CminTCL;
    TCLstring CmaxTCL;
    TCLstring minOutTCLX;
    TCLstring minOutTCLY;
    TCLstring minOutTCLZ;
    TCLstring maxOutTCLX;
    TCLstring maxOutTCLY;
    TCLstring maxOutTCLZ;

    void setDefaultArgs();
    void checkInterface();
    void printInputStats();
    void printOutputStats();
    void readSCI();
    void setBounds();
    inline double SETVAL(double val);
    void revoxelize();
    void writeSCI();
public:
    SFRGfile(const clString& id);
    SFRGfile(const SFRGfile&, int deep);
    virtual ~SFRGfile();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SFRGfile(const clString& id)
{
    return new SFRGfile(id);
}
}

static clString module_name("SFRGfile");
SFRGfile::SFRGfile(const clString& id)
: Module("SFRGfile", id, Filter), haveMinMaxTCL("haveMinMaxTCL", id, this),
  haveOutVoxelTCL("haveOutVoxelTCL", id, this), 
  haveBBoxTCL("haveBBoxTCL", id, this), NminTCL("NMinTCL", id, this),
  NmaxTCL("NmaxTCL", id, this), CminTCL("CminTCL", id, this),
  CmaxTCL("CmaxTCL", id, this), minOutTCLX("minOutTCLX", id, this),
  minOutTCLY("minOutTCLY", id, this), minOutTCLZ("minOutTCLZ", id, this),
  maxOutTCLX("maxOutTCLX", id, this), maxOutTCLY("maxOutTCLY", id, this), 
  maxOutTCLZ("maxOutTCLZ", id, this), outVoxelTCL("outVoxelTCL", id, this)
{
    iField=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(iField);
    // Create the output ports
    oField=new ScalarFieldOPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_oport(oField);
}

SFRGfile::SFRGfile(const SFRGfile& copy, int deep)
: Module(copy, deep), haveMinMaxTCL("haveMinMaxTCL", id, this),
  haveOutVoxelTCL("haveOutVoxelTCL", id, this), 
  haveBBoxTCL("haveBBoxTCL", id, this), NminTCL("NMinTCL", id, this),
  NmaxTCL("NmaxTCL", id, this), CminTCL("CminTCL", id, this),
  CmaxTCL("CmaxTCL", id, this), minOutTCLX("minOutTCLX", id, this),
  minOutTCLY("minOutTCLY", id, this), minOutTCLZ("minOutTCLZ", id, this),
  maxOutTCLX("maxOutTCLX", id, this), maxOutTCLY("maxOutTCLY", id, this), 
  maxOutTCLZ("maxOutTCLZ", id, this), outVoxelTCL("outVoxelTCL", id, this)
{
}

SFRGfile::~SFRGfile()
{
}

Module* SFRGfile::clone(int deep)
{
    return new SFRGfile(*this, deep);
}

void SFRGfile::execute() {
    setDefaultArgs();
    if(!iField->get(ifh))
	return;
    isf=ifh->getRGBase();
    if(!isf){
	error("SFRGfile can't deal with unstructured grids!");
	return;
    }
    checkInterface();

    readSCI();
    printInputStats();

    if (!haveOutVoxel) outVoxel=inVoxel;

    printOutputStats();
    revoxelize();
    setBounds();

    writeSCI();
    oField->send(ofh);
}

void SFRGfile::setDefaultArgs() {
    outVoxel = DOUBLE;

    haveBBox = false;
    haveMinMax = 0;
    haveOutVoxel=0;

    ifd=0;
    iff=0;
    ifi=0;
    ifs=0;
    ifc=0;
    ifu=0;
}

void SFRGfile::checkInterface() {
    haveMinMax=haveMinMaxTCL.get();
    if (haveMinMax) {
	(NminTCL.get()).get_double(Nmin);
	(NmaxTCL.get()).get_double(Nmin);
	(CminTCL.get()).get_double(Nmin);
	(CmaxTCL.get()).get_double(Nmin);
	Nspan=Nmax-Nmin;
    }
    haveBBox=haveBBoxTCL.get();
    if (haveBBox) {
	double tmp;
	(minOutTCLX.get()).get_double(tmp);
	minOut.x(tmp);
	(minOutTCLY.get()).get_double(tmp);
	minOut.y(tmp);
	(minOutTCLZ.get()).get_double(tmp);
	minOut.z(tmp);
	(maxOutTCLX.get()).get_double(tmp);
	maxOut.x(tmp);
	(maxOutTCLY.get()).get_double(tmp);
	maxOut.y(tmp);
	(maxOutTCLZ.get()).get_double(tmp);
	maxOut.z(tmp);
    }
    haveOutVoxel=haveOutVoxelTCL.get();
    if (haveOutVoxel) {
	outVoxel=(VTYPE) outVoxelTCL.get();
    }
}

void SFRGfile::printInputStats() {
    cerr << "\n\n\nInput is ";
    if (inVoxel == DOUBLE) cerr << "doubles ";
    else if (inVoxel == FLOAT) cerr << "floats ";
    else if (inVoxel == INT) cerr << "ints ";
    else if (inVoxel == USHORT) cerr << "unsigned shorts ";
    else if (inVoxel == CHAR) cerr << "chars ";
    else cerr << "unsigned chars ";
    cerr << "with dimensions ["<<nx<<", "<<ny<< ", "<<nz<<"] ";
    cerr << "\n    minVal="<<Omin<<", maxVal="<<Omax;
    cerr << "\n    and bounds "<<minIn<<" to "<<maxIn<<"\n\n";
}

void SFRGfile::printOutputStats() {
    cerr << "\nOutput is ";
    if (outVoxel == DOUBLE) cerr << "doubles ";
    else if (outVoxel == FLOAT) cerr << "floats ";
    else if (outVoxel == INT) cerr << "ints ";
    else if (outVoxel == USHORT) cerr << "unsigned shorts ";
    else if (outVoxel == CHAR) cerr << "char ";
    else cerr << "unsigned chars ";
    if (haveMinMax) cerr << "\n    with minVal="<<Max(Nmin,Cmin)<<", maxVal="<<Min(Nmax,Cmax);
    cerr << "\n    and bounds "<<minOut<<" to "<<maxOut<<"\n\n";
}

void SFRGfile::readSCI() {
    nx = isf->nx;
    ny = isf->ny;
    nz = isf->nz;

    ifh->get_bounds(minIn,maxIn);

    if (!haveBBox) {
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
    if (ifs) inVoxel = USHORT;
    if (ifu) inVoxel = UCHAR;
    if (ifc) inVoxel = CHAR;
}


void SFRGfile::setBounds() {
    if (outVoxel == UCHAR) {
	ifu->set_bounds(minOut, maxOut);
    } else if (outVoxel == CHAR) {
	ifc->set_bounds(minOut, maxOut);
    } else if (outVoxel == USHORT) {
	ifs->set_bounds(minOut, maxOut);
    } else if (outVoxel == INT) {
	ifi->set_bounds(minOut, maxOut);
    } else if (outVoxel == FLOAT) {
	iff->set_bounds(minOut, maxOut);
    } else if (outVoxel == DOUBLE) {
	ifd->set_bounds(minOut, maxOut);
    }
}

inline double SFRGfile::SETVAL(double val) {
    double v;
    if (!haveMinMax) return val;
    else v=(val-Omin)*Nspan/Ospan+Nmin;
    if (v<Cmin) return Cmin; else if (v>Cmax) return Cmax; else return v;
}

void SFRGfile::revoxelize() {
    if (inVoxel == outVoxel && !haveMinMax) return;

    int i,j,k;
    if (haveMinMax) {
	if (inVoxel == UCHAR) ifu->get_minmax(Omin, Omax);
	else if (inVoxel == CHAR) ifc->get_minmax(Omin, Omax);
	else if (inVoxel == USHORT) ifs->get_minmax(Omin, Omax);
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
    } else if (outVoxel == USHORT) {
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
	} else if (outVoxel == USHORT) {
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
	} else if (outVoxel == USHORT) {
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
    } else if (inVoxel == USHORT) {
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
	} else if (outVoxel == USHORT) {
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
	} else if (outVoxel == USHORT) {
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
	} else if (outVoxel == USHORT) {
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
	} else if (outVoxel == USHORT) {
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

void SFRGfile::writeSCI() {
    if (outVoxel == UCHAR) ofh=ifu;
    else if (outVoxel == CHAR) ofh=ifc;
    else if (outVoxel == USHORT) ofh=ifs;
    else if (outVoxel == INT) ofh=ifi;
    else if (outVoxel == FLOAT) ofh=iff;
    else if (outVoxel == DOUBLE) ofh=ifd;
}
    
