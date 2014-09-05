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

/* comments/disclaimers --
 *    the original data is modified if only the bounding box is flagged
 *	as having changed.
 *    in contrast a detached copy is created and modified if the type of 
 *      data (double, int, uchar, etc) or values of the data are
 *    	flagged to change
 */

#include <SCICore/Containers/Array2.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
using std::cerr;

typedef enum {DOUBLE, FLOAT, INT, USHORT, UCHAR, CHAR} VTYPE; // voxel type

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

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

    void checkInterface();
    void printInputStats();
    void printOutputStats();
    void setInputFieldVars();
    void setBounds();
    inline double SETVAL(double val);
    void revoxelize();
    void setOutputFieldHandle();
public:
    SFRGfile(const clString& id);
    virtual ~SFRGfile();
    virtual void execute();
};

extern "C" Module* make_SFRGfile(const clString& id)
{
    return new SFRGfile(id);
}

static clString module_name("SFRGfile");
SFRGfile::SFRGfile(const clString& id)
: Module("SFRGfile", id, Filter), haveMinMaxTCL("haveMinMaxTCL", id, this),
  haveOutVoxelTCL("haveOutVoxelTCL", id, this), 
  haveBBoxTCL("haveBBoxTCL", id, this), NminTCL("NminTCL", id, this),
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

SFRGfile::~SFRGfile()
{
}

void SFRGfile::execute() {
    if(!iField->get(ifh) || !ifh.get_rep())
	return;
    isf=ifh->getRGBase();
    if(!isf){
	cerr<<"SFRGfile can't deal with unstructured grids!";
	return;
    }
    checkInterface();
    setInputFieldVars();
    printInputStats();

    if (!haveOutVoxel) outVoxel=inVoxel;

    revoxelize();
    setBounds();
    printOutputStats();

    setOutputFieldHandle();
    oField->send(ofh);
}

void SFRGfile::checkInterface() {
    haveMinMax=haveMinMaxTCL.get();
    if (haveMinMax) {
	(NminTCL.get()).get_double(Nmin);
	cerr << "In checkinterface: Nmin="<<Nmin<<"\n";
	(NmaxTCL.get()).get_double(Nmax);
	cerr << "In checkinterface: Nmax="<<Nmax<<"\n";
	(CminTCL.get()).get_double(Cmin);
	cerr << "In checkinterface: Cmin="<<Cmin<<"\n";
	(CmaxTCL.get()).get_double(Cmax);
	cerr << "In checkinterface: Cmax="<<Cmax<<"\n";
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

void SFRGfile::setInputFieldVars() {
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

    ScalarFieldRGuchar *ifuo=ifu;
    ScalarFieldRGchar *ifco=ifc;
    ScalarFieldRGshort *ifso=ifs;
    ScalarFieldRGint *ifio=ifi;
    ScalarFieldRGfloat *iffo=iff;
    ScalarFieldRGdouble *ifdo=ifd;

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
			ifu->grid(i,j,k) = SETVAL(ifuo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifuo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == USHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifuo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifuo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifuo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifuo->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == CHAR) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifco->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifco->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == USHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifco->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifco->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifco->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifco->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == USHORT) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifso->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifso->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == USHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifso->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifso->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifso->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifso->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == INT) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifio->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifio->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == USHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifio->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifio->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifio->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifio->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == FLOAT) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(iffo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(iffo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == USHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(iffo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(iffo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(iffo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(iffo->grid(i,j,k));
		    }
		}
	    }
	}
    } else if (inVoxel == DOUBLE) {
	if (outVoxel == UCHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifu->grid(i,j,k) = SETVAL(ifdo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == CHAR) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifc->grid(i,j,k) = SETVAL(ifdo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == USHORT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifs->grid(i,j,k) = SETVAL(ifdo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == INT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifi->grid(i,j,k) = SETVAL(ifdo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == FLOAT) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			iff->grid(i,j,k) = SETVAL(ifdo->grid(i,j,k));
		    }
		}
	    }
	} else if (outVoxel == DOUBLE) {
	    for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
		    for (k=0; k<nz; k++) {
			ifd->grid(i,j,k) = SETVAL(ifdo->grid(i,j,k));
		    }
		}
	    }
	}
    }
}

void SFRGfile::setOutputFieldHandle() {
    if (outVoxel == UCHAR) ofh=ifu;    
    else if (outVoxel == CHAR) ofh=ifc;
    else if (outVoxel == USHORT) ofh=ifs;
    else if (outVoxel == INT) ofh=ifi;
    else if (outVoxel == FLOAT) ofh=iff;
    else if (outVoxel == DOUBLE) ofh=ifd;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log:
