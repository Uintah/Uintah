/*
 *  SFRG.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */
#include <Dataflow/Modules/Fields/SFRGfile.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Datatypes/NCScalarField.h>
#include <Packages/Uintah/Core/Datatypes/CCScalarField.h>
#include <Packages/Kurt/share/share.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldRG.h>

namespace Kurt {
using namespace SCIRun;

// ****************************************************************
// There must be some bug in the SGI compiler.  This code will have
// unresolved identifiers for the following types if I do not 
// actually instantiate them.  In the future, I would hope that the
// following four lines could be removed.
NCScalarField<int>* nci = new NCScalarField<int>;
CCScalarField<int>* cci = new CCScalarField<int>;
NCScalarField<double>* ncd = new NCScalarField<double>;
CCScalarField<double>* ccd = new CCScalarField<double>;
// ****************************************************************


// This code inherits from SFRGfile where VTYPE is 
// just a typedef to an int.  Values 0-6 are used in the parent.
const VTYPE NCDOUBLE = 7;
const VTYPE CCDOUBLE = 8;
const VTYPE NCINT = 9;
const VTYPE CCINT = 10;



class KurtSHARE SFRG : public SFRGfile {


  NCScalarField<int> *inci;
  CCScalarField<int> *icci;
  NCScalarField<double> *incd;
  CCScalarField<double> *iccd;

  virtual void setInputFieldVars();
  virtual void revoxelize();

public:
  SFRG(const clString& id);

  

  virtual ~SFRG();

};

extern "C" KurtSHARE Module* make_SFRG(const clString& id) {
  return scinew SFRG(id);
}

SFRG::SFRG(const clString& id)
  : SFRGfile( id )
{
}

SFRG::~SFRG(){
}


void SFRG::setInputFieldVars() {
  SFRGfile::setInputFieldVars();

  if(NCScalarField<double> *ncd =
     dynamic_cast<NCScalarField<double>*> (isf)){
    incd = ncd;
    inVoxel = NCDOUBLE;
    if(!haveOutVoxel){
      haveOutVoxelTCL.set(1);
      outVoxelTCL.set(0);
      outVoxel = DOUBLE;
    }
  }
  if(NCScalarField<int> *nci =
     dynamic_cast<NCScalarField<int>*> (isf)){
    inci = nci;
    inVoxel = NCINT;
    if(!haveOutVoxel){
      haveOutVoxelTCL.set(1);
      outVoxelTCL.set(2);
      outVoxel = INT;
    }
  }
  if(CCScalarField<double> *ccd =
     dynamic_cast<CCScalarField<double>*> (isf)){
    iccd = ccd;
    inVoxel = CCDOUBLE;
    if(!haveOutVoxel){
      haveOutVoxelTCL.set(1);
      outVoxelTCL.set(0);
      outVoxel = DOUBLE;
    }
  }
  if(CCScalarField<int> *cci =
     dynamic_cast<CCScalarField<int>*> (isf)){
    icci = cci;
    inVoxel = CCINT;
    if(!haveOutVoxel){
      haveOutVoxelTCL.set(1);
      outVoxelTCL.set(2);
      outVoxel = INT;
    }
  }
}

void SFRG::revoxelize() {
  if (inVoxel == outVoxel && !haveMinMax) return;

  int i,j,k;
  if (haveMinMax) {
    if (inVoxel == UCHAR || inVoxel == CHAR || inVoxel == SHORT ||
	inVoxel == USHORT || inVoxel == INT || inVoxel == FLOAT ||
	inVoxel == DOUBLE){
      SFRGfile::revoxelize();
      return;
    } 
    else if (inVoxel == NCDOUBLE) incd->get_minmax(Omin,Omax);
    else if (inVoxel == CCDOUBLE) iccd->get_minmax(Omin,Omax);
    else if (inVoxel == NCINT) inci->get_minmax(Omin,Omax);
    else iccd->get_minmax(Omin,Omax);
    
    Ospan = Omax-Omin;
  }

  cerr << "Cmin="<<Cmin<<"  Cmax="<<Cmax<<"\n";
  cerr << "Omin="<<Omin<<"  Ospan="<<Ospan<<"\n";
  cerr << "Nmin="<<Nmin<<"  Nspan="<<Nspan<<"\n";

  if (outVoxel == UCHAR) {
    ifuc = new ScalarFieldRGuchar;
    ifuc->resize(nx,ny,nz);
  } else if (outVoxel == CHAR) {
    ifc = new ScalarFieldRGchar;
    ifc->resize(nx,ny,nz);
  } else if (outVoxel == SHORT) {
    ifs = new ScalarFieldRGshort;
    ifs->resize(nx,ny,nz);
  } else if (outVoxel == USHORT) {
    ifus = new ScalarFieldRGushort;
    ifus->resize(nx,ny,nz);
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

  unsigned char tmpuchar;
  char tmpchar;
  short tmpshort;
  unsigned short tmpushort;
  double tmpdouble;
  float tmpfloat;
  int tmpint;
  if (inVoxel == NCDOUBLE) {
    if (outVoxel == UCHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpuchar = SETVAL(incd->grid(i,j,k));
	    ifuc->grid(i,j,k) = tmpuchar;
	  }
	}
      }
    } else if (outVoxel == CHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpchar = SETVAL(incd->grid(i,j,k));
	    ifc->grid(i,j,k) = tmpchar;
	  }
	}
      }
    } else if (outVoxel == SHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpshort = SETVAL(incd->grid(i,j,k));
	    ifs->grid(i,j,k) = tmpshort;
	  }
	}
      }
    } else if (outVoxel == USHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpushort = SETVAL(incd->grid(i,j,k));
	    ifus->grid(i,j,k) = tmpushort;
	  }
	}
      }
    } else if (outVoxel == INT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpint = SETVAL(incd->grid(i,j,k));
	    ifi->grid(i,j,k) = tmpint;
	  }
	}
      }
    } else if (outVoxel == FLOAT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpfloat = SETVAL(incd->grid(i,j,k));
	    iff->grid(i,j,k) = tmpfloat;
	  }
	}
      }
    } else if (outVoxel == DOUBLE) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpdouble = SETVAL(incd->grid(i,j,k));
	    ifd->grid(i,j,k) = tmpdouble;
	  }
	}
      }
    }
  } else if (inVoxel == CCDOUBLE) {
    if (outVoxel == UCHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpuchar = SETVAL(iccd->grid(i,j,k));
	    ifuc->grid(i,j,k) = tmpuchar;
	  }
	}
      }
    } else if (outVoxel == CHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpchar = SETVAL(iccd->grid(i,j,k));
	    ifc->grid(i,j,k) = tmpchar;
	  }
	}
      }
    } else if (outVoxel == SHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpshort = SETVAL(iccd->grid(i,j,k));
	    ifs->grid(i,j,k) = tmpshort;
	  }
	}
      }
    } else if (outVoxel == USHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpushort = SETVAL(iccd->grid(i,j,k));
	    ifus->grid(i,j,k) = tmpushort;
	  }
	}
      }
    } else if (outVoxel == INT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpint = SETVAL(iccd->grid(i,j,k));
	    ifi->grid(i,j,k) = tmpint;
	  }
	}
      }
    } else if (outVoxel == FLOAT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpfloat = SETVAL(iccd->grid(i,j,k));
	    iff->grid(i,j,k) = tmpfloat;
	  }
	}
      }
    } else if (outVoxel == DOUBLE) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpdouble = SETVAL(iccd->grid(i,j,k));
	    ifd->grid(i,j,k) = tmpdouble;
	  }
	}
      }
    }
  } else if (inVoxel == NCINT) {
    if (outVoxel == UCHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpuchar = SETVAL(inci->grid(i,j,k));
	    ifuc->grid(i,j,k) = tmpuchar;
	  }
	}
      }
    } else if (outVoxel == CHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpchar = SETVAL(inci->grid(i,j,k));
	    ifc->grid(i,j,k) = tmpchar;
	  }
	}
      }
    } else if (outVoxel == SHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpshort = SETVAL(inci->grid(i,j,k));
	    ifs->grid(i,j,k) = tmpshort;
	  }
	}
      }
    } else if (outVoxel == USHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpushort = SETVAL(inci->grid(i,j,k));
	    ifus->grid(i,j,k) = tmpushort;
	  }
	}
      }
    } else if (outVoxel == INT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpint = SETVAL(inci->grid(i,j,k));
	    ifi->grid(i,j,k) = tmpint;
	  }
	}
      }
    } else if (outVoxel == FLOAT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpfloat = SETVAL(inci->grid(i,j,k));
	    iff->grid(i,j,k) = tmpfloat;
	  }
	}
      }
    } else if (outVoxel == DOUBLE) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpdouble = SETVAL(inci->grid(i,j,k));
	    ifd->grid(i,j,k) = tmpdouble;
	  }
	}
      }
    }
  } else if (inVoxel == CCINT) {
    if (outVoxel == UCHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpuchar = SETVAL(icci->grid(i,j,k));
	    ifuc->grid(i,j,k) = tmpuchar;
	  }
	}
      }
    } else if (outVoxel == CHAR) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpchar = SETVAL(icci->grid(i,j,k));
	    ifc->grid(i,j,k) = tmpchar;
	  }
	}
      }
    } else if (outVoxel == SHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpshort = SETVAL(icci->grid(i,j,k));
	    ifs->grid(i,j,k) = tmpshort;
	  }
	}
      }
    } else if (outVoxel == USHORT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpushort = SETVAL(icci->grid(i,j,k));
	    ifus->grid(i,j,k) = tmpushort;
	  }
	}
      }
    } else if (outVoxel == INT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpint = SETVAL(icci->grid(i,j,k));
	    ifi->grid(i,j,k) = tmpint;
	  }
	}
      }
    } else if (outVoxel == FLOAT) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpfloat = SETVAL(icci->grid(i,j,k));
	    iff->grid(i,j,k) = tmpfloat;
	  }
	}
      }
    } else if (outVoxel == DOUBLE) {
      for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	  for (k=0; k<nz; k++) {
	    tmpdouble = SETVAL(icci->grid(i,j,k));
	    ifd->grid(i,j,k) = tmpdouble;
	  }
	}
      }
    }
  }
}
} // End namespace Kurt
