#ifndef SFRGFILE_H
#define SFRGFILE_H

#include <Core/Datatypes/ScalarField.h>
#include <Core/Containers/Array2.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Geometry/Point.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>


namespace SCIRun {
  class ScalarFieldRGdouble;
  class ScalarFieldRGfloat;
  class ScalarFieldRGint;
  class ScalarFieldRGshort;
  class ScalarFieldRGushort;
  class ScalarFieldRGchar;
  class ScalarFieldRGuchar;
  class ScalarFieldRGBase;
}

namespace SCIRun {






typedef int VTYPE;
const VTYPE DOUBLE = 0;
const VTYPE FLOAT = 1;
const VTYPE INT = 2;
const VTYPE SHORT = 3;
const VTYPE USHORT = 4;
const VTYPE CHAR = 5;
const VTYPE UCHAR = 6;


class CastField : public Module {

protected:

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
  ScalarFieldRGushort *ifus;
  ScalarFieldRGchar *ifc;
  ScalarFieldRGuchar *ifuc;
    
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
  virtual void setInputFieldVars();
  void setBounds();
  double SETVAL(double val){
    double v;
    if (!haveMinMax) return val;
    else v=(val-Omin)*Nspan/Ospan+Nmin;
    if (v<Cmin) return Cmin; else if (v>Cmax) return Cmax; else return v;
  }

  virtual void revoxelize();
  void setOutputFieldHandle();
public:
  CastField(const clString& id);
  virtual ~CastField();
  virtual void execute();
};


} // End namespace SCIRun

#endif
