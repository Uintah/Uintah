#ifndef SFRGFILE_H
#define SFRGFILE_H

#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Geometry/Point.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>


namespace SCICore {
  namespace Datatypes {
    class ScalarFieldRGdouble;
    class ScalarFieldRGfloat;
    class ScalarFieldRGint;
    class ScalarFieldRGshort;
    class ScalarFieldRGushort;
    class ScalarFieldRGchar;
    class ScalarFieldRGuchar;
    class ScalarFieldRGBase;
  }
}

namespace PSECommon{
namespace Modules {

using namespace PSECore::Dataflow;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using SCICore::Geometry::Point;

using SCICore::Datatypes::ScalarFieldRGdouble;
using SCICore::Datatypes::ScalarFieldRGfloat;
using SCICore::Datatypes::ScalarFieldRGint;
using SCICore::Datatypes::ScalarFieldRGshort;
using SCICore::Datatypes::ScalarFieldRGushort;
using SCICore::Datatypes::ScalarFieldRGchar;
using SCICore::Datatypes::ScalarFieldRGuchar;
using SCICore::Datatypes::ScalarFieldHandle;
using SCICore::Datatypes::ScalarFieldRGBase;

using PSECore::Datatypes::ScalarFieldIPort;
using PSECore::Datatypes::ScalarFieldOPort;



typedef int VTYPE;
const VTYPE DOUBLE = 0;
const VTYPE FLOAT = 1;
const VTYPE INT = 2;
const VTYPE SHORT = 3;
const VTYPE USHORT = 4;
const VTYPE CHAR = 5;
const VTYPE UCHAR = 6;


class SFRGfile : public Module {

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
  SFRGfile(const clString& id);
  virtual ~SFRGfile();
  virtual void execute();
};


} // end namespace Modules
} // end namespace PSECommon

#endif
