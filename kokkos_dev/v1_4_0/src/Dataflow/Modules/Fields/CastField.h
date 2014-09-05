/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef SFRGFILE_H
#define SFRGFILE_H

#include <Core/Datatypes/Field.h>
#include <Core/Containers/Array2.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/Point.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>

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

  FieldIPort* iField;
  FieldOPort* oField;

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

#if 0
  FieldRGdouble *ifd;
  FieldRGfloat *iff;
  FieldRGint *ifi;
  FieldRGshort *ifs;
  FieldRGushort *ifus;
  FieldRGchar *ifc;
  FieldRGuchar *ifuc;
#endif
    
  FieldHandle ifh;
  //  FieldRGBase *isf;
  FieldHandle ofh;

  GuiInt haveMinMaxTCL;
  GuiInt haveOutVoxelTCL;
  GuiInt haveBBoxTCL;
  GuiInt outVoxelTCL;
  GuiString NminTCL;
  GuiString NmaxTCL;
  GuiString CminTCL;
  GuiString CmaxTCL;
  GuiString minOutTCLX;
  GuiString minOutTCLY;
  GuiString minOutTCLZ;
  GuiString maxOutTCLX;
  GuiString maxOutTCLY;
  GuiString maxOutTCLZ;

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
  CastField(const string& id);
  virtual ~CastField();
  virtual void execute();
};


} // End namespace SCIRun

#endif
