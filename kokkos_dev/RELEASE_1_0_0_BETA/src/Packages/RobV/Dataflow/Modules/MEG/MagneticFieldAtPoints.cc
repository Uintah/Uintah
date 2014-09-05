/*
 *  MagneticFieldAtPoints.cc
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Packages/RobV/Core/Datatypes/MEG/VectorFieldMI.h>
#include <Core/Math/Trig.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
//#include <iostream.h>

namespace RobV {
using namespace SCIRun;

class MagneticFieldAtPoints : public Module {
  ColumnMatrixIPort* sourceLocationP;
  VectorFieldIPort* currentDensityFieldP;
  MatrixIPort* detectorPtsP;
  MatrixOPort* magneticFieldAtPointsP;
public:   
  MagneticFieldAtPoints(const clString& id);
  virtual ~MagneticFieldAtPoints();
  virtual void execute();

};

extern "C" Module* make_MagneticFieldAtPoints(const clString& id)
{
    return new MagneticFieldAtPoints(id);
}

MagneticFieldAtPoints::MagneticFieldAtPoints(const clString& id): Module("MagneticFieldAtPoints", id, Filter)
{

  sourceLocationP=new ColumnMatrixIPort(this, "SourceLocation", ColumnMatrixIPort::Atomic);
  add_iport(sourceLocationP);

  currentDensityFieldP=new VectorFieldIPort(this, "currentDensityField", VectorFieldIPort::Atomic);
  add_iport(currentDensityFieldP);

  detectorPtsP =new MatrixIPort(this, "DetectorLocations", MatrixIPort::Atomic);
  add_iport(detectorPtsP);

  // Create the output port
  magneticFieldAtPointsP =new MatrixOPort(this, "MagneticFieldAtPoints", MatrixIPort::Atomic);
  add_oport(magneticFieldAtPointsP);
}

MagneticFieldAtPoints::~MagneticFieldAtPoints()
{
}

void MagneticFieldAtPoints::execute() {

  VectorFieldHandle densityField;
  if (!currentDensityFieldP->get(densityField)) return;

  MatrixHandle detectLocsM;
  if (!detectorPtsP->get(detectLocsM)) return;

  ColumnMatrixHandle sourceLocsM;
//  if (!sourceLocationP->get(sourceLocsM)) return;
  sourceLocationP->get(sourceLocsM);

//  ColumnMatrix* sourceLocation = dynamic_cast<ColumnMatrix*>(sourceLocsM.get_rep());

  DenseMatrix* detectorPts = dynamic_cast<DenseMatrix*>(detectLocsM.get_rep());

  Vector magneticField;

  DenseMatrix* magneticMatrix = new DenseMatrix(detectorPts->nrows(),detectorPts->ncols());

  double errorTotal = 0.0;

  for (int i=0; i<detectorPts->ncols(); i++) {

    double x = (*detectorPts)[0][i];
    double y = (*detectorPts)[1][i];
    double z = (*detectorPts)[2][i];
     
    Point  pt (x,y,z);
    
    ((VectorFieldMI*)(densityField.get_rep()))->interpolate(pt, magneticField);

    Vector value(0,0,0);
#if 0
    // start of B(P) stuff
 
    double x1 = (*sourceLocation)[0];
    double y1 = (*sourceLocation)[1];
    double z1 = (*sourceLocation)[2];
    double theta = (*sourceLocation)[3];
    double phi = (*sourceLocation)[4];
    double magnitude = (*sourceLocation)[5];

    Point  pt2 (x1,y1,z1);
    
    Vector dir(Sin(theta)*Cos(phi), Sin(theta)*Sin(phi), Cos(theta));

    Vector P = dir*magnitude;

    Vector radius = pt - pt2; // detector - source

    Vector valuePXR = Cross(P,radius);
    double length = radius.length();

    double mu = 1.0;
    Vector value = (valuePXR/(length*length*length))*(mu/(4*M_PI));

    // end of B(P) stuff

    //double diff = value.length()-magneticField.length();
    //double diff = value.length()/magneticField.length();

    double diff = (fabs(value.length() - magneticField.length())) / value.length();

    errorTotal += diff*diff;
#endif
    (*magneticMatrix)[0][i] = magneticField.x() + value.x();
    (*magneticMatrix)[1][i] = magneticField.y() + value.y();
    (*magneticMatrix)[2][i] = magneticField.z() + value.z();
#if 0
    double angle = Dot(value,magneticField)/(value.length()*magneticField.length());

    double degrees = acos(angle) * 180/PI;

    cerr<<"Angle: "<<degrees<<"\n";

    cerr << "Total Mag: " << (magneticField+value).length() <<"\n";
    cerr << "B(P) Mag: " << value.length() <<"\n";
    cerr << "B(J) Mag: " << magneticField.length() <<"\n";

    cerr << "DetectorPoint: ("<<(*detectorPts)[0][i]<<","<<(*detectorPts)[1][i]<<","<<(*detectorPts)[2][i]<<")"<<"  Total MagneticField: ("<<(*magneticMatrix)[0][i]<<","<<(*magneticMatrix)[1][i]<<","<<(*magneticMatrix)[2][i]<<")  B(P): ("<<value.x()<<","<<value.y()<<","<<value.z()<<")  B(J): ("<<magneticField.x()<<","<<magneticField.y()<<","<<magneticField.z()<<")\n\n";
#endif    
  }
#if 0
  double rmsError = sqrt(errorTotal/(detectorPts->ncols()));

  double x = (*sourceLocation)[0];
  double y = (*sourceLocation)[1];
  double z = (*sourceLocation)[2];

  //  cerr << "Error for location ("<<x<<","<<y<<","<<z<<"): "<<rmsError<<"\n";
#endif
  MatrixHandle mH(magneticMatrix);
  magneticFieldAtPointsP->send(mH);
} // End namespace RobV
}
