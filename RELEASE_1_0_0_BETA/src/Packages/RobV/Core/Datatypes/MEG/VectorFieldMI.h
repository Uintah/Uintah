/*
 *  VectorFieldMI.h
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_VectorFieldMI_h
#define SCI_Packages_DaveW_Datatypes_VectorFieldMI_h 1

#include <Core/Containers/Array1.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Persistent/Pstreams.h>

namespace RobV {
using namespace SCIRun;


class VectorFieldMI : public VectorField {

public:   

  VectorFieldMI();
  VectorFieldMI(VectorField* vf);
  virtual ~VectorFieldMI();
  virtual int interpolate(const Point& p, Vector& value);
  virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);
  virtual void get_boundary_lines(Array1<Point>& lines);
  virtual void io(Piostream&);
  virtual VectorField* clone();
  virtual void compute_bounds();

  static PersistentTypeID type_id;

  MeshHandle getMesh();
  
private:
  VectorField* field;
  Array1<Vector> interp;
  Array1<double> vol;
  Array1<Point> centro;
  int nelems;

};
} // End namespace RobV



#endif
