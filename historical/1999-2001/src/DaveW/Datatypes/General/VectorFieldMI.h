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

#ifndef SCI_DaveW_Datatypes_VectorFieldMI_h
#define SCI_DaveW_Datatypes_VectorFieldMI_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Persistent/Pstreams.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::Array1;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

using namespace SCICore::Datatypes;

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

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/02 04:45:51  dmw
// magnetic field
//
//

#endif
