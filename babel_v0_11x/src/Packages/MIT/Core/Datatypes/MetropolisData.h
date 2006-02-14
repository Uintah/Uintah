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


#ifndef Datatypes_MetropolisData_h
#define Datatypes_MetropolisData_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Color.h>

namespace MIT {

using namespace SCIRun;

//
// Measurements
//

class SCICORESHARE Measurements : public Datatype {
 public:
  Array1<double> t;
  Array2<double> concentration;
  
 public:
  Measurements() : Datatype() {}
  virtual ~Measurements() {}
  
  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

};

typedef LockingHandle<Measurements> MeasurementsHandle;

// 
// Distribution
//

class SCICORESHARE Distribution : public Datatype {
 public:  
  Array2<double> sigma;
  double kappa;
  Array1<double> theta;

  Distribution() : Datatype() {}
  virtual ~Distribution() {}

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

typedef LockingHandle<Distribution> DistributionHandle;

//
// Results
//

class SCICORESHARE Results : public Datatype {
 public:  
  Array1< double > k_;
  Array1< Array1<double> > data_;
  Array1< Color.h> color_;

  Results() : Datatype() {}
  virtual ~Results() {}

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  int size() { return data_.size(); }
};

 typedef LockingHandle<Results> ResultsHandle;

}

#endif // Datatypes_MetropolisData_h

