/*
 * Hacked code for communicating over ports in
 * SCIRun
 *
 * Anonymous (ok, Peter-Pike Sloan...)
 */

#ifndef KludgeMessage_H_
#define KludgeMessage_H_

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>

#include <Geometry/Point.h>

class KludgeMessage;
typedef LockingHandle<KludgeMessage> KludgeMessageHandle;

struct SourceRecord {
  Point  loc;
  double theta;
  double phi;

  double v; // volate source

  void CreateDipole(Point &p0, Point &p1, double &v0, double &v1);
  
  // these are used for debugging purposes mostly
  
  double err;
};

class KludgeMessage : public Datatype {
public:
  
  // these can be empty - that means you are
  // just sending the scalp potentials

  Array1<double> src_mag;  // contribution of source
  Array1<Point>  src_loc;  // location of source
  
  // surf_pots points on the scalp
  // the other 2 arrays are 3*surf_pots.size() in length
  // and contain indices and weights for the sample
  // points

  Array1<double> surf_pots; // potentials at surface locations
  Array1<int>    surf_pti;  // point indeces for surface
  Array1<double> surf_wts;  // barycentric coordinates on a face...

  // this is a hack - just use this record to create a source

  Array1<SourceRecord> src_recs;

public:
  KludgeMessage();
  KludgeMessage(KludgeMessage&);
  
  virtual KludgeMessage* clone();
  virtual ~KludgeMessage();

  // Persistent representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

class AmoebaMessage;
typedef LockingHandle<AmoebaMessage> AmoebaMessageHandle;

// maybe add some more stuff later...

struct AmoebaRecord {
  Array1< SourceRecord > sources; // nodes on the graph
  int generation;                 // incremented when modified...

  AmoebaRecord& operator=(AmoebaRecord& cp) 
  { 
    sources = cp.sources;
    generation = cp.generation;
    return *this;
  };

};

class AmoebaMessage : public Datatype {
public:
  Array1< AmoebaRecord > amoebas;
  
public:
  AmoebaMessage();
  AmoebaMessage(AmoebaMessage&);
  
  virtual AmoebaMessage* clone();
  virtual ~AmoebaMessage();

  // Persistent representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

void Pio(Piostream&, AmoebaRecord&);
void Pio(Piostream&, SourceRecord&);

#endif
