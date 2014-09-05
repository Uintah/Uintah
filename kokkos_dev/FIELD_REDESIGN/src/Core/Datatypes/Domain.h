// Domain.h - Manages sets of Attributes and Geometries
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_Domain_h
#define SCI_project_Domain_h 1

#include <SCICore/Datatypes/Field.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Datatypes/Attrib.h>



#include <vector>
#include <map>
#include <string>

namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Domain;
typedef LockingHandle<Domain> DomainHandle;

class SCICORESHARE Domain:public Datatype{
public:

  /////////
  // Maps names to geometries (and attributes)
  map<string, GeomHandle> geoms; 
  map<string, AttribHandle> attribs;

  ////////
  // constructors, destructor
  Domain();
  Domain(const Domain&);
  ~Domain();

  // Methods for adding/accessing geometries in domain
  void addGeom(GeomHandle);
  GeomHandle getGeom(string);
  
  // add a geometry that shares nodes with another geometry
  void addGeomShared(GeomHandle item, GeomHandle nodes);
    
  // Methods for adding/accessing attributes in domain
  void addAttrib(AttribHandle);
  AttribHandle getAttrib(string);
  
  // returns all of the attributes associated with the given geometry
  vector<AttribHandle> GetAttrib(const GeomHandle&);  

  inline map<string, GeomHandle> GetGeom() {return geoms;};

  // inline void clear_geoms(){geoms.clear();};
  // inline void clear_attribs(){attribs.clear();};
  
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // end namespace datatype
} // end namespace scicore
#endif
