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

//    File   : Clipper.h
//    Author : Michael Callahan
//    Date   : September 2001

#if !defined(Clipper_h)
#define Clipper_h

#include <Core/Containers/Handle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Geometry/Transform.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class Clipper : public Datatype
{
public:
  virtual ~Clipper();

  virtual bool inside_p(const Point &p);
  virtual bool mesh_p() { return false; }

  static  PersistentTypeID type_id;
  void    io(Piostream &stream);
};


typedef Handle<Clipper> ClipperHandle;



class IntersectionClipper : public Clipper
{
private:
  ClipperHandle clipper0_;
  ClipperHandle clipper1_;

public:
  IntersectionClipper(ClipperHandle c0, ClipperHandle c1);

  virtual bool inside_p(const Point &p);

  static  PersistentTypeID type_id;
  void    io(Piostream &stream);
};


class UnionClipper : public Clipper
{
private:
  ClipperHandle clipper0_;
  ClipperHandle clipper1_;

public:
  UnionClipper(ClipperHandle c0, ClipperHandle c1);

  virtual bool inside_p(const Point &p);

  static  PersistentTypeID type_id;
  void    io(Piostream &stream);
};


class InvertClipper : public Clipper
{
private:
  ClipperHandle clipper_;

public:
  InvertClipper(ClipperHandle clipper);

  virtual bool inside_p(const Point &p);

  static  PersistentTypeID type_id;
  void    io(Piostream &stream);
};

  
class BoxClipper : public Clipper
{
private:
  Transform trans_;

public:
  BoxClipper(Transform &t);

  virtual bool inside_p(const Point &p);

  static  PersistentTypeID type_id;
  void    io(Piostream &stream);
};



template <class MESH>
class MeshClipper : public Clipper
{
private:
  LockingHandle<MESH> mesh_;

public:
  MeshClipper(LockingHandle<MESH> mesh) : mesh_(mesh) { 
    mesh->synchronize(Mesh::LOCATE_E);
  }

  virtual bool inside_p(const Point &p)
  {
    typename MESH::Elem::index_type indx;
    return mesh_->locate(indx, p);
  }
  virtual bool mesh_p() { return true; }

  void    io(Piostream &stream) {}
};



} // end namespace SCIRun

#endif // Clipper_h


