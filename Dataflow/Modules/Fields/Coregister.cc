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


/*
 *  Coregister.cc:  Coregister two point clouds
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Algorithms/Geometry/CoregPts.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class Coregister : public Module
{
public:
  Coregister(const string& id);
  virtual ~Coregister();
  virtual void execute();
};


extern "C" Module* make_Coregister(const string& id) {
  return new Coregister(id);
}


Coregister::Coregister(const string& id)
  : Module("Coregister", id, Filter, "Fields", "SCIRun")
{
}

Coregister::~Coregister()
{
}

void
Coregister::execute()
{
  FieldIPort *fixed = (FieldIPort *) get_iport("Fixed PointCloud");
  FieldIPort *mobile = (FieldIPort *) get_iport("Mobile PointCloud");

  FieldHandle fixedH, mobileH;
  PointCloud<double> *fixedPC, *mobilePC;
  PointCloudMeshHandle fixedM, mobileM;
  PointCloudMesh::Node::size_type nnodes;
  
  if (!fixed) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!mobile) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }

  if (!fixed->get(fixedH)) return;
  fixedPC = dynamic_cast<PointCloud<double> *>(fixedH.get_rep());
  if (!fixedPC) return;

  fixedM = fixedPC->get_typed_mesh();

  if (!mobile->get(mobileH)) return;
  mobilePC = dynamic_cast<PointCloud<double> *>(mobileH.get_rep());
  if (!mobilePC) return;

  mobileM = mobilePC->get_typed_mesh();

  fixedM->size(nnodes);
  if (nnodes < 3) {
    cerr << "Error: fixed PointCloud needs at least 3 input points.\n";
    return;
  }
  mobileM->size(nnodes);
  if (nnodes < 3) {
    cerr << "Error: mobile PointCloud needs at least 3 input points.\n";
    return;
  }

  MatrixOPort *omat = (MatrixOPort *)get_oport("Transform");
  if (!omat) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  CoregPtsAnalytic *coreg = scinew CoregPtsAnalytic;
  Array1<Point> fixedPts, mobilePts;
  Transform trans;

  PointCloudMesh::Node::iterator fni, mni;
  fixedM->begin(fni);
  mobileM->begin(mni);
  int i;
  for (i=0; i<3; i++) {
    Point p;
    fixedM->get_center(p, *fni);
    fixedPts.add(p);
    mobileM->get_center(p, *mni);
    mobilePts.add(p);
    ++fni;
    ++mni;
  }
  
  Array1<Point> transPts;

  coreg->setOrigPtsA(mobilePts);
  coreg->setOrigPtsP(fixedPts);
  coreg->getTrans(trans);
  double misfit;
  coreg->getMisfit(misfit);
  coreg->getTransPtsA(transPts);
  cerr << "Here's the misfit: "<<misfit<<"\n";
  DenseMatrix *dm = scinew DenseMatrix(trans);
  omat->send(MatrixHandle(dm));
}


} // End namespace SCIRun

