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
 *  ApplyFEMCurrentSource.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;

namespace BioPSE {

using namespace SCIRun;

class ApplyFEMCurrentSource : public Module {
  FieldIPort* inmesh;
  MatrixIPort * isource;
  MatrixIPort * isrcmat;  
  MatrixIPort* irhs;
  MatrixIPort* imap;
  MatrixOPort* orhs;
  MatrixOPort* oidx;
  GuiString sourceNodeTCL;
  GuiString sinkNodeTCL;
  GuiString modeTCL;
  int loc;
  int gen;
public:
  ApplyFEMCurrentSource(const clString& id);
  virtual ~ApplyFEMCurrentSource();
  virtual void execute();
};

extern "C" Module* make_ApplyFEMCurrentSource(const clString& id)
{
  return scinew ApplyFEMCurrentSource(id);
}

ApplyFEMCurrentSource::ApplyFEMCurrentSource(const clString& id)
  : Module("ApplyFEMCurrentSource", id, Filter), 
  sourceNodeTCL("sourceNodeTCL", id, this),
  sinkNodeTCL("sinkNodeTCL", id, this),
  modeTCL("modeTCL", id, this)
{
  // Create the input port
  inmesh = scinew FieldIPort(this, "TetVol", FieldIPort::Atomic);
  add_iport(inmesh);
  isource=scinew MatrixIPort(this, "Source", MatrixIPort::Atomic);
  add_iport(isource);
  irhs=scinew MatrixIPort(this, "Input RHS", MatrixIPort::Atomic);
  add_iport(irhs);
  isrcmat=scinew MatrixIPort(this, "SourceMat", MatrixIPort::Atomic);
  add_iport(isrcmat);
  imap=scinew MatrixIPort(this, "Electrode Map", MatrixIPort::Atomic);
  add_iport(imap);
  
  // Create the output ports
  orhs=scinew MatrixOPort(this,"Output RHS", MatrixIPort::Atomic);
  add_oport(orhs);
  oidx=scinew MatrixOPort(this,"Elem Index", MatrixIPort::Atomic);
  add_oport(oidx);
}

ApplyFEMCurrentSource::~ApplyFEMCurrentSource()
{
}

void ApplyFEMCurrentSource::execute()
{
  FieldHandle mesh;
  //     cerr << "ApplyFEMCurrentSource: about to read inputs...\n";
  if (!inmesh->get(mesh) || !mesh.get_rep()) return;
  
  MatrixHandle rhsh;
#if 0 // FIX_ME mesh to TetVol
  ColumnMatrix* rhs = scinew ColumnMatrix(mesh->nodes.size());
  rhsh=rhs;
  MatrixHandle rhshIn;
  ColumnMatrix* rhsIn;
  // if the user passed in a vector the right size, copy it into ours
  if (irhs->get(rhshIn) && 
      (rhsIn=dynamic_cast<ColumnMatrix*>(rhshIn.get_rep())) && 
      (rhsIn->nrows()==mesh->nodes.size()))
    for (int i=0; i<mesh->nodes.size(); i++) (*rhs)[i]=(*rhsIn)[i];
  else
    rhs->zero();
  
  clString mode=modeTCL.get();
  clString sourceNodeS=sourceNodeTCL.get();
  clString sinkNodeS=sinkNodeTCL.get();

//  cerr << "mode="<<mode<<"  sourceNodeS="<<sourceNodeS<<"  sinkNodeS="<<sinkNodeS<<"\n";

  int sourceNode=-1;
  int sinkNode=-1;
  if (mode=="electrodes") {
    if (!sourceNodeS.get_int(sourceNode) ||
	!sinkNodeS.get_int(sinkNode)) {
      cerr << "ApplyFEMCurrentSource error - need source/sink pair.\n";
      return;
    }
    MatrixHandle imapH;
    ColumnMatrix *imapp;
    if (!imap->get(imapH) || 
	!(imapp=dynamic_cast<ColumnMatrix*>(imapH.get_rep()))) {
      cerr << "ApplyFEMCurrentSource error - need electrode map.\n";
      return;
    }
    if (sourceNode < 1 || sinkNode < 1 || sourceNode > imapp->nrows() ||
	sinkNode > imapp->nrows()) {
      cerr << "ApplyFEMCurrentSource error - nodes must be within range [0,"<<imapp->nrows()<<"]\n";
      return;
    }
    (*rhs)[(*imapp)[sourceNode-1]]=1;
    (*rhs)[(*imapp)[sinkNode-1]]=-1;
    cerr << "ApplyFEMCurrentSource - using source/sink pair "<<(*imapp)[sourceNode-1]<<"/"<<(*imapp)[sinkNode-1]<<"\n";
    orhs->send(rhsh);    
    return;
  }
      
  MatrixHandle mh=0;
  ColumnMatrix *mp=0;
  
  MatrixHandle mmh=0;
  Matrix* mmp=0;

  if ((!isource->get(mh) || !(mp=dynamic_cast<ColumnMatrix*>(mh.get_rep()))) &&
      (!isrcmat->get(mmh) || !(mmp=mmh.get_rep()))) return; 
  
  if ((mp && mp->nrows()<6) || (mmp && mmp->ncols()<6)) {
    cerr << "ApplyFEMCurrentSource error - every dipole source must have at least six terms\n";
    return;
  }
  
  Vector dir;
  Point p;
  int nsrcs=1;
  if (mmp) nsrcs=mmp->nrows();
  
  for (int ii=0; ii<nsrcs; ii++) {
    if (mp) {
      dir=Vector((*mp)[3], (*mp)[4], (*mp)[5]);
      p=Point((*mp)[0], (*mp)[1], (*mp)[2]);
      if (mp->nrows() == 7) loc=(int)((*mp)[6]);
    } else {
      dir=Vector((*mmp)[ii][3], (*mmp)[ii][4], (*mmp)[ii][5]);
      p=Point((*mmp)[ii][0], (*mmp)[ii][1], (*mmp)[ii][2]);
      if (mmp->nrows() == 7) loc=(int)((*mmp)[ii][6]);
    }
    if (mesh->locate(&loc, p)) {
      //	 cerr << "ApplyFEMCurrentSource: Found Dipole in element "<<loc<<"\n";
      double s1, s2, s3, s4;
      
      // use these next six lines if we're using a dipole
      Vector g1, g2, g3, g4;
      mesh->get_grad(mesh->elems[loc], p, g1, g2, g3, g4);
      
      //	 cerr << "ApplyFEMCurrentSource :: p="<<p<<"  dir="<<dir<<"\n";
      //	 cerr << "Dir="<<dir<<"  g1="<<g1<<"  g2="<<g2<<"\n";
      //	 cerr << "g3="<<g3<<"  g4="<<g4<<"\n";
      s1=Dot(g1,dir);
      s2=Dot(g2,dir);
      s3=Dot(g3,dir);
      s4=Dot(g4,dir);
      
      // use this next line if we're using a monopole
      // mesh->get_interp(mesh->elems[loc], p, s1, s2, s3, s4);
      
      // s1*=src; s2*=src; s3*=src; s4*=src;
      int i1, i2, i3, i4;
      i1=mesh->elems[loc]->n[0];
      i2=mesh->elems[loc]->n[1];
      i3=mesh->elems[loc]->n[2];
      i4=mesh->elems[loc]->n[3];
      (*rhs)[i1]+=s1;
      (*rhs)[i2]+=s2;
      (*rhs)[i3]+=s3;
      (*rhs)[i4]+=s4;
      
#if 0
      cerr << "ApplyFEMCurrentSource :: Here's the RHS vector: ";
      for (int jj=0; jj<mesh->nodes.size(); jj++) 
	cerr << (*rhs)[jj]<<" ";
      cerr << "\n";
      cerr << "ApplyFEMCurrentSource :: Here's the dipole: ";
      if (mmp)
	for (jj=0; jj<5; jj++) 
	  cerr << (*mmp)[ii][jj]<<" ";
      else
	for (jj=0; jj<5; jj++) 
	  cerr << (*mp)[jj]<<" ";
      cerr << "\n";
#endif
      
    } else {
      loc=0;
      dir=Vector(0,0,0);
      cerr << "Dipole: "<<p<<" not located within mesh!\n";
    }
    gen=rhsh->generation;
    //     cerr << "ApplyFEMCurrentSource: about to send result...\n";
    orhs->send(rhsh);
    //     cerr << "ApplyFEMCurrentSource: sent result!\n";
    
    ColumnMatrix *idxvec = new ColumnMatrix(6);
    idxvec->zero();
    (*idxvec)[0]=loc*3;
    (*idxvec)[1]=dir.x();
    (*idxvec)[2]=loc*3+1;
    (*idxvec)[3]=dir.y();
    (*idxvec)[4]=loc*3+2;
    (*idxvec)[5]=dir.z();
    oidx->send(MatrixHandle(idxvec));
  }
#endif
}
} // End namespace BioPSE




