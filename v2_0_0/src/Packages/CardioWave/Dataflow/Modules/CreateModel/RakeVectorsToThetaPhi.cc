/*
 *  RakeVectorsToThetaPhi.cc
 *
 *   Written by:
 *   David Weinstein
 *   April 2003
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/ColumnMatrix.h>

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE RakeVectorsToThetaPhi : public Module {
public:
  RakeVectorsToThetaPhi(GuiContext *context);
  virtual ~RakeVectorsToThetaPhi();
  virtual void execute();
};


DECLARE_MAKER(RakeVectorsToThetaPhi)


RakeVectorsToThetaPhi::RakeVectorsToThetaPhi(GuiContext *context)
  : Module("RakeVectorsToThetaPhi", context, Source, "CreateModel", "CardioWave")
{
}

RakeVectorsToThetaPhi::~RakeVectorsToThetaPhi(){
}

void RakeVectorsToThetaPhi::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *imesh = (FieldIPort*)get_iport("Vectors");
  if (!imesh) {
    error("Unable to initialize iport 'Vectors'.");
    return;
  }
  
  MatrixOPort *otheta = (MatrixOPort*)get_oport("Theta");
  if (!otheta) {
    error("Unable to initialize oport 'Theta'.");
    return;
  }

  MatrixOPort *ophi = (MatrixOPort*)get_oport("Phi");
  if (!ophi) {
    error("Unable to initialize oport 'Phi'.");
    return;
  }

  FieldHandle meshH;
  if (!imesh->get(meshH) || 
      !meshH.get_rep())
    return;

  PointCloudField<Vector> *vecfld = dynamic_cast<PointCloudField<Vector> *>(meshH.get_rep());
  if (!vecfld)
  {
    error("Input field wasn't a PointCloudField<Vector>.");
    return;
  }
  
  if (vecfld->data_at() != Field::NODE) {
    error("Input field doesn't have data at Nodes.");
    return;
  }

  int npts = vecfld->fdata().size();
  if (npts < 2) {
    error("Input field must have at least two points in order to ascertain direction");
    return;
  }

  ColumnMatrix *thetas = new ColumnMatrix(npts);
  ColumnMatrix *phis = new ColumnMatrix(npts);

  PointCloudMeshHandle pcmH = vecfld->get_typed_mesh();
  PointCloudMesh::Node::iterator nb, ne;
  pcmH->begin(nb);
  pcmH->end(ne);
  Point p1, p2;
  pcmH->get_point(p1, *nb);
  ++nb;
  pcmH->get_point(p2, *nb);
  Vector v = p2-p1;
  v.normalize();
  Vector thetaVec, phiVec;
  if (v.z()>0.999999) {
    thetaVec=Vector(0,1,0);
    phiVec=Vector(1,0,0);
  } else {
    thetaVec=Vector(0,0,1);
    phiVec=Cross(thetaVec, v);
    phiVec.normalize();
    thetaVec=Cross(v, phiVec);
  }
  double thetaDot, phiDot, vDot;
  Vector vec=vecfld->fdata()[0];
  thetaDot=Dot(thetaVec, vec);
  phiDot=Dot(phiVec, vec);
  vDot=Dot(v, vecfld->fdata()[0]);
  cerr << "v="<<v<<" thetaVec="<<thetaVec<<" phiVec="<<phiVec<<"\n";

  double thAngle=(180./M_PI)*atan2(thetaDot, phiDot);
  if (thAngle>90) thAngle-=180.;
  else if (thAngle<-90) thAngle+=180.;
  (*thetas)[0]= thAngle;
  double phAngle=(180./M_PI)*atan2(vDot, thetaDot);
  if (phAngle>90) phAngle-=180.;
  else if (phAngle<-90) phAngle+=180.;
  (*phis)[0]=phAngle;
  cerr << "vec[0]="<<vec<<" thAngle="<<thAngle<<" phAngle="<<phAngle<<"\n";

  int cnt=1;
  while(nb != ne) {
    vec=vecfld->fdata()[cnt];
    thetaDot=Dot(thetaVec, vec);
    phiDot=Dot(phiVec, vec);
    vDot=Dot(v, vecfld->fdata()[cnt]);
    thAngle=(180./M_PI)*atan2(thetaDot, phiDot);    
    if ((thAngle-(*thetas)[cnt-1])>90) thAngle-=180.;
    else if ((thAngle-(*thetas)[cnt-1])<-90) thAngle+=180.;
    (*thetas)[cnt]=thAngle;
    phAngle=(180./M_PI)*atan2(vDot, thetaDot);
    if ((phAngle-(*phis)[cnt-1])>90) phAngle-=180.;
    else if ((phAngle-(*phis)[cnt-1])<-90) phAngle+=180.;
    (*phis)[cnt]=phAngle;
    cerr << "vec["<<cnt<<"]="<<vec<<" thAngle="<<thAngle<<" phAngle="<<phAngle<<"\n";
    ++cnt;
    ++nb;
  }
  MatrixHandle thH(thetas);
  MatrixHandle phH(phis);
  otheta->send(thetas);
  ophi->send(phis);
}
} // End namespace CardioWave
