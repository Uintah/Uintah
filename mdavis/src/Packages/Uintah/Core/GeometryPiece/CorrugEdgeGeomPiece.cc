#include <Packages/Uintah/Core/GeometryPiece/CorrugEdgeGeomPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

//////////
// Constructor : Initialize stuff
CorrugEdgeGeomPiece::CorrugEdgeGeomPiece(ProblemSpecP& ps)
{
  setName("corrugated");
  ps->require("xymin", d_xymin);
  ps->require("xymax", d_xymax);
  if ((d_xymax-d_xymin).length2() <= 0.0)
    SCI_THROW(ProblemSetupException("CorrugEdgeGeom: Check data in input"));
  cout << "xmin = " << d_xymin << " xmax = " << d_xymax << endl;

  ps->require("thickness", d_thickness);
  if (d_thickness <= 0.0)
    SCI_THROW(ProblemSetupException("CorrugEdgeGeom: Thickness <= 0"));
  cout << "thickness = " << d_thickness << endl;

  d_normal = Vector(0.0,0.0,1.0);
  ps->get("normal", d_normal);
  cout << "normal = " << d_normal << endl;

  d_edge = "x+";
  ps->require("corr_edge", d_edge);
  if (d_edge != "x+" && d_edge != "x-" && d_edge != "y+" && d_edge != "y-")
    SCI_THROW(ProblemSetupException("CorrugEdgeGeom: Unknown edge."));
  cout << "corr_edge = " << d_edge << endl;

  d_curve = "sin";
  ps->require("curve", d_curve);
  if (d_curve != "sin" && d_curve != "cos")
    SCI_THROW(ProblemSetupException("CorrugEdgeGeom: Unknown curve"));
  cout << "curve = " << d_curve << endl;

  ps->require("wavelength", d_wavelength);
  if (d_wavelength <= 0.0)
    SCI_THROW(ProblemSetupException("CorrugEdgeGeom: Wavelength <= 0.0"));
  cout << "wavelength = " << d_wavelength << endl;

  ps->require("amplitude", d_amplitude);
  if (d_amplitude <= 0.0)
    SCI_THROW(ProblemSetupException("CorrugEdgeGeom: Amplitude <= 0.0"));
  cout << "amplitude = " << d_amplitude << endl;
}

//////////
// Destructor
CorrugEdgeGeomPiece::~CorrugEdgeGeomPiece()
{
}


CorrugEdgeGeomPiece* CorrugEdgeGeomPiece::clone()
{
  return scinew CorrugEdgeGeomPiece(*this);
}


/////////////////////////////////////////////////////////////////////////////
/*! Find if a point is inside the plate with corrugated edge */
/////////////////////////////////////////////////////////////////////////////
bool 
CorrugEdgeGeomPiece::inside(const Point& p) const
{
  bool isInside = false;

  cout << "CorrugEdgeGeomPiece:inside(p) not yet implemented." << endl;
  return isInside;

}

/////////////////////////////////////////////////////////////////////////////
/*! Find the bounding box for the plate */
/////////////////////////////////////////////////////////////////////////////
Box 
CorrugEdgeGeomPiece::getBoundingBox() const
{
  Point lo = d_xymin;
  Vector top = d_xymax.asVector() + d_normal*d_thickness;
  Point hi(top.x(), top.y(), top.z());

  return Box(lo,hi);
}

//////////////////////////////////////////////////////////////////////////
/* Create particles */
//////////////////////////////////////////////////////////////////////////
int 
CorrugEdgeGeomPiece::createPoints()
{
  double lambda = d_wavelength;
  double a = d_amplitude;
  double L = 0.0; double W = 0.0; double T = d_thickness;

  cout << "lambda = " << lambda << " a = " << a << endl;
  // Decide whether sin or cos curve is to be used
  double sinFactor = 0.0;
  double cosFactor = 0.0;
  if (d_curve == "sin") sinFactor = 1.0;
  else cosFactor = 1.0;
  cout << "sinFactor = " << sinFactor << " cosFactor = " << cosFactor << endl;

  // Decide the Length (corrugated dimension) and 
  // Width (uncorrugated dimension)
  if (d_edge == "x+" || d_edge == "x-") {
    W = d_xymax.x() - d_xymin.x();
    L = d_xymax.y() - d_xymin.y();
  }
  else {
    L = d_xymax.x() - d_xymin.x();
    W = d_xymax.y() - d_xymin.y();
  }
  cout << "L = " << L << " W = " << W << endl;
  double dx = d_dx;
  int nL = (int) ceil(L/dx); double dxL = L/(double) nL;
  int nW = (int) ceil(W/dx); double dxW = W/(double) nW;
  int nT = (int) ceil(T/dx); double dxT = T/(double) nT;
  double da = a/(double) nW;
  cout << "dx = " << dx << " da = " << da << endl;
  cout << "nL = " << nL << " nW = " << nW << " nT = " << nT << endl;

  double xstart = 0.0;
  double ystart = 0.0;
  double xsign = 1.0;
  double ysign = 1.0;
  if (d_edge == "x-") {
    xstart = d_xymin.x();
    ystart = d_xymin.y();;
    xsign = 1.0;
    ysign = 1.0;
  } else if (d_edge == "x+") {
    xstart = d_xymin.x()+W;
    ystart = d_xymin.y();
    xsign = -1.0;
    ysign = 1.0;
  } else if (d_edge == "y-") {
    xstart = d_xymin.x();
    ystart = d_xymin.y();
    xsign = 1.0;
    ysign = 1.0;
  } else if (d_edge == "y+") {
    xstart = d_xymin.x();
    ystart = d_xymin.y()+W;
    xsign = 1.0;
    ysign = -1.0;
  }
  if (d_edge == "x+" || d_edge == "x-") {
    double zz = d_xymin.z();
    for (int kk = 0; kk < nT+1; ++kk) {
      a = d_amplitude;
      double xx = xstart;
      for (int jj = 0; jj < nW+1; ++jj) {
        double yy = ystart;
        for (int ii = 0; ii < nL+1; ++ii) {
          double x = xx + a*sin(2.0*yy*M_PI/lambda)*sinFactor +
                          a*cos(2.0*yy*M_PI/lambda)*cosFactor;
          d_points.push_back(Point(x,yy,zz));
          d_volume.push_back(dx*dx*dx);
          yy += ysign*dxL;
        }
        xx += xsign*dxW;
        a -= da;
      }
      zz += dxT;
    }
  } else {
    double zz = d_xymin.z();
    for (int kk = 0; kk < nT+1; ++kk) {
      a = d_amplitude;
      double yy = ystart;
      for (int jj = 0; jj < nW+1; ++jj) {
        double xx = xstart;
        for (int ii = 0; ii < nL+1; ++ii) {
          double y = yy + a*sin(2.0*xx*M_PI/lambda)*sinFactor +
                          a*cos(2.0*xx*M_PI/lambda)*cosFactor;
          d_points.push_back(Point(xx,y,zz));
          d_volume.push_back(dx*dx*dx);
          xx += xsign*dxL;
        }
        yy += ysign*dxW;
        a -= da;
      }
      zz += dxT;
    }
  }
  cout << "Number of points = " << d_points.size() << endl;
  return d_points.size();
}

