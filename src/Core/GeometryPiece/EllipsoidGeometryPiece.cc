/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/GeometryPiece/EllipsoidGeometryPiece.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <cmath>

using namespace Uintah;
using namespace std;

const string EllipsoidGeometryPiece::TYPE_NAME = "ellipsoid";
const double EllipsoidGeometryPiece::geomTol = 1.0e-12;

EllipsoidGeometryPiece::EllipsoidGeometryPiece(ProblemSpecP& ps)
{

  name_ = "Unnamed " + TYPE_NAME + " from PS";

  Vector zero = Vector(0.,0.,0.);

  ps->require("origin",d_origin);
  // Get Vector axes
  ps->getWithDefault("v1",d_v1,     zero);
  ps->getWithDefault("v2",d_v2,     zero);
  ps->getWithDefault("v3",d_v3,     zero);

  // Get orthagonal axes
  ps->getWithDefault("rx",d_r1, 0.0);
  ps->getWithDefault("ry",d_r2, 0.0);
  ps->getWithDefault("rz",d_r3, 0.0);
  
  // Run helper function to determine if inputs are correct
  initializeEllipsoidData();
}

EllipsoidGeometryPiece::EllipsoidGeometryPiece(const Point& origin,
                                               double radx, double rady, double radz )
{
  d_origin = origin;
  d_r1 = radx;
  d_r2 = rady;
  d_r3 = radz;
  
  // Make sure there is no uninitialized variables going into initialization routine
  d_v1 = Vector(1.0, 0.0, 0.0);
  d_v2 = Vector(0.0, 1.0, 0.0);
  d_v3 = Vector(0.0, 0.0, 1.0);
  
  // Run helper function to determine if inputs are correct
  initializeEllipsoidData();
}

EllipsoidGeometryPiece::EllipsoidGeometryPiece(const Point& origin,
                                               Vector one, Vector two, Vector three )
{
  d_origin = origin;
  d_v1 = one;
  d_v2 = two;
  d_v3 = three;
  
  // Make sure there is no uninitialized variables going into initialization routine
  d_r1 = 0.0;
  d_r2 = 0.0;
  d_r3 = 0.0;
  
  // Run helper function to determine if inputs are correct
  initializeEllipsoidData();
}

EllipsoidGeometryPiece::~EllipsoidGeometryPiece()
{
}

void EllipsoidGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("origin",d_origin);
  if(xyzAligned) { // only need to output radii
    ps->appendElement("rx",    d_r1);
    ps->appendElement("ry",    d_r2);
    ps->appendElement("rz",    d_r3);
  } else {
    ps->appendElement("v1",    d_v1);
    ps->appendElement("v2",    d_v2);
    ps->appendElement("v3",    d_v3);
  }
}

GeometryPieceP EllipsoidGeometryPiece::clone() const
{
  return scinew EllipsoidGeometryPiece(*this);
}

bool EllipsoidGeometryPiece::inside(const Point& p) const
{
  // This can be sped up, but this is simple.
  Vector pTransformed = p-d_origin;
  if (Dot(pTransformed,d_m3E*(pTransformed)) - 1.0 < geomTol)
  {
    return true;
  }
  return false;
  
}

Box EllipsoidGeometryPiece::getBoundingBox() const
{

  Point minCorner(d_origin-boundOffset);
  Point maxCorner(d_origin+boundOffset);

  return Box(minCorner,maxCorner);
}

void EllipsoidGeometryPiece::initializeEllipsoidData()
{

  // Linear algebraic representation using formula:
  // E := { x | (x-c)^T US^2U^T (x-c) <= 1.0 }

  // U is vector whose columns are unit vector representations of the
  //   directions of the ellipsoid axes.
  // S is the diagonal matrix with the ii component equivalent to the
  //   inverse squared magnitude of the ellipsoid axes
  //   (i.e. S_11 := 1.0/||d_v1||^2 )
  // determine whether input is from vector or double

  if (    d_v1.length() > geomTol
      &&  d_v2.length() > geomTol
      &&  d_v3.length() > geomTol)
  {
    // Check for orthagonality
    if((fabs(Dot(d_v1,d_v2)) >= 1e-12) ||
       (fabs(Dot(d_v2,d_v3)) >= 1e-12) ||
       (fabs(Dot(d_v3,d_v1)) >= 1e-12) )
    {
      throw ProblemSetupException("Input File Error: (Ellipsoid initialization) input vectors (v1,v2,v3) are not orthagonal to within 1e-12 or each other", __FILE__, __LINE__, false);
    }

    // Express in compact matrix representation
    Vector u1 = d_v1;
    d_r1 = u1.normalize();
    Vector u2 = d_v2;
    d_r2 = u2.normalize();
    Vector u3 = d_v3;
    d_r3 = u3.normalize();
    // Initialize variables for rotation

    Matrix3 U(u1[0], u2[0], u3[0],
              u1[1], u2[1], u3[1],
              u1[2], u2[2], u3[2]);

    Matrix3 S(1.0/(d_r1*d_r1), 0.0            , 0.0,
              0.0            , 1.0/(d_r2*d_r2), 0.0,
              0.0            , 0.0            , 1.0/(d_r3*d_r3) );
    Matrix3 SInv(d_r1*d_r1,       0.0,       0.0,
                       0.0, d_r2*d_r2,       0.0,
                       0.0,       0.0, d_r3*d_r3);

    d_m3E = U*S*U.Transpose();
    Matrix3 d_m3E_Inv = U*SInv*U.Transpose();
    boundOffset = Vector(sqrt(d_m3E_Inv(0,0)),
                         sqrt(d_m3E_Inv(1,1)),
                         sqrt(d_m3E_Inv(2,2)));

  } else if(d_r1 > 0.0 && d_r2 > 0.0 && d_r3 > 0.0) {
    // create vector representation along the cartesian axes
    d_v1 = *(new Vector(d_r1, 0.0, 0.0));
    d_v2 = *(new Vector(0.0, d_r2, 0.0));
    d_v3 = *(new Vector(0.0, 0.0, d_r3));
    
    // set flag such that rotation doesnt need to occur in inside()
    xyzAligned = true;
      
  } else {
      throw ProblemSetupException("Input File Error: (Ellipsoid initialization) input radii (rx,ry,rz) must have values > 0.0", __FILE__, __LINE__, false );
  }
}
