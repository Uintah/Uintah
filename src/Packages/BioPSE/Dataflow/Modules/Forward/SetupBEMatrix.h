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
 *  SetupBEMatrix.h:  class to build Boundary Elements matrix
 *
 *  Written by:
 *   Saeed Babaei Zadeh
 *   Norteastern University
 *   July 2003
 *   Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurfField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <math.h>

#include <algorithm>
#include <map>
#include <iostream>
#include <string>
#include <fstream>

#define epsilon 1e-12
#define PI M_PI

 namespace BioPSE {

using namespace SCIRun;


class SetupBEMatrix : public Module
{
  vector<Vector>           avInn_;
  MatrixHandle       hZoi_;
  typedef LockingHandle<DenseMatrix>     DenseMatrixHandle;
  
private:

  bool ray_triangle_intersect(double &t,
			      const Point &p,
			      const Vector &v,
			      const Point &p0,
			      const Point &p1,
			      const Point &p2) const;
  void compute_intersections(vector<pair<double, int> > &results,
			     const TriSurfMeshHandle &mesh,
			     const Point &p, const Vector &v,
			     int marker) const;


  int compute_parent(const vector<TriSurfMeshHandle> &meshes, int index);

  bool compute_nesting(vector<int> &nesting,
		       const vector<TriSurfMeshHandle> &meshes);

  void calc_tri_area(TriSurfMeshHandle, vector<Vector>&);

  void build_Zoi( const vector<TriSurfMeshHandle> &,
                                vector<int> &,
                                vector<double>&,
                                int hs,
                                int ms,
                                MatrixHandle &);

inline void  SetupBEMatrix::get_g_coef(
					const Vector&,
					const Vector&,
					const Vector&,
					const Vector&,
					double,
					double,
					const Vector&,
					DenseMatrix&);


inline void  SetupBEMatrix::get_cruse_weights(
					const Vector&,
					const Vector&,
					const Vector&,
					double,
					double,
					double,
					DenseMatrix&);


inline void  SetupBEMatrix::getOmega(
				     const Vector&,
				     const Vector&,
				     const Vector&,
				     DenseMatrix&);


inline double  SetupBEMatrix::do_radon_g(
					const Vector&,
					const Vector&,
					const Vector&,
                                        const Vector&,
                                        double,
                                        double,
                                        DenseMatrix&);

inline void  SetupBEMatrix::get_auto_g(
					const Vector&,
					const Vector&,
					const Vector&,
                                        unsigned int,
					DenseMatrix&,
                                        double,
                                        double,
                                        DenseMatrix&);

inline double  SetupBEMatrix::get_new_auto_g(
					const Vector&,
					const Vector&,
					const Vector&);


inline void SetupBEMatrix::make_auto_P(TriSurfMeshHandle,
                                        DenseMatrixHandle&,
                                        double,
                                        double,
                                        double);

inline void SetupBEMatrix::make_cross_P(TriSurfMeshHandle,
                                         TriSurfMeshHandle,
                                         DenseMatrixHandle&,
                                         double,
                                         double,
                                         double);

inline void SetupBEMatrix::make_cross_G(TriSurfMeshHandle,
                                         TriSurfMeshHandle,
                                         DenseMatrixHandle&,
                                         double,
                                         double,
                                         double);

inline void SetupBEMatrix::make_auto_G(TriSurfMeshHandle,
                                        DenseMatrixHandle&,
                                        double,
                                        double,
                                        double);

void SetupBEMatrix::concat_rows(DenseMatrixHandle,
                                 DenseMatrixHandle,
                                 DenseMatrix*);

void SetupBEMatrix::concat_cols(DenseMatrixHandle,
                                 DenseMatrixHandle,
                                 DenseMatrix*);

public:

  //! Constructor
  SetupBEMatrix(GuiContext *context);

  //! Destructor
  virtual ~SetupBEMatrix();

  virtual void execute();
};


DECLARE_MAKER(SetupBEMatrix)


SetupBEMatrix::SetupBEMatrix(GuiContext *context):
  Module("SetupBEMatrix", context, Source, "Forward", "BioPSE")
{
}


SetupBEMatrix::~SetupBEMatrix()
{
}


// C++ized MollerTrumbore97 Ray Triangle intersection test.
#define EPSILON 1.0e-6
bool
SetupBEMatrix::ray_triangle_intersect(double &t,
				       const Point &point,
				       const Vector &dir,
				       const Point &p0,
				       const Point &p1,
				       const Point &p2) const
{
  // Find vectors for two edges sharing p0.
  const Vector edge1 = p1 - p0;
  const Vector edge2 = p2 - p0;

  // begin calculating determinant - also used to calculate U parameter.
  const Vector pvec = Cross(dir, edge2);

  // if determinant is near zero, ray lies in plane of triangle.
  const double det = Dot(edge1, pvec);
  if (det > -EPSILON && det < EPSILON)
  {
    return false;
  }
  const double inv_det = 1.0 / det;

  // Calculate distance from vert0 to ray origin.
  const Vector tvec = point - p0;

  // Calculate U parameter and test bounds.
  const double u = Dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0)
  {
    return false;
  }

  // Prepare to test V parameter.
  const Vector qvec = Cross(tvec, edge1);

  // Calculate V parameter and test bounds.
  const double v = Dot(dir, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0)
  {
    return false;
  }

  // Calculate t, ray intersects triangle.
  t = Dot(edge2, qvec) * inv_det;

  return true;
}


void
SetupBEMatrix::compute_intersections(vector<pair<double, int> >
&results,
				      const TriSurfMeshHandle &mesh,
				      const Point &p, const Vector &v,
				      int marker) const
{
  TriSurfMesh::Face::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  double t;
  while (itr != eitr)
  {
    TriSurfMesh::Node::array_type nodes;
    mesh->get_nodes(nodes, *itr);
    Point p0, p1, p2;
    mesh->get_center(p0, nodes[0]);
    mesh->get_center(p1, nodes[1]);
    mesh->get_center(p2, nodes[2]);
    if (ray_triangle_intersect(t, p, v, p0, p1, p2))
    {
      results.push_back(pair<double, int>(t, marker));
    }
    ++itr;
  }
}

static bool
pair_less(const pair<double, int> &a,
	  const pair<double, int> &b)
{
  return a.first < b.first;
}


int
SetupBEMatrix::compute_parent(const vector<TriSurfMeshHandle> &meshes,
			       int index)
{
  Point point;
  meshes[index]->get_center(point, TriSurfMesh::Node::index_type(0));
  Vector dir(1.0, 1.0, 1.0);
  vector<pair<double, int> > intersections;

  unsigned int i;
  for (i = 0; i < meshes.size(); i++)
  {
    compute_intersections(intersections, meshes[i], point, dir, i);
  }

  std::sort(intersections.begin(), intersections.end(), pair_less);

  vector<int> counts(meshes.size(), 0);
  for (i = 0; i < intersections.size(); i++)
  {
    if (intersections[i].second == index)
    {
      // First odd count is parent.
      for (int j = i-1; j >= 0; j--)
      {
	if (counts[intersections[j].second] & 1)
	{
	  return intersections[j].second;
	}
      }
      // No odd parent, is outside.
      return meshes.size();
    }
    counts[intersections[i].second]++;
  }

  // Indeterminant, we should intersect with ourselves.
  return meshes.size();
}



bool
SetupBEMatrix::compute_nesting(vector<int> &nesting,
				const vector<TriSurfMeshHandle> &meshes)
{
  nesting.resize(meshes.size());

  unsigned int i;
  for (i = 0; i < meshes.size(); i++)
  {
    nesting[i] = compute_parent(meshes, i);
  }

  return true;
}




inline void  SetupBEMatrix::getOmega(
				     const Vector& y1,
				     const Vector& y2,
				     const Vector& y3,
				     DenseMatrix& coef)
{
/*
 This function deals with the analytical solutions of the various integrals in the stiffness matrix
 The function is concerned with the integrals of the linear interpolations functions over the triangles
 and takes care of the solid spherical angle. As in most cases not all values are needed the computation
 is split up in integrals from one surface to another one

 The computational scheme follows the analytical formulas derived by the
    de Munck 1992 (IEEE Trans Biomed Engng, 39-9, pp 986-90)
*/

  Vector y21 = y2 - y1;
  Vector y32 = y3 - y2;
  Vector y13 = y1 - y3;

  Vector Ny( y1.length() , y2.length() , y3.length() );

  Vector Nyij( y21.length() , y32.length() , y13.length() );


  Vector gamma( 0 , 0 , 0 );
  double NomGamma , DenomGamma;

  NomGamma = Ny[0]*Nyij[0] + Dot(y1,y21);
  DenomGamma = Ny[1]*Nyij[0] + Dot(y2,y21);
  if (fabs(DenomGamma-NomGamma) > epsilon && (DenomGamma != 0) && NomGamma != 0 )
  	 gamma[0] = -1/Nyij[0] * log(NomGamma/DenomGamma);

  NomGamma = Ny[1]*Nyij[1] + Dot(y2,y32);
  DenomGamma = Ny[2]*Nyij[1] + Dot(y3,y32);
  if (fabs(DenomGamma-NomGamma) > epsilon && (DenomGamma != 0) && NomGamma != 0 )
  	gamma[1] = -1/Nyij[1] * log(NomGamma/DenomGamma);

  NomGamma = Ny[2]*Nyij[2] + Dot(y3,y13);
  DenomGamma = Ny[0]*Nyij[2] + Dot(y1,y13);
  if (fabs(DenomGamma-NomGamma) > epsilon && (DenomGamma != 0) && NomGamma != 0 )
  	gamma[2] = -1/Nyij[2] * log(NomGamma/DenomGamma);

  double d = Dot( y1, Cross(y2, y3) );

  Vector OmegaVec = (gamma[2]-gamma[0])*y1 + (gamma[0]-gamma[1])*y2 + (gamma[1]-gamma[2])*y3;

  /*
      In order to avoid problems with the arctan used in de Muncks paper
      the result is tested. A problem is that his formula under certain
      circumstances leads to unexpected changes of signs. Hence to avoid
      this, the denominator is checked and 2*pi is added if necessary.
      The problem without the two pi results in the following situation in
      which division of the triangle into three pieces results into
      an opposite sign compared to the sperical angle of the total
      triangle. These cases are rare but existing.
  */

  double Nn=0 , Omega=0 ;
  Nn = Ny[0]*Ny[1]*Ny[2] + Ny[0]*Dot(y2,y3) + Ny[2]*Dot(y1,y2) + Ny[1]*Dot(y3,y1);
  if (Nn > 0)  Omega = 2 * atan( d / Nn );
  if (Nn < 0)  Omega = 2 * atan( d / Nn ) + 2*PI ;
  if (Nn == 0)
        if ( d > 0 ) Omega = PI;
    	        else  Omega = -PI;

  Vector N = Cross(y21, -y13);
  double Zn1 = Dot(Cross(y2, y3) , N);
  double Zn2 = Dot(Cross(y3, y1) , N);
  double Zn3 = Dot(Cross(y1, y2) , N);

  double A2 = -N.length2();
  coef[0][0] = (1/A2) * ( Zn1*Omega + d * Dot(y32, OmegaVec) );
  coef[0][1] = (1/A2) * ( Zn2*Omega + d * Dot(y13, OmegaVec) );
  coef[0][2] = (1/A2) * ( Zn3*Omega + d * Dot(y21, OmegaVec) );

  return;

}


inline void  SetupBEMatrix::get_cruse_weights(
					const Vector& p1,
					const Vector& p2,
					const Vector& p3,
					double s,
					double r,
					double area,
					DenseMatrix& cruse_weights)
{
 /*
 Inputs: p1,p2,p3= cartesian coordiantes of the triangle vertices ;
         area = triangle area
 Output: cruse_weights = The weighting factors for the 7 Radon points of the triangle
 Format of the cruse_weights matix
              Radon point 1       Radon Point 2    ...     Radon Point 7
 Vertex 1 ->
 Vertex 2 ->
 Vertex 3 ->

 Set up the local coordinate system around the triangle. This is a 2-D system and
 for ease of use, the x-axis is chosen as the line between the first and second vertex
 of the triangle. From that the vertex of the third point is found in local coordinates.
 */

// The angle between the F2 and F3, at vertex 1 is (pi - 'alpha').
  Vector fg2 = p1 - p3;
  double fg2_length = fg2.length();

  Vector fg3 = p2 - p1;
  double fg3_length = fg3.length();

  double cos_alpha = - Dot(fg3,fg2) / (fg2_length * fg3_length);
  double sin_alpha = sqrt(1 - cos_alpha * cos_alpha);

// Now the vertices in local coordinates
  Vector locp1(0 , 0 , 0);
  Vector locp2(fg3_length , 0 , 0);
  Vector locp3(fg2_length * cos_alpha , fg2_length * sin_alpha , 0);

  DenseMatrix Fx(3, 1);
  Fx[0][0] = locp3[0] - locp2[0];
  Fx[1][0] = locp1[0] - locp3[0];
  Fx[2][0] = locp2[0] - locp1[0];

  DenseMatrix Fy(3, 1);
  Fy[0][0] = locp3[1] - locp2[1];
  Fy[1][0] = locp1[1] - locp3[1];
  Fy[2][0] = locp2[1] - locp1[1];

  Vector centroid = (locp1 + locp2 + locp3) / 3;
  DenseMatrix loc_radpt_x(1, 7);
  DenseMatrix loc_radpt_y(1, 7);
  Vector temp;
  loc_radpt_x[0][0] = centroid[0];
  loc_radpt_y[0][0] = centroid[1];
  temp = (1-s) * centroid;
  loc_radpt_x[0][1] = temp[0] + locp1[0]*s;
  loc_radpt_y[0][1] = temp[1] + locp1[1]*s;
  loc_radpt_x[0][2] = temp[0] + locp2[0]*s;
  loc_radpt_y[0][2] = temp[1] + locp2[1]*s;
  loc_radpt_x[0][3] = temp[0] + locp3[0]*s;
  loc_radpt_y[0][3] = temp[1] + locp3[1]*s;
  temp = (1-r) * centroid;
  loc_radpt_x[0][4] = temp[0] + locp1[0]*r;
  loc_radpt_y[0][4] = temp[1] + locp1[1]*r;
  loc_radpt_x[0][5] = temp[0] + locp2[0]*r;
  loc_radpt_y[0][5] = temp[1] + locp2[1]*r;
  loc_radpt_x[0][6] = temp[0] + locp3[0]*r;
  loc_radpt_y[0][6] = temp[1] + locp3[1]*r;

  DenseMatrix temp1(3, 7);
  DenseMatrix temp2(3, 7);
  DenseMatrix A(3, 7);
  Mult(temp1, Fy, loc_radpt_x);
  Mult(temp2, Fx, loc_radpt_y);
  Add(A, 1, temp1, -1, temp2);
  A.mult(0.5/area);

  DenseMatrix ones(1, 7);
  for (int i=0; i < 7 ; i++) ones[0][i] = 1;
  DenseMatrix E(3, 1);
/*
 E is a 1X3 matrix: [1st vertex  ;  2nd vertex  ;  3rd vertex]
   E = [1/3 ; 1/3 ; 1/3] + (0.5/area)*(Fy*xmid - Fx*ymid);
 but there is no need to compute the E because by our chioce of the
 local coordinates, it is easy to show that the E is always [1 ; 0 ; 0]!
*/
  E[0][0] = 1;
  E[1][0] = 0;
  E[2][0] = 0;
  Mult(temp1, E, ones);
  Add(cruse_weights, 1, temp1, -1, A);

  return;
}


inline void  SetupBEMatrix::get_g_coef(
					const Vector& p1,
					const Vector& p2,
					const Vector& p3,
					const Vector& op,
					double s,
					double r,
					const Vector& centroid,
					DenseMatrix& g_coef)
{
// Inputs: p1,p2,p3= cartesian coordiantes of the triangle vertices ; op= Observation Point
// Output: g_coef = G Values (Coefficients) at 7 Radon's points = 1/r
  Vector radpt;
  Vector temp;

  radpt = centroid - op;
  g_coef[0][0] = 1 / radpt.length();

  temp = centroid * (1-s) - op;
  radpt = temp + p1 * s;
  g_coef[0][1] = 1 / radpt.length();
  radpt = temp + p2 * s;
  g_coef[0][2] = 1 / radpt.length();
  radpt = temp + p3 * s;
  g_coef[0][3] = 1 / radpt.length();

  temp = centroid * (1-r) - op;
  radpt = temp + p1 * r;
  g_coef[0][4] = 1 / radpt.length();
  radpt = temp + p2 * r;
  g_coef[0][5] = 1 / radpt.length();
  radpt = temp + p3 * r;
  g_coef[0][6] = 1 / radpt.length();

  return;
}




inline void  SetupBEMatrix::get_auto_g(
					const Vector& p1,
					const Vector& p2,
					const Vector& p3,
                                        unsigned int op_n,
					DenseMatrix& g_values,
                                        double s,
                                        double r,
                                        DenseMatrix& R_W)
{
     /*
        A routine to solve the Auto G-parameter integral for a triangle from
        a "closed" observation point.
        The scheme is the standard one for all the BEM routines.
        Input are the observation point and triangle co-ordinates
        Input:
        op_n = observation point number (1, 2 or 3)
        p1,p2,p3 = triangle co-ordinates (REAL)

         Output:
         g_values
                = the total values for the 1/r integral for
                each of the subtriangles associated with each
                integration triangle vertex about the observation
                point defined by the calling
                program.
      */

   Vector p5 = (p1 + p2) / 2;
   Vector p6 = (p2 + p3) / 2;
   Vector p4 = (p1 + p3) / 2;
   Vector ctroid = (p1 + p2 + p3) / 3;
   Vector op;

   switch(op_n)
       {
        case 0:
                op = p1;
                g_values[0][0] = get_new_auto_g(op, p5, p4) + do_radon_g(p5, ctroid, p4, op, s, r, R_W);
                g_values[1][0] = do_radon_g(p2, p6, p5, op, s, r, R_W) + do_radon_g(p5, p6, ctroid, op, s, r, R_W);
                g_values[2][0] = do_radon_g(p3, p4, p6, op, s, r, R_W) + do_radon_g(p4, ctroid, p6, op, s, r, R_W);
                break;
        case 1:
                op = p2;
                g_values[0][0] = do_radon_g(p1, p5, p4, op, s, r, R_W) + do_radon_g(p5, ctroid, p4, op, s, r, R_W);
                g_values[1][0] = get_new_auto_g(op, p6, p5) + do_radon_g(p5, p6, ctroid, op, s, r, R_W);
                g_values[2][0] = do_radon_g(p3, p4, p6, op, s, r, R_W) + do_radon_g(p4, ctroid, p6, op, s, r, R_W);
                break;
        case 2:
                op = p3;
                g_values[0][0] = do_radon_g(p1, p5, p4, op, s, r, R_W) + do_radon_g(p5, ctroid, p4, op, s, r, R_W);
                g_values[1][0] = do_radon_g(p2, p6, p5, op, s, r, R_W) + do_radon_g(p5, p6, ctroid, op, s, r, R_W);
                g_values[2][0] = get_new_auto_g(op, p4, p6) + do_radon_g(p4, ctroid, p6, op, s, r, R_W);
                break;
        default:;
       }
   return;
   }


inline double  SetupBEMatrix::get_new_auto_g(
					const Vector& op,
					const Vector& p2,
					const Vector& p3)
{
  //  Inputs: op,p2,p3= cartesian coordiantes of the triangle vertices ; op= Observation Point
  //  Output: g1 = G value for the triangle for "auto_g"
  //  This function is called from get_auto_g.m

  double delta_min = 0.00001;
  unsigned int max_number_of_divisions = 256;

  Vector a = p2 - op; double a_mag = a.length();
  Vector b = p3 - p2; double b_mag = b.length();
  Vector c = op - p3; double c_mag = c.length();

  Vector aV = Cross(p2 - op, p3 - p2)*0.5;
  double area = aV.length();
  double area2 = 2*area;
  double h = (area2) / b_mag;
  double alfa=0;  if (h<a_mag) alfa = acos(h/a_mag);
  double AC = a_mag*c_mag;
  double teta = 0; if (area2<AC) teta = asin( area2 / AC );

  unsigned int nod = 1;
  double sai_old = sqrt(area2 * teta);
  double delta = 1;

  double gama, gama_j, rhoj_1, rhoj, sum, sai_new=0;

  while( (delta >= delta_min) && (nod <= max_number_of_divisions) )
  {
    nod = 2*nod;
    gama = teta / nod;
    sum = 0;
    gama_j = 0;
    rhoj_1 = a_mag;
    for ( unsigned int j = 1; j <= nod; j++)
    {
      gama_j = gama_j + gama;
      rhoj = h / cos(alfa - gama_j);
      sum = sum + sqrt( fabs(rhoj * rhoj_1) );
      rhoj_1 = rhoj;
    }
    sai_new = sum * sqrt( fabs(gama * sin(gama)) );
    delta = fabs( (sai_new - sai_old) / (sai_new + sai_old) );
    sai_old = sai_new;
  }
   return sai_new;
}


inline double  SetupBEMatrix::do_radon_g(
					const Vector& p1,
					const Vector& p2,
					const Vector& p3,
                                        const Vector& op,
                                        double s,
                                        double r,
                                        DenseMatrix& R_W)
{
  //  Inputs: p1,p2,p3= cartesian coordiantes of the triangle vertices ; op= Observation Point
  //  Output: g2 = G value for the triangle for "auto_g"
  //  This function is called from get_auto_g.m

  Vector centroid = (p1 + p2 + p3) / 3;

  DenseMatrix g_coef(1, 7);
  get_g_coef(p1, p2, p3, op, s, r, centroid, g_coef);

  double g2 = 0;
  for (int i=0; i<7; i++)   g2 = g2 + g_coef[0][i]*R_W[0][i];

  Vector aV = Cross(p2 - p1, p3 - p2)*0.5;

  return g2 * aV.length();
}

void SetupBEMatrix::concat_rows(DenseMatrixHandle m1H, DenseMatrixHandle m2H, DenseMatrix *out) {

    int r, c;
    if (m1H->ncols() != m2H->ncols()) {
	  warning("Two matrices must have same number of columns");
	  exit(0);  }
    for (r = 0; r <= m1H->nrows()-1; r++)
    {
      for (c = 0; c <= m1H->ncols()-1; c++)
      {
        out->put(r, c, m1H->get(r,c));
      }
    }

    for (r = m1H->nrows(); r <= m1H->nrows()+m2H->nrows()-1; r++)
    {
      for (c = 0; c <= m2H->ncols()-1; c++)
      {
        out->put(r, c, m2H->get(r - m1H->nrows(), c));
      }
    }
    return;
}

void SetupBEMatrix::concat_cols(DenseMatrixHandle m1H, DenseMatrixHandle m2H, DenseMatrix *out) {

    int r, c;
    if (m1H->nrows() != m2H->nrows()) {
	  warning("Two matrices must have same number of rows");
	  exit(0);  }
    for (r = 0; r <= m1H->nrows()-1; r++)
    {
      for (c = 0; c <= m1H->ncols()-1; c++)
      {
        out->put(r, c, m1H->get(r,c));
      }
    }

    for (r = 0; r <= m2H->nrows()-1; r++)
    {
      for (c = m1H->ncols(); c <= m1H->ncols()+m2H->ncols()-1; c++)
      {
        out->put(r, c, m2H->get(r,c - m1H->ncols()));
      }
    }
    return;
}

 void SetupBEMatrix::build_Zoi(const vector<TriSurfMeshHandle> &meshes,
                        vector<int> &nesting,
                        vector<double> &conductivities,
                        int hs,
                        int ms,
                        MatrixHandle &hZoi_){

  // hs is the surface whose potentials are given and ms is the surface whose potentials are desired.

  vector<DenseMatrixHandle> PP;
  DenseMatrixHandle h_PP_;
  int i ,j;
  double in_cond, out_cond, op_cond;
  int no_of_fields = nesting.size();

  // PP Begin
  for (i=0; i<no_of_fields; i++)
     for (j=0; j<no_of_fields; j++)
      {
        if (i==j)
          {
            in_cond = conductivities[i];
            if (nesting[i] == no_of_fields)  // The outermost surface is different from the others.
               {out_cond = 0;
                op_cond = in_cond;}
            else
               {out_cond = conductivities[nesting[i]];
                op_cond = out_cond;}
            make_auto_P(meshes[i], h_PP_, in_cond, out_cond, op_cond);
          }
        else
          {
            in_cond = conductivities[j];
            if (nesting[j] == no_of_fields) out_cond = 0;
            else out_cond = conductivities[nesting[j]];

            if (nesting[i] == no_of_fields) op_cond = conductivities[i];
            else op_cond = conductivities[nesting[i]];

            make_cross_P(meshes[i], meshes[j], h_PP_, in_cond, out_cond, op_cond);
          }
        PP.push_back(h_PP_);
      }

  // PP End

  vector<DenseMatrixHandle> GG;
  DenseMatrixHandle h_GG_;

  // GG Begin
  int given_potentials_surface_index = hs; // This is the surface whose potentials are given i.e. heart surface
  calc_tri_area(meshes[given_potentials_surface_index], avInn_);

  for (i=0; i<no_of_fields; i++)
      {
        if (i==given_potentials_surface_index)
          {
            in_cond = conductivities[i];
            if (nesting[i] == no_of_fields)  // The Outermost surface is different from the others.
               {out_cond = 0;
                op_cond = in_cond;}
            else
               {out_cond = conductivities[nesting[i]];
                op_cond = out_cond;}

            make_auto_G(meshes[i], h_GG_, in_cond, out_cond, op_cond);
          }
        else
          {
            in_cond = conductivities[given_potentials_surface_index];
            if (nesting[given_potentials_surface_index] == no_of_fields) out_cond = 0;
            else out_cond = conductivities[nesting[given_potentials_surface_index]];

            if (nesting[i] == no_of_fields) op_cond = conductivities[i];
            else op_cond = conductivities[nesting[i]];

            make_cross_G(meshes[i], meshes[given_potentials_surface_index], h_GG_, in_cond, out_cond, op_cond);
          }
            GG.push_back(h_GG_);
      }



  if (no_of_fields>2)
  {

  // Page 288, Pilkington Paper, IEEE Trans. on Biomedical Engineering, Vol. BME-33. No.3, March 1986
  DenseMatrixHandle hPhh_ = PP[hs*(no_of_fields+1)];
  DenseMatrixHandle hPmm_ = PP[ms*(no_of_fields+1)];
  DenseMatrixHandle hGmh_ = GG[ms];
  DenseMatrixHandle hGhh_ = GG[hs];
  DenseMatrixHandle hPmh_ = PP[ms*no_of_fields+hs];
  DenseMatrixHandle hPhm_ = PP[hs*no_of_fields+ms];

  DenseMatrix *omatrix = 0;
  TriSurfMesh::Node::size_type nsize_hs, nsize_ms, nsize;
  meshes[given_potentials_surface_index]->size(nsize_hs);
  meshes[ms]->size(nsize_ms);

  DenseMatrixHandle hGth_;
  int first = 1;
  for (i=0; i<no_of_fields; i++)
    if (i!=hs && i!=ms)
    {
      if(first)
        {
          first = 0;
          hGth_ = GG[i];
        }
      else
        {
          omatrix=scinew DenseMatrix(hGth_->nrows()+GG[i]->nrows(),nsize_hs);
          concat_rows(hGth_, GG[i], omatrix);
          hGth_ = omatrix;
        }
    }
 // Gth Done!
//  cout<<"\nGth Done!";

  DenseMatrixHandle hPht_;
  first = 1;
  for (i=hs*no_of_fields; i<no_of_fields*(hs+1); i++)
    if (i!=hs*(no_of_fields+1) && i!=hs*no_of_fields+ms)
    {
      if(first)
        {
          first = 0;
          hPht_ = PP[i];
        }
      else
        {
          omatrix=scinew DenseMatrix(nsize_hs,hPht_->ncols()+PP[i]->ncols());
          concat_cols(hPht_, PP[i], omatrix);
          hPht_ = omatrix;
        }
    }
 // Pht Done!
//  cout<<"\nPht Done!";

  DenseMatrixHandle hPmt_;
  first = 1;
  for (i=ms*no_of_fields; i<no_of_fields*(ms+1); i++)
    if (i!=ms*no_of_fields+hs && i!=ms*(no_of_fields+1))
    {
      if(first)
        {
          first = 0;
          hPmt_ = PP[i];
        }
      else
        {
          omatrix=scinew DenseMatrix(nsize_ms,hPmt_->ncols()+PP[i]->ncols());
          concat_cols(hPmt_, PP[i], omatrix);
          hPmt_ = omatrix;
        }
    }
 // Pmt Done!
//  cout<<"\nPmt Done!";

  DenseMatrixHandle hPth_;
  first = 1;
  for (i=0; i<no_of_fields; i++)
    if (i!=ms && i!=hs)
    {
     if(first)
      {
        first = 0;
        hPth_ = PP[i*no_of_fields+hs];
      }
     else
      {
        omatrix=scinew DenseMatrix(hPth_->nrows()+PP[i*no_of_fields+hs]->nrows(),nsize_hs);
        concat_rows(hPth_, PP[i*no_of_fields+hs], omatrix);
        hPth_ = omatrix;
      }
     }
 // Pth Done!
//  cout<<"\nPth Done!";


  DenseMatrixHandle hPtm_;
  first = 1;
  for (i=0; i<no_of_fields; i++)
    if (i!=ms && i!=hs)
    {
     if(first)
      {
        first = 0;
        hPtm_ = PP[i*no_of_fields+ms];
      }
     else
      {
        omatrix=scinew DenseMatrix(hPtm_->nrows()+PP[i*no_of_fields+ms]->nrows(),nsize_ms);
        concat_rows(hPtm_, PP[i*no_of_fields+ms], omatrix);
        hPtm_ = omatrix;
      }
     }
 // Ptm Done!
//  cout<<"\nPtm Done!";

  DenseMatrixHandle hPtt_, hPtt_row_;
  int firstrow = 1;

  for (i=0; i<no_of_fields; i++)
    if (i!=ms && i!=hs)
    {
     meshes[i]->size(nsize);
     first = 1;
     for (j=i*no_of_fields; j<no_of_fields*(i+1); j++)
         if (j!=i*no_of_fields+hs && j!=i*no_of_fields+ms)
         {
           if(first)
             {
               first = 0;
               hPtt_row_ = PP[j];
             }
           else
             {
               omatrix=scinew DenseMatrix(nsize,hPtt_row_->ncols()+PP[j]->ncols());
               concat_cols(hPtt_row_, PP[j], omatrix);
               hPtt_row_ = omatrix;
             }
         }
     // One rwo of Ptt is ready
     if(firstrow)
        {
         firstrow = 0;
         hPtt_ = hPtt_row_;
        }
     else
        {
         omatrix=scinew DenseMatrix(hPtt_row_->nrows()+hPtt_->nrows(), hPtt_row_->ncols());
         concat_rows(hPtt_, hPtt_row_, omatrix);
         hPtt_ = omatrix;
        }
    }
 // Ptt Done!
//  cout<<"\nPtt Done!";

  hGhh_->invert();

  DenseMatrixHandle hG_m_h_ = new DenseMatrix(hGmh_->nrows(), hGmh_->ncols());
  Mult(*(hG_m_h_.get_rep()), *(hGmh_.get_rep()), *(hGhh_.get_rep())); // G_m_h <- Gmh*Ghh^-1

  DenseMatrixHandle hG_t_h_ = new DenseMatrix(hGth_->nrows(), hGth_->ncols());
  Mult(*(hG_t_h_.get_rep()), *(hGth_.get_rep()), *(hGhh_.get_rep())); // G_t_h <- Gth*Ghh^-1

  DenseMatrixHandle htmpMM_ = new DenseMatrix(hPmm_->nrows(), hPmm_->ncols());
  Mult(*htmpMM_.get_rep(), *hG_m_h_.get_rep(), *hPhm_.get_rep());   // tmpMM <- Gmh*Ghh^-1*Phm
  Add(1, *(hPmm_.get_rep()), -1, *(htmpMM_.get_rep()));    // Pmm <- Pmm-Gmh*Ghh^-1*Phm

  DenseMatrixHandle htmpMT_ = new DenseMatrix(hPmt_->nrows(), hPmt_->ncols());
  Mult(*htmpMT_.get_rep(), *hG_m_h_.get_rep(), *hPht_.get_rep());   // tmpMT <- Gmh*Ghh^-1*Pht
  Add(1, *(hPmt_.get_rep()), -1, *(htmpMT_.get_rep()));    // Pmt <- Pmt-Gmh*Ghh^-1*Pht

  DenseMatrixHandle htmpTT_ = new DenseMatrix(hPtt_->nrows(), hPtt_->ncols());
  Mult(*htmpTT_.get_rep(), *hG_t_h_.get_rep(), *hPht_.get_rep());   // tmpTT <- Gth*Ghh^-1*Pht
  Add(1, *(hPtt_.get_rep()), -1, *(htmpTT_.get_rep()));    // Ptt <- Ptt-Gth*Ghh^-1*Pht
  hPtt_->invert();
  htmpTT_ = 0; // Freeing the memory

  DenseMatrixHandle htmpTM_ = new DenseMatrix(hPtm_->nrows(), hPtm_->ncols());
  Mult(*htmpTM_.get_rep(), *hG_t_h_.get_rep(), *hPhm_.get_rep());   // tmpTM <- Gth*Ghh^-1*Phm
  Add(-1, *(hPtm_.get_rep()), 1, *(htmpTM_.get_rep()));    // Ptm <- -Ptm+Gth*Ghh^-1*Phm
  htmpTM_ = 0; // Freeing the memory

  Mult(*htmpMT_.get_rep(), *hPmt_.get_rep(), *hPtt_.get_rep());   // tmpMT <- (Pmt-Gmh*Ghh^-1*Pht)*(Ptt-Gth*Ghh^-1*Pht)^-1
  Mult(*htmpMM_.get_rep(), *htmpMT_.get_rep(), *hPtm_.get_rep());   // tmpMM <- (Pmt-Gmh*Ghh^-1*Pht)*(Ptt-Gth*Ghh^-1*Pht)^-1*(-Ptm+Gth*Ghh^-1*Phm)
  Add(1, *(hPmm_.get_rep()), 1, *(htmpMM_.get_rep()));
  hPmm_->invert();    // Pmm <- Part A

  DenseMatrixHandle htmpMH_ = new DenseMatrix(hPmh_->nrows(), hPmh_->ncols());
  Mult(*htmpMH_.get_rep(), *hG_m_h_.get_rep(), *hPhh_.get_rep());   // tmpMH <- Gmh*Ghh^-1*Phh
  Add(-1, *(hPmh_.get_rep()), 1, *(htmpMH_.get_rep()));    // Pmh <- -Pmh+Gmh*Ghh^-1*Phh

  DenseMatrixHandle htmpTH_ = new DenseMatrix(hPth_->nrows(), hPth_->ncols());
  Mult(*htmpTH_.get_rep(), *hG_t_h_.get_rep(), *hPhh_.get_rep());   // tmpTH <- Gth*Ghh^-1*Phh
  Add(1, *(hPth_.get_rep()), -1, *(htmpTH_.get_rep()));    // Pth <- Pth-Gth*Ghh^-1*Phh

  Mult(*htmpMH_.get_rep(), *htmpMT_.get_rep(), *hPth_.get_rep());   // tmpMH <- (Pmt-Gmh*Ghh^-1*Pht)*(Ptt-Gth*Ghh^-1*Pht)^-1*(Pth-Gth*Ghh^-1*Phh)
  htmpMT_ = 0;  // Freeing the memory

  Add(1, *(hPmh_.get_rep()), 1, *(htmpMH_.get_rep()));    // Pmh <- Part B

  DenseMatrixHandle hZmh_ = new DenseMatrix(hPmh_->nrows(), hPmh_->ncols());
  Mult(*hZmh_.get_rep(), *hPmm_.get_rep(), *hPmh_.get_rep());

  hZoi_ = hZmh_.get_rep();

  } // end   if (no_of_fields>1)

  else   //  for just two surfaces: heart and body
  {
  GG[hs]->invert(); // GG[hs] = hGhh_

  DenseMatrixHandle hZbh_ = new DenseMatrix(GG[ms]->nrows(), GG[ms]->ncols()); // GG[ms] = hGbh_
  Mult(*(hZbh_.get_rep()), *(GG[ms].get_rep()), *(GG[hs].get_rep())); // Zbh <- Gbh*Ghh^-1
  DenseMatrix tmpBB(PP[ms*(no_of_fields+1)]->nrows(), PP[ms*(no_of_fields+1)]->ncols()); // PP[ms*(no_of_fields+1)] = hPbb_
  Mult(tmpBB, *hZbh_.get_rep(), *PP[hs*(no_of_fields-1)+1].get_rep());   // tmpBB <- Gbh*Ghh^-1*Phb   PP[hs*(no_of_fields-1)+1]=hPhb_
  Add(1, *(PP[ms*(no_of_fields+1)].get_rep()), -1, tmpBB);    // Pbb <- part A (Pbb-Gbh*Ghh^-1*Phb)
  PP[ms*(no_of_fields+1)]->invert();
  DenseMatrix tmpBH(PP[ms*(no_of_fields-1)+1]->nrows(), PP[ms*(no_of_fields-1)+1]->ncols()); // PP[ms*(no_of_fields-1)+1] = hPbh_
  Mult(tmpBH, *hZbh_.get_rep(), *PP[hs*(no_of_fields+1)].get_rep());  // tmpBH <- Gbh*Ghh^-1*Phh   PP[hs*(no_of_fields+1)] = hPhh_
  Add(-1, *(PP[ms*(no_of_fields-1)+1].get_rep()), 1, tmpBH);   // Pbh <- part B (Gbh*Ghh^-1*Phh-Pbh)
  Mult(*(hZbh_.get_rep()), *(PP[ms*(no_of_fields+1)].get_rep()), *(PP[ms*(no_of_fields-1)+1].get_rep()));
  hZbh_->mult(-1);
  hZoi_ = hZbh_.get_rep();

  } // end else   if (no_of_fields>1)

//  puts("\nDONE!");

  return;
}


void SetupBEMatrix::make_auto_G(TriSurfMeshHandle hsurf, DenseMatrixHandle &h_GG_,
                                 double in_cond, double out_cond, double op_cond)
{
  TriSurfMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  h_GG_ = tmp;
  DenseMatrix& auto_G = *tmp;
  auto_G.zero();

  const double mult = 1/(2*PI)*((out_cond - in_cond)/op_cond);  // op_cond=out_cond for all the surfaces but the outermost surface which in op_cond=in_cond

  TriSurfMesh::Node::array_type nodes;

  TriSurfMesh::Node::iterator ni, nie;
  TriSurfMesh::Face::iterator fi, fie;
  DenseMatrix cruse_weights(3, 7);
  DenseMatrix g_coef(1, 7);
  DenseMatrix R_W(1,7); // Radon Points Weights
  DenseMatrix temp(1,7);
  DenseMatrix g_values(3, 1);

  double area;

  double sqrt15 = sqrt(15.0);
  R_W[0][0] = 9/40;
  R_W[0][1] = (155 + sqrt15) / 1200;
  R_W[0][2] = R_W[0][1];
  R_W[0][3] = R_W[0][1];
  R_W[0][4] = (155 - sqrt15) / 1200;
  R_W[0][5] = R_W[0][4];
  R_W[0][6] = R_W[0][4];

  double s = (1 - sqrt15) / 7;
  double r = (1 + sqrt15) / 7;

  hsurf->begin(fi); hsurf->end(fie);
  for (; fi != fie; ++fi)
  { //! find contributions from every triangle
    hsurf->get_nodes(nodes, *fi);
    Vector p1(hsurf->point(nodes[0]));
    Vector p2(hsurf->point(nodes[1]));
    Vector p3(hsurf->point(nodes[2]));

    area = avInn_[*fi].length();

    get_cruse_weights(p1, p2, p3, s, r, area, cruse_weights);
    Vector centroid = (p1 + p2 + p3) / 3;

    hsurf->begin(ni); hsurf->end(nie);
    for (; ni != nie; ++ni)
    { //! for every node
      TriSurfMesh::Node::index_type ppi = *ni;
      Vector op(hsurf->point(ppi));

          if (ppi == nodes[0])       get_auto_g(p1, p2, p3, 0, g_values, s, r, R_W);
     else if (ppi == nodes[1])       get_auto_g(p1, p2, p3, 1, g_values, s, r, R_W);
     else if (ppi == nodes[2])       get_auto_g(p1, p2, p3, 2, g_values, s, r, R_W);
     else
     {
      get_g_coef(p1, p2, p3, op, s, r, centroid, g_coef);

      for (int i=0; i<7; i++)  temp[0][i] = g_coef[0][i]*R_W[0][i];

      Mult_X_trans(g_values, cruse_weights, temp);
      g_values.mult(area);
          } // else

      for (int i=0; i<3; ++i)
	auto_G[ppi][nodes[i]]+=g_values[i][0]*mult;
     }
  }
//  cout<<"\nauto_G ready";
  return;
}


void SetupBEMatrix::make_cross_G(TriSurfMeshHandle hsurf1, TriSurfMeshHandle hsurf2, DenseMatrixHandle &h_GG_,
                                 double in_cond, double out_cond, double op_cond)
{
  TriSurfMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TriSurfMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  h_GG_ = tmp;
  DenseMatrix& cross_G = *tmp;
  cross_G.zero();

  const double mult = 1/(2*PI)*((out_cond - in_cond)/op_cond);
//   out_cond and in_cond belong to hsurf2 and op_cond is the out_cond of hsurf1 for all the surfaces but the outermost surface which in op_cond=in_cond

  TriSurfMesh::Node::array_type nodes;

  TriSurfMesh::Node::iterator  ni, nie;
  TriSurfMesh::Face::iterator  fi, fie;

  DenseMatrix cruse_weights(3, 7);
  DenseMatrix g_coef(1, 7);
  DenseMatrix R_W(1,7); // Radon Points Weights
  DenseMatrix temp(1,7);
  DenseMatrix g_values(3, 1);

  double area;

  double sqrt15 = sqrt(15.0);
  R_W[0][0] = 9/40;
  R_W[0][1] = (155 + sqrt15) / 1200;
  R_W[0][2] = R_W[0][1];
  R_W[0][3] = R_W[0][1];
  R_W[0][4] = (155 - sqrt15) / 1200;
  R_W[0][5] = R_W[0][4];
  R_W[0][6] = R_W[0][4];

  double s = (1 - sqrt15) / 7;
  double r = (1 + sqrt15) / 7;

  hsurf2->begin(fi); hsurf2->end(fie);
  for (; fi != fie; ++fi)
  { //! find contributions from every triangle
    hsurf2->get_nodes(nodes, *fi);
    Vector p1(hsurf2->point(nodes[0]));
    Vector p2(hsurf2->point(nodes[1]));
    Vector p3(hsurf2->point(nodes[2]));

    area = avInn_[*fi].length();

    get_cruse_weights(p1, p2, p3, s, r, area, cruse_weights);
    Vector centroid = (p1 + p2 + p3) / 3;

    hsurf1->begin(ni); hsurf1->end(nie);
    for (; ni != nie; ++ni)
    { //! for every node
      TriSurfMesh::Node::index_type ppi = *ni;
      Vector op(hsurf1->point(ppi));
      get_g_coef(p1, p2, p3, op, s, r, centroid, g_coef);

      for (int i=0; i<7; i++)  temp[0][i] = g_coef[0][i]*R_W[0][i];

      Mult_X_trans(g_values, cruse_weights, temp);
      g_values.mult(area);

      for (int i=0; i<3; ++i)
	cross_G[ppi][nodes[i]]+=g_values[i][0]*mult;
     }
  }
//   cout<<"\ncross_G ready";
   return;
}


void SetupBEMatrix::make_cross_P(TriSurfMeshHandle hsurf1, TriSurfMeshHandle hsurf2, DenseMatrixHandle &h_PP_,
                                 double in_cond, double out_cond, double op_cond)
{

  TriSurfMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TriSurfMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  h_PP_ = tmp;
  DenseMatrix& cross_P = *tmp;
  cross_P.zero();

  const double mult = 1/(2*PI)*((out_cond - in_cond)/op_cond);
//   out_cond and in_cond belong to hsurf2 and op_cond is the out_cond of hsurf1 for all the surfaces but the outermost surface which in op_cond=in_cond
  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  int i;

  TriSurfMesh::Node::iterator  ni, nie;
  TriSurfMesh::Face::iterator  fi, fie;

  hsurf1->begin(ni); hsurf1->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf1->point(ppi);

    hsurf2->begin(fi); hsurf2->end(fie);
    for (; fi != fie; ++fi){ //! find contributions from every triangle

      hsurf2->get_nodes(nodes, *fi);
      Vector v1 = hsurf2->point(nodes[0]) - pp;
      Vector v2 = hsurf2->point(nodes[1]) - pp;
      Vector v3 = hsurf2->point(nodes[2]) - pp;

      getOmega(v1, v2, v3, coef);

      for (i=0; i<3; ++i)
	cross_P[ppi][nodes[i]]+=coef[0][i]*mult;
    }
  }
//   cout<<"\ncross_P ready";
 return;
}


void SetupBEMatrix::make_auto_P(TriSurfMeshHandle hsurf, DenseMatrixHandle &h_PP_,
                                 double in_cond, double out_cond, double op_cond)
{

  TriSurfMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  h_PP_ = tmp;
  DenseMatrix& auto_P = *tmp;
  auto_P.zero();

  const double mult = 1/(2*PI)*((out_cond - in_cond)/op_cond);  // op_cond=out_cond for all the surfaces but the outermost surface which in op_cond=in_cond

  TriSurfMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);

  TriSurfMesh::Node::iterator ni, nie;
  TriSurfMesh::Face::iterator fi, fie;

  unsigned int i;

  hsurf->begin(ni); hsurf->end(nie);

  for (; ni != nie; ++ni){ //! for every node
    TriSurfMesh::Node::index_type ppi = *ni;
    Point pp = hsurf->point(ppi);

    hsurf->begin(fi); hsurf->end(fie);
    for (; fi != fie; ++fi) { //! find contributions from every triangle

      hsurf->get_nodes(nodes, *fi);
      if (ppi!=nodes[0] && ppi!=nodes[1] && ppi!=nodes[2]){
	 Vector v1 = hsurf->point(nodes[0]) - pp;
	 Vector v2 = hsurf->point(nodes[1]) - pp;
	 Vector v3 = hsurf->point(nodes[2]) - pp;

	 getOmega(v1, v2, v3, coef);

	 for (i=0; i<3; ++i)
	   auto_P[ppi][nodes[i]]+=coef[0][i]*mult;
      }
    }
  }

  //! accounting for autosolid angle
  for (i=0; i<nnodes; ++i){
    auto_P[i][i] = -( 1+((out_cond - in_cond)/op_cond) ) -auto_P.sumOfRow(i);
  }
//  cout<<"\nauto_P ready";
  return;
}


//! precalculate triangles area
void SetupBEMatrix::calc_tri_area(TriSurfMeshHandle hsurf, vector<Vector>& areaV){

  TriSurfMesh::Face::iterator  fi, fie;
  TriSurfMesh::Node::array_type     nodes;

  hsurf->begin(fi); hsurf->end(fie);
  for (; fi != fie; ++fi) {
    hsurf->get_nodes(nodes, *fi);
    Vector v1 = hsurf->point(nodes[1]) - hsurf->point(nodes[0]);
    Vector v2 = hsurf->point(nodes[2]) - hsurf->point(nodes[1]);
    areaV.push_back(Cross(v1, v2)*0.5);
  }
  return;
}

} // end namespace BioPSE
