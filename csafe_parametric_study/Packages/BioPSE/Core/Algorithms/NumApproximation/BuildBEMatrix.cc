/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  BuildBEMatrix.cc:  class to build Boundary Elements matrix
 *
 *  Written by:
 *   Saeed Babaeizadeh
 *   Northeastern University
 *   January 2006
 *   Copyright (C) 2001 SCI Group
 */

#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildBEMatrix.h>

namespace BioPSE {

using namespace SCIRun;


inline void  BuildBEMatrix::getOmega(
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
  if (Nn < 0)  Omega = 2 * atan( d / Nn ) + 2*M_PI ;
  if (Nn == 0)
        if ( d > 0 ) Omega = M_PI;
    	        else  Omega = -M_PI;

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


inline void  BuildBEMatrix::get_cruse_weights(
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
  A.scalar_multiply(0.5/area);

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


inline void  BuildBEMatrix::get_g_coef(
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


inline void  BuildBEMatrix::get_auto_g(
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


inline double  BuildBEMatrix::get_new_auto_g(
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
  double teta = 0; if (area2<=AC) teta = asin( area2 / AC );

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
      sum = sum + sqrt( Abs(rhoj * rhoj_1) );
      rhoj_1 = rhoj;
    }
    sai_new = sum * sqrt( Abs(gama * sin(gama)) );
    delta = 0;
    if (sai_new + sai_old)
	    delta = Abs( (sai_new - sai_old) / (sai_new + sai_old) );
    sai_old = sai_new;
  }
   return sai_new;
}


inline double  BuildBEMatrix::do_radon_g(
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


void BuildBEMatrix::make_auto_G(TSMesh::handle_type hsurf, DenseMatrixHandle &h_GG_,
                                 double in_cond, double out_cond, double op_cond, vector<double> avInn_)
{
  TSMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  h_GG_ = tmp;
  DenseMatrix& auto_G = *tmp;
  auto_G.zero();

  const double mult = 1/(2*M_PI)*((out_cond - in_cond)/op_cond);  // op_cond=out_cond for all the surfaces but the outermost surface which in op_cond=in_cond

  TSMesh::Node::array_type nodes;

  TSMesh::Node::iterator ni, nie;
  TSMesh::Face::iterator fi, fie;
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

    area = avInn_[*fi];

    get_cruse_weights(p1, p2, p3, s, r, area, cruse_weights);
    Vector centroid = (p1 + p2 + p3) / 3;

    hsurf->begin(ni); hsurf->end(nie);
    for (; ni != nie; ++ni)
    { //! for every node
      TSMesh::Node::index_type ppi = *ni;
      Vector op(hsurf->point(ppi));

            if (ppi == nodes[0])       get_auto_g(p1, p2, p3, 0, g_values, s, r, R_W);
     else if (ppi == nodes[1])       get_auto_g(p1, p2, p3, 1, g_values, s, r, R_W);
     else if (ppi == nodes[2])       get_auto_g(p1, p2, p3, 2, g_values, s, r, R_W);
     else
     {
      get_g_coef(p1, p2, p3, op, s, r, centroid, g_coef);

      for (int i=0; i<7; i++)  temp[0][i] = g_coef[0][i]*R_W[0][i];

      Mult_X_trans(g_values, cruse_weights, temp);
      g_values.scalar_multiply(area);
          } // else

      for (int i=0; i<3; ++i)
	auto_G[ppi][nodes[i]]+=g_values[i][0]*mult;
     }
  }
  return;
}


void BuildBEMatrix::make_cross_G(TSMesh::handle_type hsurf1, TSMesh::handle_type hsurf2, DenseMatrixHandle &h_GG_,
                                 double in_cond, double out_cond, double op_cond, vector<double> avInn_)
{
  TSMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TSMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  h_GG_ = tmp;
  DenseMatrix& cross_G = *tmp;
  cross_G.zero();

  const double mult = 1/(2*M_PI)*((out_cond - in_cond)/op_cond);
//   out_cond and in_cond belong to hsurf2 and op_cond is the out_cond of hsurf1 for all the surfaces but the outermost surface which in op_cond=in_cond

  TSMesh::Node::array_type nodes;

  TSMesh::Node::iterator  ni, nie;
  TSMesh::Face::iterator  fi, fie;

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

    area = avInn_[*fi];

    get_cruse_weights(p1, p2, p3, s, r, area, cruse_weights);
    Vector centroid = (p1 + p2 + p3) / 3;

    hsurf1->begin(ni); hsurf1->end(nie);
    for (; ni != nie; ++ni)
    { //! for every node
      TSMesh::Node::index_type ppi = *ni;
      Vector op(hsurf1->point(ppi));
      get_g_coef(p1, p2, p3, op, s, r, centroid, g_coef);

      for (int i=0; i<7; i++)  temp[0][i] = g_coef[0][i]*R_W[0][i];

      Mult_X_trans(g_values, cruse_weights, temp);
      g_values.scalar_multiply(area);

      for (int i=0; i<3; ++i)
	cross_G[ppi][nodes[i]]+=g_values[i][0]*mult;
     }
  }
   return;
}


void BuildBEMatrix::make_cross_P(TSMesh::handle_type hsurf1, TSMesh::handle_type hsurf2, DenseMatrixHandle &h_PP_,
                                 double in_cond, double out_cond, double op_cond)
{

  TSMesh::Node::size_type nsize1; hsurf1->size(nsize1);
  TSMesh::Node::size_type nsize2; hsurf2->size(nsize2);
  DenseMatrix* tmp = new DenseMatrix(nsize1, nsize2);
  h_PP_ = tmp;
  DenseMatrix& cross_P = *tmp;
  cross_P.zero();

  const double mult = 1/(2*M_PI)*((out_cond - in_cond)/op_cond);
//   out_cond and in_cond belong to hsurf2 and op_cond is the out_cond of hsurf1 for all the surfaces but the outermost surface which in op_cond=in_cond
  TSMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);
  int i;

  TSMesh::Node::iterator  ni, nie;
  TSMesh::Face::iterator  fi, fie;

  hsurf1->begin(ni); hsurf1->end(nie);
  for (; ni != nie; ++ni){ //! for every node
    TSMesh::Node::index_type ppi = *ni;
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
 return;
}


void BuildBEMatrix::make_auto_P(TSMesh::handle_type hsurf, DenseMatrixHandle &h_PP_,
                                 double in_cond, double out_cond, double op_cond)
{

  TSMesh::Node::size_type nsize; hsurf->size(nsize);
  unsigned int nnodes = nsize;
  DenseMatrix* tmp = new DenseMatrix(nnodes, nnodes);
  h_PP_ = tmp;
  DenseMatrix& auto_P = *tmp;
  auto_P.zero();

  const double mult = 1/(2*M_PI)*((out_cond - in_cond)/op_cond);  // op_cond=out_cond for all the surfaces but the outermost surface which in op_cond=in_cond

  TSMesh::Node::array_type nodes;
  DenseMatrix coef(1, 3);

  TSMesh::Node::iterator ni, nie;
  TSMesh::Face::iterator fi, fie;

  unsigned int i;

  hsurf->begin(ni); hsurf->end(nie);

  for (; ni != nie; ++ni){ //! for every node
    TSMesh::Node::index_type ppi = *ni;
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
  return;
}


// precalculate triangles area
void BuildBEMatrix::pre_calc_tri_areas(TSMesh::handle_type hsurf, vector<double>& areaV){

  TSMesh::Face::iterator  fi, fie;
  TSMesh::Node::array_type     nodes;

  hsurf->begin(fi); hsurf->end(fie);
  for (; fi != fie; ++fi) areaV.push_back(hsurf->get_area(*fi));
  return;
}


} // end namespace BioPSE
