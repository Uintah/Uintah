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
 *  SetupEITMatrix.cc: solves EIT forward problems based on BEM.
 *
 *  Written by:
 *   Saeed Babaeizadeh - Northeastern University
 *   April 2006
 *
 */


#include <Packages/BioPSE/Dataflow/Modules/Forward/SetupEITMatrix.h>


namespace BioPSE {

using namespace SCIRun;

unsigned int first_time_run_EIT = 1;
vector<int> filed_generation_no_old_EIT, old_nesting_EIT;
vector<double> old_conductivities_EIT;

SetupEITMatrix::SetupEITMatrix(GuiContext *context):
  Module("SetupEITMatrix", context, Source, "Forward", "BioPSE")
{
}


SetupEITMatrix::~SetupEITMatrix()
{
}


void
SetupEITMatrix::execute()
{
  MatrixOPort* oportMatrix_ = (MatrixOPort *)get_oport("BEM Forward Matrix");
  port_range_type range = get_iports("Surface");
  if (range.first == range.second)
  {
    remark("No surfaces connected.");
    return;
  }

  // Gather up the surfaces from the input ports.
  vector<FieldHandle> fields;
  vector<TSMesh::handle_type> meshes;
  vector<double> conductivities;
  vector<int> filed_generation_no_new;
  string condStr; double condVal;
  port_map_type::iterator pi = range.first;

  while (pi != range.second)
  {

    FieldIPort *fip = (FieldIPort *)get_iport(pi->second);
    FieldHandle field;
    if (fip->get(field))
    {
      if (field.get_rep() == 0)
      {
	warning("Surface port '" + to_string(pi->second) + "' contained no data.");
	++pi;
	continue;
      }


      TSMesh *mesh = 0;
      if (!(mesh = dynamic_cast<TSMesh *>(field->mesh().get_rep())))
      {
	error("Surface port '" + to_string(pi->second) +
	      "' does not contain a TriSurfField");
	return;
      }

      if (!field->get_property("Inside Conductivity", condStr))
      {
	error("The 'Inside Conductivity' of the Surface port '" + to_string(pi->second) + "' was not set. It assumes to be zero!");
        condVal = 0;
      }
      else condVal = atof(condStr.c_str());

      fields.push_back(field);
      meshes.push_back(mesh);
      conductivities.push_back(Abs(condVal));
      if(first_time_run_EIT)
       {
        filed_generation_no_old_EIT.push_back(-1);
        old_conductivities_EIT.push_back(-1);
        old_nesting_EIT.push_back(-1);
       }
      filed_generation_no_new.push_back(field->generation);
    }
    ++pi;
  }

  first_time_run_EIT = 0;

  // Compute the nesting tree for the input meshes.
  vector<int> nesting;
  if (!compute_nesting(nesting, meshes))
  {
    error("Unable to compute a valid nesting for this set of surfaces.");
  }

  
   // Check to see if the input fields are new
   int new_fields = 0, new_nesting = 0;
   double new_conductivities = 0;
   int no_of_fields =  nesting.size();
   for (int i=0; i < no_of_fields; i++)
    {
      new_fields += Abs( filed_generation_no_new[i] - filed_generation_no_old_EIT[i] );
      new_nesting += Abs( nesting[i] - old_nesting_EIT[i] );
      new_conductivities += Abs( conductivities[i] - old_conductivities_EIT[i] );
      filed_generation_no_old_EIT[i] = filed_generation_no_new[i];
      old_nesting_EIT[i] = nesting[i];
      old_conductivities_EIT[i] = conductivities[i];
    }

   
   if(new_fields>(no_of_fields+2) || new_nesting || new_conductivities )  // If the input fields are new
       build_Zoi(meshes, nesting, conductivities, hZoi_);
   else
       remark("Field inputs are old. Resending stored matrix.");

   // -- sending handles to cloned objects
   MatrixHandle hzoi(hZoi_->clone());
   oportMatrix_->send_and_dereference(hzoi);

   return;
}

// C++ized MollerTrumbore97 Ray Triangle intersection test.
#define EPSILON 1.0e-6
bool
SetupEITMatrix::ray_triangle_intersect(double &t,
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
SetupEITMatrix::compute_intersections(vector<pair<double, int> >
&results,
				      const TSMesh::handle_type &mesh,
				      const Point &p, const Vector &v,
				      int marker) const
{
  TSMesh::Face::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  double t;
  while (itr != eitr)
  {
    TSMesh::Node::array_type nodes;
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
SetupEITMatrix::compute_parent(const vector<TSMesh::handle_type> &meshes,
                              int index)
{
  Point point;
  meshes[index]->get_center(point, TSMesh::Node::index_type(0));
  Vector dir(1.0, 1.0, 1.0);
  vector<pair<double, int> > intersections;

  unsigned int i;
  for (i = 0; i < (unsigned int)meshes.size(); i++)
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
      return (int)meshes.size();
    }
    counts[intersections[i].second]++;
  }

  // Indeterminant, we should intersect with ourselves.
  return (int)meshes.size();
}



bool
SetupEITMatrix::compute_nesting(vector<int> &nesting,
				const vector<TSMesh::handle_type> &meshes)
{
  nesting.resize(meshes.size());

  unsigned int i;
  for (i = 0; i < (unsigned int)meshes.size(); i++)
  {
    nesting[i] = compute_parent(meshes, i);
  }

  return true;
}


void SetupEITMatrix::build_Zoi(const vector<TSMesh::handle_type> &meshes,
                        vector<int> &nesting,
                        vector<double> &conductivities,
                        MatrixHandle &hZoi_){


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
             BuildBEMatrix::make_auto_P(meshes[i], h_PP_, out_cond, in_cond, op_cond);
          }
        else
          {
            in_cond = conductivities[j];
            if (nesting[j] == no_of_fields) out_cond = 0;
            else out_cond = conductivities[nesting[j]];

            if (nesting[i] == no_of_fields) op_cond = conductivities[i];
            else op_cond = conductivities[nesting[i]];

            BuildBEMatrix::make_cross_P(meshes[i], meshes[j], h_PP_, in_cond, out_cond, op_cond);
          }
        PP.push_back(h_PP_);
      }
  // PP End

  vector<DenseMatrixHandle> GG;
  DenseMatrixHandle h_GG_;

  // GG Begin
  vector<double>   avInn_;
  int hs;
  for (i=0; i<(int)nesting.size(); i++) 
	if (nesting[i] == (int)nesting.size()) hs = i;
  int given_currents_surface_index = hs; // This is the surface whose currents are given
  BuildBEMatrix::pre_calc_tri_areas(meshes[given_currents_surface_index], avInn_);

  for (i=0; i<no_of_fields; i++)
      {
        if (i==given_currents_surface_index)
          {
            in_cond = conductivities[i];
            if (nesting[i] == no_of_fields)  // The Outermost surface is different from the others.
               {out_cond = 0;
                op_cond = in_cond;}
            else
               {out_cond = conductivities[nesting[i]];
                op_cond = out_cond;}
            BuildBEMatrix::make_auto_G(meshes[i], h_GG_, out_cond, in_cond, op_cond, avInn_);
          }
        else
          {
            in_cond = conductivities[given_currents_surface_index];
            if (nesting[given_currents_surface_index] == no_of_fields) out_cond = 0;
            else out_cond = conductivities[nesting[given_currents_surface_index]];

            if (nesting[i] == no_of_fields) op_cond = conductivities[i];
            else op_cond = conductivities[nesting[i]];

            BuildBEMatrix::make_cross_G(meshes[i], meshes[given_currents_surface_index], h_GG_, in_cond, out_cond, op_cond, avInn_);
          }
            GG.push_back(h_GG_);
      }

 if (no_of_fields == 1) // only one surface
 {
  PP[hs]->invert(); 
  DenseMatrixHandle hZbb_ = new DenseMatrix(GG[hs]->nrows(), GG[hs]->ncols()); 
  Mult(*(hZbb_.get_rep()), *(PP[hs].get_rep()), *(GG[hs].get_rep())); 
  hZbb_->scalar_multiply(-1/conductivities[hs]);
  hZoi_ = hZbb_.get_rep();
 }
 else  if (no_of_fields == 2)  // two surfaces (one internal inhomogeneity)
  {
  int ms;
  (hs == 0) ? ms =1 : ms = 0;
  PP[ms*(no_of_fields+1)]->invert(); 
  DenseMatrixHandle hZbb_ = new DenseMatrix(GG[hs]->nrows(), GG[hs]->ncols()); 
  DenseMatrix tmp1(PP[hs*(no_of_fields-1)+1]->nrows(), PP[hs*(no_of_fields-1)+1]->ncols()); 
  Mult(tmp1, *PP[hs*(no_of_fields-1)+1].get_rep(), *PP[ms*(no_of_fields+1)].get_rep());  
  DenseMatrix tmp2(PP[hs*(no_of_fields+1)]->nrows(), PP[hs*(no_of_fields+1)]->ncols()); 
  Mult(tmp2, tmp1, *(PP[ms*(no_of_fields-1)+1].get_rep()));
  Add(1,*PP[hs*(no_of_fields+1)].get_rep() , -1, tmp2);  
  PP[hs*(no_of_fields+1)]->invert(); 
  Mult(tmp2, tmp1, *(GG[ms].get_rep()));
  Add(1,*GG[hs].get_rep() , -1, tmp2);  
  Mult(*(hZbb_.get_rep()), *(PP[hs*(no_of_fields+1)].get_rep()), *(GG[hs].get_rep()));
  hZbb_->scalar_multiply(-1/conductivities[hs]);
  hZoi_ = hZbb_.get_rep();
 
  } // end else   if (no_of_fields==2)

  return;
}

} // end namespace BioPSE
