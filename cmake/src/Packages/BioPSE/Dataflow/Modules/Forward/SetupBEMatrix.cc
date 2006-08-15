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
 *  SetupBEMatrix.cc: 
 *
 *  Written by:
 *   Saeed Babaeizadeh - Northeastern University
 *   Michael Callahan - Department of Computer Science - University of Utah
 *   May, 2003
 *
 *   Copyright (C) 2003 SCI Group
 */


#include <Packages/BioPSE/Dataflow/Modules/Forward/SetupBEMatrix.h>


namespace BioPSE {

using namespace SCIRun;

unsigned int first_time_run = 1;
vector<int> field_generation_no_old, old_nesting;
vector<double> old_conductivities;

SetupBEMatrix::SetupBEMatrix(GuiContext *context):
  Module("SetupBEMatrix", context, Source, "Forward", "BioPSE")
{
}


SetupBEMatrix::~SetupBEMatrix()
{
}


void
SetupBEMatrix::execute()
{
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
  vector<int> field_generation_no_new;
  string condStr; double condVal;
  port_map_type::iterator pi = range.first;
  int input=-1, output=-1;
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

      if (field->get_property("in/out", condStr))
         if (condStr == "in")
           input = pi->second;
         else if  (condStr == "out")
           output = pi->second;

      fields.push_back(field);
      meshes.push_back(mesh);
      conductivities.push_back(Abs(condVal));
      if(first_time_run)
       {
        field_generation_no_old.push_back(-1);
        old_conductivities.push_back(-1);
        old_nesting.push_back(-1);
       }
      field_generation_no_new.push_back(field->generation);
    }
    ++pi;
  }

  first_time_run = 0;

   if (input==-1 || output==-1)
   {
     error(" You must define one source as the 'input' and another one as the 'output' ");
     return;
    }

  // Compute the nesting tree for the input meshes.
  vector<int> nesting;
  if (!compute_nesting(nesting, meshes))
  {
    error("Unable to compute a valid nesting for this set of surfaces.");
  }
  
  if (nesting[input] == (int)nesting.size())
	conductivities[output] = 0; // for ESI problem: the outermost surface is the input
  else
	conductivities[input] = 0; // does not matter, set to 0 only for the program to work right

   // Check to see if the input fields are new
   int new_fields = 0, new_nesting = 0;
   double new_conductivities = 0;
   int no_of_fields =  nesting.size();
   for (int i=0; i < no_of_fields; i++)
    {
      new_fields += Abs( field_generation_no_new[i] - field_generation_no_old[i] );
      new_nesting += Abs( nesting[i] - old_nesting[i] );
      new_conductivities += Abs( conductivities[i] - old_conductivities[i] );
      field_generation_no_old[i] = field_generation_no_new[i];
      old_nesting[i] = nesting[i];
      old_conductivities[i] = conductivities[i];
    }

   if(new_fields>(no_of_fields+2) || new_nesting || new_conductivities )  // If the input fields are new
       build_Zoi(meshes, nesting, conductivities, input, output, hZoi_);
   else
       remark("Field inputs are old. Resending stored matrix.");

   // -- sending handles to cloned objects
   MatrixHandle hzoi(hZoi_->clone());
   send_output_handle("BEM Forward Matrix", hzoi);

   return;
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
SetupBEMatrix::compute_parent(const vector<TSMesh::handle_type> &meshes,
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
SetupBEMatrix::compute_nesting(vector<int> &nesting,
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


void SetupBEMatrix::build_Zoi(const vector<TSMesh::handle_type> &meshes,
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
            BuildBEMatrix::make_auto_P(meshes[i], h_PP_, in_cond, out_cond, op_cond);
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
  int given_potentials_surface_index = hs; // This is the surface whose potentials are given i.e. heart surface
  BuildBEMatrix::pre_calc_tri_areas(meshes[given_potentials_surface_index], avInn_);

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

            BuildBEMatrix::make_auto_G(meshes[i], h_GG_, in_cond, out_cond, op_cond, avInn_);
          }
        else
          {
            in_cond = conductivities[given_potentials_surface_index];
            if (nesting[given_potentials_surface_index] == no_of_fields) out_cond = 0;
            else out_cond = conductivities[nesting[given_potentials_surface_index]];

            if (nesting[i] == no_of_fields) op_cond = conductivities[i];
            else op_cond = conductivities[nesting[i]];

            BuildBEMatrix::make_cross_G(meshes[i], meshes[given_potentials_surface_index], h_GG_, in_cond, out_cond, op_cond, avInn_);
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

  //DenseMatrix *omatrix = 0;
  TSMesh::Node::size_type nsize_hs, nsize_ms, nsize;
  meshes[given_potentials_surface_index]->size(nsize_hs);
  meshes[ms]->size(nsize_ms);

  DenseMatrixHandle omatrix;
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
          Concat_rows(*omatrix.get_rep() ,*hGth_.get_rep(), *GG[i].get_rep());
          hGth_ = omatrix;
        }
    }

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
	  Concat_cols(*omatrix.get_rep() ,*hPht_.get_rep(), *PP[i].get_rep());
          hPht_ = omatrix;
        }
    }

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
	  Concat_cols(*omatrix.get_rep() ,*hPmt_.get_rep(), *PP[i].get_rep());
          hPmt_ = omatrix;
        }
    }

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
       	Concat_rows(*omatrix.get_rep() ,*hPth_.get_rep(), *PP[i*no_of_fields+hs].get_rep());
        hPth_ = omatrix;
      }
     }

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
       	Concat_rows(*omatrix.get_rep() ,*hPtm_.get_rep(), *PP[i*no_of_fields+ms].get_rep());
        hPtm_ = omatrix;
      }
     }

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
	       Concat_cols(*omatrix.get_rep() ,*hPtt_row_.get_rep(), *PP[j].get_rep());
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
	 Concat_rows(*omatrix.get_rep() ,*hPtt_.get_rep(), *hPtt_row_.get_rep());
         hPtt_ = omatrix;
        }
    }

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
  hZbh_->scalar_multiply(-1.0);
  hZoi_ = hZbh_.get_rep();

  } // end else   if (no_of_fields>1)

  return;
}


} // end namespace BioPSE
