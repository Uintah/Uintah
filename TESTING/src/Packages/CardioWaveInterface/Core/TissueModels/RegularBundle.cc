/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>

#include <Packages/CardioWaveInterface/Core/TissueModels/RegularBundle.h>
#include <vector>

namespace CardioWaveInterface {

using namespace SCIRun;

bool TissueModel_RegularBundle::create_mesh(FieldHandle& output)
{
  int numnodes_x = (numelems_x_ics_ + 2*numelems_x_ecs_)*numcellsx_ + 1;
  int numnodes_y = (numelems_y_ics_ + 2*numelems_y_ecs_)*numcellsy_ + 1;
  int numnodes_z = numelems_z_*numcellsz_ + numelems_bath_start_ + numelems_bath_end_ + 1;
  
  double dz = cell_length_/numelems_z_;
  double dx_ics = sqrt(cell_crosssection_)/numelems_x_ics_;
  double dy_ics = sqrt(cell_crosssection_)/numelems_y_ics_;
  double dx_ecs = (sqrt(cell_crosssection_/ics_vol_frac_) - sqrt(cell_crosssection_))/(2*numelems_x_ecs_);
  double dy_ecs = (sqrt(cell_crosssection_/ics_vol_frac_) - sqrt(cell_crosssection_))/(2*numelems_y_ecs_);


  StructHexVolMesh<HexTrilinearLgn<Point> >* omesh = scinew StructHexVolMesh<HexTrilinearLgn<Point> >(numnodes_x,numnodes_y,numnodes_z);
  MeshHandle mesh = dynamic_cast<Mesh *>(omesh);

  if (mesh.get_rep() == 0)
  {
    error("TissueModel_RegularBundle: Could not obtain memory for mesh");
    return (false);
  }

  GenericField<StructHexVolMesh<HexTrilinearLgn<Point> >,ConstantBasis<int>,FData3d<int,StructHexVolMesh<HexTrilinearLgn<Point> > > >* ofield =
     scinew GenericField<StructHexVolMesh<HexTrilinearLgn<Point> >,ConstantBasis<int>,FData3d<int,StructHexVolMesh<HexTrilinearLgn<Point> > > >(omesh);

  output = dynamic_cast<Field*>(ofield);
  if (output.get_rep() == 0)
  {
    error("TissueModel_RegularBundle: Could not obtain memory for field");
    return (false);
  }


  std::vector<double> x(numnodes_x);
  std::vector<int> xe(numnodes_x-1);

  x[0] = 0.0;
  int m = 1; int n = 1;
  for (int p=0; p<numelems_x_ecs_; p++, m++) {x[m] = x[m-1]+dx_ecs; xe[m-1] = 0; }
  for (int r=0; r<(numcellsx_-1); r++, n++)
  {
    for (int p=0; p<numelems_x_ics_;p++, m++) { x[m] = x[m-1]+dx_ics; xe[m-1] = n; }
    for (int p=0; p<2*numelems_x_ecs_;p++, m++) { x[m] = x[m-1]+dx_ecs; xe[m-1] = 0; }
  }
  for (int p=0; p<numelems_x_ics_;p++, m++) { x[m] = x[m-1]+dx_ics; xe[m-1] = n; }
  for (int p=0; p<numelems_x_ecs_; p++, m++) { x[m] = x[m-1]+dx_ecs; xe[m-1] = 0; }

  std::vector<double> y(numnodes_y);
  std::vector<int> ye(numnodes_y-1);

  y[0] = 0.0;
  m = 1; n = 1;
  for (int p=0; p<numelems_y_ecs_; p++, m++) {y[m] = y[m-1]+dy_ecs; ye[m-1] = 0; }
  for (int r=0; r<(numcellsy_-1); r++, n++)
  {
    for (int p=0; p<numelems_y_ics_;p++, m++) { y[m] = y[m-1]+dy_ics; ye[m-1] = n; }
    for (int p=0; p<2*numelems_y_ecs_;p++, m++) { y[m] = y[m-1]+dy_ecs; ye[m-1] = 0; }
  }
  for (int p=0; p<numelems_y_ics_;p++, m++) { y[m] = y[m-1]+dy_ics; ye[m-1] = n; }
  for (int p=0; p<numelems_y_ecs_; p++, m++) { y[m] = y[m-1]+dy_ecs; ye[m-1] = 0; }

  int cx, cy;
  if (disable_center_)
  {
    cx = numcellsx_/2;
    cy = numcellsy_/2;
    
    std::cout << "cx=" << cx << "\n"; 
    std::cout << "cy=" << cy << "\n"; 
  }  
        
  for (int i=0; i<numnodes_x; i++)
  {
    for (int j=0; j<numnodes_y; j++)
    {
      for (int k=0; k<numnodes_z; k++)
      {
        omesh->set_point(Point(x[i],y[j],dz*k),LatVolMesh<HexTrilinearLgn<Point> >::Node::index_type(omesh,i,j,k));
      }
    }
  }

  for (int i=0; i<numnodes_x-1; i++)
  {
    for (int j=0; j<numnodes_y-1; j++)
    {
      for (int k=0; k<numnodes_z-1; k++)
      {
        
        int ze = 0;
        if (k >= numelems_bath_start_ && k < numelems_bath_end_ + (numcellsz_*(numelems_z_))) ze = (k-numelems_bath_start_)/numelems_z_ + 1;
        if (disable_center_)
        {
          if ((j >= numelems_y_ecs_+(numelems_y_ics_+2*numelems_y_ecs_)*cy)&& (j < (-numelems_y_ecs_)+(numelems_y_ics_+2*numelems_y_ecs_)*(cy+1)) &&
              (i >= numelems_x_ecs_+(numelems_x_ics_+2*numelems_x_ecs_)*cx)&& (i < (-numelems_x_ecs_)+(numelems_x_ics_+2*numelems_x_ecs_)*(cx+1)))
          {
            ze = 0;
          }
        }
        if (ze && xe[i] && ye[j])
        {
          ofield->set_value(xe[i]+((numcellsx_)*(ye[j]-1))+(numcellsx_*numcellsy_)*(ze-1),LatVolMesh<HexTrilinearLgn<Point> >::Elem::index_type(omesh,i,j,k));
        }
        else
        {
          ofield->set_value(0,LatVolMesh<HexTrilinearLgn<Point> >::Elem::index_type(omesh,i,j,k));
        }
      }
    }
  }
 
  int zstart = static_cast<int>((numelems_z_/2))-static_cast<int>(((numconnectionx_)/2));
  int zend = static_cast<int>((numelems_z_/2))+static_cast<int>(((numconnectionx_+1)/2));
   
  int ystart = static_cast<int>((numelems_y_ics_/2))-static_cast<int>(((numconnectionx_)/2));
  int yend = static_cast<int>((numelems_y_ics_/2))+static_cast<int>(((numconnectionx_+1)/2));
  
     
  for (int i=0; i < numcellsx_-1; i++)
  {
    for (int j=0; j < numcellsy_; j++)
    {
      for (int k=0; k < numcellsz_; k++)
      {
        if (disable_center_)
        {
          if ((i==cx)||(i==(cx-1)) && j==cy) continue;
        }

        for (int p= (numelems_x_ecs_+numelems_x_ics_)*(i+1)+i*numelems_x_ecs_; p < (2*numelems_x_ecs_+numelems_x_ics_)*(i+1);p++)
        {
          for (int q = ystart+numelems_y_ecs_+j*(2*numelems_y_ecs_+numelems_y_ics_); q < yend+numelems_y_ecs_+j*(2*numelems_y_ecs_+numelems_y_ics_); q++)
          {
            for (int r= numelems_bath_start_+zstart+k*(numelems_z_); r < numelems_bath_start_+zend+k*(numelems_z_); r++)
            {
              ofield->set_value(1+i+j*numcellsx_+k*(numcellsx_*numcellsy_),LatVolMesh<HexTrilinearLgn<Point> >::Elem::index_type(omesh,p,q,r));
            }
          }
        }

        for (int p= (2*numelems_x_ecs_+numelems_x_ics_)*(i+1); p < (2*numelems_x_ecs_+numelems_x_ics_)*(i+1)+numelems_x_ecs_;p++)
        {
          for (int q = ystart+numelems_y_ecs_+j*(2*numelems_y_ecs_+numelems_y_ics_); q < yend+numelems_y_ecs_+j*(2*numelems_y_ecs_+numelems_y_ics_); q++)
          {
            for (int r= numelems_bath_start_+zstart+k*(numelems_z_); r < numelems_bath_start_+zend+k*(numelems_z_); r++)
            {
              ofield->set_value(2+i+j*numcellsx_+k*(numcellsx_*numcellsy_),LatVolMesh<HexTrilinearLgn<Point> >::Elem::index_type(omesh,p,q,r));
            }
          }
        }
      }
    }
  }



  zstart = static_cast<int>((numelems_z_/2))-static_cast<int>(((numconnectiony_)/2));
  zend = static_cast<int>((numelems_z_/2))+static_cast<int>(((numconnectiony_+1)/2));
   
  int xstart = static_cast<int>((numelems_x_ics_/2))-static_cast<int>(((numconnectiony_)/2));
  int xend = static_cast<int>((numelems_x_ics_/2))+static_cast<int>(((numconnectiony_+1)/2));
     
  for (int i=0; i < numcellsx_; i++)
  {
    for (int j=0; j < numcellsy_-1; j++)
    {
      for (int k=0; k < numcellsz_; k++)
      {
        if (disable_center_)
        {
          if ((j==cy)||(j==(cy-1)) && i==cx) continue;
        }

        for (int p = xstart+numelems_x_ecs_+i*(2*numelems_x_ecs_+numelems_x_ics_); p < xend+numelems_x_ecs_+i*(2*numelems_x_ecs_+numelems_x_ics_); p++)
        {
          for (int q= (numelems_y_ecs_+numelems_y_ics_)*(j+1)+j*numelems_y_ecs_; q < (2*numelems_y_ecs_+numelems_y_ics_)*(j+1);q++)
          {
            for (int r= numelems_bath_start_+zstart+k*(numelems_z_); r < numelems_bath_start_+zend+k*(numelems_z_); r++)
            {
              ofield->set_value(1+i+j*numcellsx_+k*(numcellsx_*numcellsy_),LatVolMesh<HexTrilinearLgn<Point> >::Elem::index_type(omesh,p,q,r));
            }
          }
        }

        for (int p = xstart+numelems_x_ecs_+i*(2*numelems_x_ecs_+numelems_x_ics_); p < xend+numelems_x_ecs_+i*(2*numelems_x_ecs_+numelems_x_ics_); p++)
        {
          for (int q= (2*numelems_y_ecs_+numelems_y_ics_)*(j+1); q < (2*numelems_y_ecs_+numelems_y_ics_)*(j+1)+numelems_y_ecs_;q++)
          {
            for (int r= numelems_bath_start_+zstart+k*(numelems_z_); r < numelems_bath_start_+zend+k*(numelems_z_); r++)
            {
              ofield->set_value(1+i+(j+1)*numcellsx_+k*(numcellsx_*numcellsy_),LatVolMesh<HexTrilinearLgn<Point> >::Elem::index_type(omesh,p,q,r));
            }
          }
        }
 
      }
    }
  }

  return (true); 
}


}
