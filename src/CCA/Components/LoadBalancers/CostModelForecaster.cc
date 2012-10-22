/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/LoadBalancers/CostModelForecaster.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Util/DebugStream.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/Mat.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
   
namespace Uintah
{
static DebugStream stats("ProfileStats", false);
static DebugStream stats2("ProfileStats2", false);
void
CostModelForecaster::addContribution( DetailedTask *task, double cost )
{
  const PatchSubset *patches=task->getPatches();
  
  if( patches == 0 ) {
    return;
  }
 
  //compute cost per cell so the measured time can be distributed poportionally by cells
  int num_cells=0;
  for(int p=0;p<patches->size();p++)
  {
    const Patch* patch=patches->get(p);
    num_cells+=patch->getNumExtraCells();
  }
  double cost_per_cell=cost/num_cells;

  //loop through patches
  for(int p=0;p<patches->size();p++)
  {
    const Patch* patch=patches->get(p);

    execTimes[patch->getID()]+=patch->getNumExtraCells()*cost_per_cell;
  }
}

void CostModelForecaster::outputError(const GridP grid) 
{
  static int iter=0;
  iter++;
  vector<vector<int> > num_particles;
  vector<vector<double> > costs;
  d_lb->collectParticles(grid.get_rep(),num_particles);
  getWeights(grid.get_rep(), num_particles,costs);

  double size=0;
  double sum_error_local=0,sum_aerror_local=0,max_error_local=0;
  for(int l=0;l<grid->numLevels();l++)
  {
    LevelP level=grid->getLevel(l);
    size+=level->numPatches();
    for(int p=0;p<level->numPatches();p++)
    {
      const Patch* patch=level->getPatch(p);
      
      if(d_lb->getPatchwiseProcessorAssignment(patch)!=d_myworld->myrank())
        continue;

      //cout << d_myworld->myrank() << " patch:" << patch->getID() << " exectTime: " << execTimes[patch->getID()] << " cost: " << costs[l][p] << endl;
      double error=(execTimes[patch->getID()]-costs[l][p])/(execTimes[patch->getID()]+costs[l][p]);
      IntVector low(patch->getCellLowIndex()), high(patch->getCellHighIndex());
      if(stats2.active())
        cout << "PROFILESTATS: " << iter << " " << fabs(error) << " " << l << " " 
            << low[0] << " " << low[1] << " " << low[2] << " " << high[0] << " " << high[1] << " " << high[2] << endl;

      if(fabs(error)>max_error_local)
        max_error_local=fabs(error);
      sum_error_local+=error;
      sum_aerror_local+=fabs(error);
     }
  }
  double sum_error=0,sum_aerror=0,max_error=0;
  if(d_myworld->size()>1)
  {
    MPI_Reduce(&sum_error_local,&sum_error,1,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    MPI_Reduce(&sum_aerror_local,&sum_aerror,1,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    MPI_Reduce(&max_error_local,&max_error,1,MPI_DOUBLE,MPI_MAX,0,d_myworld->getComm());
  }
  else
  {
    sum_error=sum_error_local;
    sum_aerror=sum_aerror_local;
    max_error=max_error_local;
  }

  if(d_myworld->myrank()==0 && stats.active())
  {
    sum_error/=size;
    sum_aerror/=size;
    cout << "sMPE: " << sum_error << " sMAPE: " << sum_aerror << " MAXsPE: " << max_error << endl;
  }
}
void CostModelForecaster::collectPatchInfo(const GridP grid, vector<PatchInfo> &patch_info) 
{

  vector<vector<int> > num_particles;
  d_lb->collectParticles(grid.get_rep(),num_particles);

  vector<PatchInfo> patchList;
  vector<int> num_patches(d_myworld->size(),0);

  int total_patches=0;
  //for each level
  for(int l=0;l<grid->numLevels();l++) {
    //for each patch
    const LevelP& level = grid->getLevel(l);
    total_patches+=level->numPatches();
    for (int p=0;p<level->numPatches();p++) {
      const Patch *patch = level->getPatch(p);
      //compute number of patches on each processor
      int owner=d_lb->getPatchwiseProcessorAssignment(patch);
      num_patches[owner]++;
      //if I own patch
      if(owner==d_myworld->myrank())
      {
        // add to patch list
        PatchInfo pinfo(num_particles[l][p],patch->getNumCells(),patch->getNumExtraCells()-patch->getNumCells(),execTimes[patch->getID()]);
        patchList.push_back(pinfo);
      }
    }
  }

  vector<int> displs(d_myworld->size(),0), recvs(d_myworld->size(),0);

  //compute recvs and displs
  for(int i=0;i<d_myworld->size();i++)
    recvs[i]=num_patches[i]*sizeof(PatchInfo);
  for(int i=1;i<d_myworld->size();i++)
    displs[i]=displs[i-1]+recvs[i-1];

  patch_info.resize(total_patches);
  //allgather the patch info
  if(d_myworld->size()>1)
  {
    MPI_Allgatherv(&patchList[0], patchList.size()*sizeof(PatchInfo),  MPI_BYTE,
                    &patch_info[0], &recvs[0], &displs[0], MPI_BYTE,
                    d_myworld->getComm());
  }
  else
  {
    patch_info=patchList;
  }

}

//computes the least squares approximation to x given the NxM matrix A and the Nx1 vector b.
void min_norm_least_sq(vector<vector<double> > &A, vector<double> &b, vector<double> &x)
{
  int rows=A.size();
  int cols=A[0].size();

  //compute A^T*A
  static vector<vector<double> > ATA;
  static vector<double> ATb;
  //storing L in the bottom of the symmetric matrix ATA
  static vector<vector<double> > &L=ATA;

  //resize ATA to a MxM matrix 
  ATA.resize(cols);
  for(int i=0;i<cols;i++)
  {
    ATA[i].resize(cols);
  }
  ATb.resize(cols);

  //initialize ATA and ATb to 0
  for (int i=0; i<cols; i++) {
    for (int j=0; j<cols; j++)
      ATA[i][j]=0;
    ATb[i]=0;
  }

  //compute the top half of the symmetric matrix ATA
  for (int r=0; r<rows; r++) 
  {
    for (int i=0;i<cols;i++)
      for (int j=0;j<=i;j++)
        ATA[i][j]+=A[r][j]*A[r][i];
  }

#if 0
  if(Parallel::getMPIRank()==0)
  {
    for (int i=0;i<cols;i++)
    {
      cout << "ATA " << i << ": ";
      for (int j=0;j<cols;j++)
      {
        cout << ATA[i][j] << " ";
      }
      cout << endl;
    }
  }
#endif

  //compute ATb
  for (int r=0; r<rows; r++)
    for (int j=0; j<cols; j++)
      ATb[j] += A[r][j]*b[r];

#if 0
  if(Parallel::getMPIRank()==0)
  {
    cout << " ATB: "; 
    for(int j=0;j<cols; j++)
      cout << ATb[j] << " ";
    cout << endl;
  }
#endif

  //solve ATA*x=ATb for x using cholesky's algorithm 
  //to decompose ATA into L*LT

  //LLT decomposition
  for(int k=0;k<cols;k++)
  {
    double sum=0;
    for(int s=0;s<k;s++)  //Dot Product
      sum+=(L[k][s]*L[k][s]); 

    L[k][k]=sqrt(ATA[k][k]-sum);

    for(int i=k+1;i<cols;i++)
    {
      sum=0;
      for(int s=0;s<k;s++)
        sum+=(L[i][s]*L[k][s]); //Dot Product

      L[i][k]=((ATA[i][k]-sum)/L[k][k]);
    }
  }

#if 0
  for (int i=0;i<cols;i++)
  {
    cout << "L " << i << ": ";
    for (int j=0;j<=i;j++)
    {
      cout << L[i][j] << " ";
    }
    cout << endl;
  }
#endif

  //Solve using FSA then BSA algorithm 

  static vector<double> y;
  y.resize(cols);

  //Forward Substitution algorithm
  for(int i=0;i<cols;i++)
  {
    double sum=0;

    for(int j=0;j<i;j++)
      sum+=(L[i][j]*y[j]);

    y[i]=(ATb[i]-sum)/L[i][i];
  }

  //Backwards Substitution algorithm
  for(int i=cols-1;i>=0;i--)
  {
    double sum=0;

    for(int j=i+1;j<cols;j++)
      sum+=(L[j][i]*x[j]);

    x[i]=(y[i]-sum)/L[i][i];
  }
}
void
CostModelForecaster::finalizeContributions( const GridP currentGrid )
{

  //least squares to compute coefficients
#if 0 //parallel

#else //serial
  //collect the patch information needed to compute the coefficients
  vector<PatchInfo> patch_info;
  collectPatchInfo(currentGrid,patch_info);

#if 0
  if(stats.active() && d_myworld->myrank()==0)
  {
    static int j=0;
    for(size_t i=0;i<patch_info.size();i++)
    {
      stats << j << " " << patch_info[i] << endl;
    }
    j++;
  }
#endif

  outputError(currentGrid);

  int rows=patch_info.size();

  vector<int> fields;
  for(int i=0;i<3;i++)
  {
    //If a column would make the matrix singular remove it.
    //this occurs if all patches have the same number of cells
    //or all patches have the same number of particles, etc.
    if(d_x[i]!=0) //if it has been previously detected as singualr then assume it will always be singular...
    {
      int first_val=patch_info[0][i];
      size_t j;
      for(j=0;j<patch_info.size();j++)
      {
        if(patch_info[j][i]!=first_val)
        {
          //cout << "patch_info[" << j << "][" << i <<"]:" << patch_info[j][i] << " first_val: " << first_val << endl;
          //add this field
          fields.push_back(i);
          break;
        }
      }
      if(j==patch_info.size())
      {
        //singular on this field, set its coefficent to 0
        if(d_myworld->myrank()==0)
          cout << "Removing profiling field '" << PatchInfo::type(i) << "' because it is singular\n";

        d_x[i]=0;
      }
    }
  }
  //add patch overhead field
  fields.push_back(3);

  int cols=fields.size();

  static vector<vector<double> > A;
  static vector<double> b,x;

  //resize vectors
  b.resize(rows);
  x.resize(cols);
  //resize matrix
  A.resize(rows);

  //set b vector and A matrix
  for(int i=0;i<rows;i++)
  {
    b[i]=patch_info[i].execTime;
    A[i].resize(cols);
    //add fields to matrix
    for(size_t f=0;f<fields.size();f++)
      A[i][f]=patch_info[i][fields[f]];
  }

  //compute least squares
  min_norm_least_sq(A,b,x);
#if 0
  if(d_myworld->myrank()==0)
  {
    cout << " Coefficients: ";
    for(int i=0;i<cols;i++)
      cout << x[i] << " ";
    cout << endl;
  }
#endif

#endif

  static int iter=0;
  iter++;
  double alpha=2.0/(min(iter,d_timestepWindow)+1);
  //update coefficients using fading memory filter
  for(size_t f=0;f<fields.size();f++)
    d_x[fields[f]]=x[f]*alpha+d_x[fields[f]]*(1-alpha);

  //update model coefficents
  setCosts(d_x[3], d_x[0], d_x[1], d_x[2]);
  
  if(d_myworld->myrank()==0 && stats.active())
    cout << "Update: patchCost: " << d_patchCost << " cellCost: " << d_cellCost << " d_extraCellCost: " << d_extraCellCost << " particleCost: " << d_particleCost << endl;
  execTimes.clear();
}

void
CostModelForecaster::getWeights(const Grid* grid, vector<vector<int> > num_particles, vector<vector<double> >&costs)
{
  CostModeler::getWeights(grid,num_particles,costs);
}
  
ostream& operator<<(ostream& out, const CostModelForecaster::PatchInfo &pi)
{
  out << pi.num_cells << " " << pi.num_extraCells << " " << pi.num_particles << " " << pi.execTime ;
  return out;
}
}
