/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/LoadBalancers/CostModelForecaster.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Util/DebugStream.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/Mat.h>

using namespace Uintah;
   
namespace Uintah {
  extern DebugStream g_profile_stats;
  extern DebugStream g_profile_stats2;
}

void
CostModelForecaster::addContribution( DetailedTask *task, double cost )
{
  const PatchSubset *patches=task->getPatches();
  
  if( patches == 0 ) {
    return;
  }
 
  //compute cost per cell so the measured time can be distributed poportionally by cells
  int num_cells=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch=patches->get(p);
    num_cells+=patch->getNumExtraCells();
  }
  double cost_per_cell=cost/num_cells;

  //loop through patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch=patches->get(p);

    d_execTimes[patch->getID()] +=patch->getNumExtraCells()*cost_per_cell;
  }
}
//______________________________________________________________________
//
void CostModelForecaster::outputError(const GridP grid) 
{
  static int iter=0;
  iter++;
  std::vector<std::vector<int> > num_particles;
  std::vector<std::vector<double> > costs;
  
  d_lb->collectParticles(grid.get_rep(),num_particles);
  getWeights(grid.get_rep(), num_particles,costs);

  double size=0;
  double sum_error_local = 0;
  double sum_aerror_local= 0;
  double max_error_local = 0;
  
  //__________________________________
  //
  for(int l=0;l<grid->numLevels();l++)
  {
    LevelP level=grid->getLevel(l);
    size+=level->numPatches();
    
    //__________________________________
    //
    for(int p=0;p<level->numPatches();p++)
    {
      const Patch* patch=level->getPatch(p);
      
      if(d_lb->getPatchwiseProcessorAssignment(patch)!=d_myworld->myRank()){
        continue;
      }
      
      double error = (d_execTimes[patch->getID()] - costs[l][p])/(d_execTimes[patch->getID()] + costs[l][p]);

//      std::cout << d_myworld->myRank() << " patch:" << patch->getID() << " exectTime: " << d_execTimes[patch->getID()] 
//           << " cost: " << costs[l][p] << " error: " << error << std::endl;
     
      IntVector low(patch->getCellLowIndex());
      IntVector high(patch->getCellHighIndex());
      
      if(g_profile_stats2.active()){
        g_profile_stats2 << "PROFILESTATS: " << iter << " " << fabs(error) << " " << l << " " 
            << low[0] << " " << low[1] << " " << low[2] << " " << high[0] << " " << high[1] << " " << high[2] << std::endl;
      }

      if(fabs(error)>max_error_local){
        max_error_local=fabs(error);
      }
      
      sum_error_local += error;
      sum_aerror_local+= fabs(error);
     }
  }
  
  double sum_error=0;
  double sum_aerror=0;
  double max_error=0;
  
  if(d_myworld->nRanks()>1){
    Uintah::MPI::Reduce(&sum_error_local, &sum_error, 1,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    Uintah::MPI::Reduce(&sum_aerror_local,&sum_aerror,1,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    Uintah::MPI::Reduce(&max_error_local, &max_error, 1,MPI_DOUBLE,MPI_MAX,0,d_myworld->getComm());
  }
  else
  {
    sum_error  = sum_error_local;
    sum_aerror = sum_aerror_local;
    max_error  = max_error_local;
  }

  if(d_myworld->myRank()==0 && g_profile_stats.active()) {
    sum_error/=size;
    sum_aerror/=size;
    g_profile_stats << "sMPE: " << sum_error << " sMAPE: " << sum_aerror << " MAXsPE: " << max_error << std::endl;
  }
}
//______________________________________________________________________
//
void CostModelForecaster::collectPatchInfo(const GridP grid, std::vector<PatchInfo> &patch_info) 
{

  std::vector<std::vector<int> > num_particles;
  d_lb->collectParticles(grid.get_rep(),num_particles);

  std::vector<PatchInfo> patchList;
  std::vector<int> num_patches(d_myworld->nRanks(),0);

  int total_patches=0;
  
  for(int l=0;l<grid->numLevels();l++) {

    const LevelP& level = grid->getLevel(l);
    total_patches+=level->numPatches();
    
    for (int p=0;p<level->numPatches();p++) {
      const Patch *patch = level->getPatch(p);
      //compute number of patches on each processor
      int owner=d_lb->getPatchwiseProcessorAssignment(patch);
      num_patches[owner]++;
      
      //if I own patch
      if(owner==d_myworld->myRank()){
        // add to patch list
        PatchInfo pinfo(num_particles[l][p],patch->getNumCells(),patch->getNumExtraCells()-patch->getNumCells(),d_execTimes[patch->getID()]);
        patchList.push_back(pinfo);
      }
    }
  }

  std::vector<int> displs(d_myworld->nRanks(),0), recvs(d_myworld->nRanks(),0);

  //compute recvs and displs
  for(int i=0;i<d_myworld->nRanks();i++){
    recvs[i] = num_patches[i]*sizeof(PatchInfo);
  }
  
  for(int i=1;i<d_myworld->nRanks();i++) {
    displs[i] = displs[i-1]+recvs[i-1];
  }
  
  patch_info.resize(total_patches);
  
  //allgather the patch info
  if(d_myworld->nRanks()>1){
    Uintah::MPI::Allgatherv(&patchList[0], patchList.size()*sizeof(PatchInfo),  MPI_BYTE,
                    &patch_info[0], &recvs[0], &displs[0], MPI_BYTE,
                    d_myworld->getComm());
  }
  else
  {
    patch_info=patchList;
  }

}
//______________________________________________________________________
//
//computes the least squares approximation to x given the NxM matrix A and the Nx1 vector b.
void min_norm_least_sq(std::vector<std::vector<double> > &A, std::vector<double> &b, std::vector<double> &x)
{
  int rows = A.size();
  int cols = A[0].size();

  //compute A^T*A
  static std::vector<std::vector<double> > ATA;
  static std::vector<double> ATb;
  //storing L in the bottom of the symmetric matrix ATA
  static std::vector<std::vector<double> > &L=ATA;

  //resize ATA to a MxM matrix 
  ATA.resize(cols);
  for(int i=0;i<cols;i++){
    ATA[i].resize(cols);
  }
  
  ATb.resize(cols);

  //initialize ATA and ATb to 0
  for (int i=0; i<cols; i++) {
    for (int j=0; j<cols; j++){
      ATA[i][j]=0;
    }
    ATb[i]=0;
  }

  //compute the top half of the symmetric matrix ATA
  for (int r=0; r<rows; r++) {
    for (int i=0;i<cols;i++){
      for (int j=0;j<=i;j++){
        ATA[i][j]+=A[r][j]*A[r][i];
      }
    }
  }

#if 0
  if(Parallel::getMPIRank()==0)
  {
    for (int i=0;i<cols;i++)
    {
      std::cout << "ATA " << i << ": ";
      for (int j=0;j<cols;j++)
      {
        std::cout << ATA[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }
#endif

  //compute ATb
  for (int r=0; r<rows; r++){
    for (int j=0; j<cols; j++){
      ATb[j] += A[r][j]*b[r];
    }
  }

#if 0
  if(Parallel::getMPIRank()==0)
  {
    std::cout << " ATB: "; 
    for(int j=0;j<cols; j++)
      std::cout << ATb[j] << " ";
    std::cout << std::endl;
  }
#endif

  //__________________________________
  //solve ATA*x=ATb for x using cholesky's algorithm 
  //to decompose ATA into L*LT

  //LLT decomposition
  for(int k=0;k<cols;k++)
  {
    double sum=0;
    for(int s=0;s<k;s++){  //Dot Product
      sum+=(L[k][s]*L[k][s]); 
    }

    L[k][k]=sqrt(ATA[k][k]-sum);

    for(int i=k+1;i<cols;i++){
      sum=0;
      
      for(int s=0;s<k;s++){
        sum+=(L[i][s]*L[k][s]); //Dot Product
      }

      L[i][k]=((ATA[i][k]-sum)/L[k][k]);
    }
  }

#if 0
  for (int i=0;i<cols;i++)
  {
    std::cout << "L " << i << ": ";
    for (int j=0;j<=i;j++)
    {
      std::cout << L[i][j] << " ";
    }
    std::cout << std::endl;
  }
#endif

  //Solve using FSA then BSA algorithm 

  static std::vector<double> y;
  y.resize(cols);
  
  //__________________________________
  //Forward Substitution algorithm
  for(int i=0;i<cols;i++){
    double sum=0;

    for(int j=0;j<i;j++){
      sum+=(L[i][j]*y[j]);
    }
    
    y[i]=(ATb[i]-sum)/L[i][i];
  }

  //__________________________________
  //Backwards Substitution algorithm
  for(int i=cols-1;i>=0;i--)
  {
    double sum=0;

    for(int j=i+1;j<cols;j++){
      sum+=(L[j][i]*x[j]);
    }

    x[i]=(y[i]-sum)/L[i][i];
  }
}

//______________________________________________________________________
//  See section 5.2.1.1 of Justin Luitjens Dissertation
void
CostModelForecaster::finalizeContributions( const GridP currentGrid )
{

  //least squares to compute coefficients
#if 0 //parallel

#else //serial
  //collect the patch information needed to compute the coefficients
  std::vector<PatchInfo> patch_info;
  collectPatchInfo(currentGrid,patch_info);

#if 0
  if(g_profile_stats.active() && d_myworld->myRank()==0){
    static int j=0;
    
    for(size_t i=0;i<patch_info.size();i++){
      g_profile_stats << j << " " << patch_info[i] << endl;
    }
    j++;
  }
#endif

  outputError(currentGrid);

  int rows=patch_info.size();

  //__________________________________
  //  Forming linear system, Eq. 5.3
  std::vector<int> fields;
  for(int i=0;i<3;i++){
  
    //__________________________________
    //If a column would make the matrix singular remove it.
    //this occurs if all patches have the same number of cells
    //or all patches have the same number of particles, etc.
    
    if( d_x[i]!=0) {  //if it has been previously detected as singualr then assume it will always be singular...
      int first_val=patch_info[0][i];
      size_t j;
      
      for(j=0;j<patch_info.size();j++){
      
        if(patch_info[j][i] != first_val){
        
          //std::cout << "patch_info[" << j << "][" << i <<"]:" << patch_info[j][i] << " first_val: " << first_val << std::endl;
          //add this field
          fields.push_back(i);
          break;
        }
      }
      
      //singular on this field, set its coefficent to 0
      if(j == patch_info.size()){  
        proc0cout << "Removing profiling field (i=" << i <<") '" << PatchInfo::type(i) << "' because it is singular\n";
        d_x[i]=0;
      }
    }
  }
  
  //__________________________________
  //add patch overhead field
  fields.push_back(3);

  int cols=fields.size();

  static std::vector<std::vector<double> > A;
  static std::vector<double> b;
  static std::vector<double> x;

  //resize vectors & matrix
  b.resize(rows);
  x.resize(cols);
  A.resize(rows);

  //set b vector and A matrix
  for(int i=0;i<rows;i++){
  
    b[i]=patch_info[i].execTime;
    A[i].resize(cols);
    
    //add fields to matrix
    for(size_t f=0;f<fields.size();f++){
      A[i][f] = patch_info[i][fields[f]];
    }
  }

  //compute least squares
  min_norm_least_sq(A,b,x);
  
#if 0
  if(d_myworld->myRank()==0){
    std::cout << " Coefficients: ";
    for(int i=0;i<cols;i++){
      std::cout << "x["<<i<<"]: "<< x[i]<< "\n";
    }
    std::cout << std::endl;
  }
#endif

#endif

  static int iter=0;
  iter++;
  
  // Eq. 5.5 in Dissertation
  double alpha=2.0/(std::min(iter,d_timestepWindow)+1);
  
  // Eq. 5.4
  //update coefficients using fading memory filter
  for(size_t f=0;f<fields.size();f++){
    d_x[fields[f]] = x[f]*alpha + d_x[fields[f]]*(1-alpha);
  }
  
  //update model coefficents
  setCosts(d_x[3], d_x[0], d_x[1], d_x[2]);
  
  if(d_myworld->myRank()==0 && g_profile_stats.active()){
    g_profile_stats << "Update: patchCost: " << d_patchCost << " cellCost: " << d_cellCost << " d_extraCellCost: " << d_extraCellCost << " particleCost: " << d_particleCost << std::endl;
  }
  
  d_execTimes.clear();
}

//______________________________________________________________________
//
void
CostModelForecaster::getWeights(const Grid* grid, std::vector<std::vector<int> > num_particles, std::vector<std::vector<double> >&costs)
{
  CostModeler::getWeights(grid,num_particles,costs);
}

//______________________________________________________________________
// 
std::ostream& operator<<(std::ostream& out, const CostModelForecaster::PatchInfo &pi)
{
  out << "NumCells: " << pi.num_cells << " NumExtraCells: " << pi.num_extraCells << " NumParticles: " << pi.num_particles << " ExecTime: " << pi.execTime ;
  return out;
}
