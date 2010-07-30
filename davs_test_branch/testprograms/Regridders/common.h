#ifndef RGCOMMON
#define RGCOMMON

int rank, num_procs;
class Sphere
{
  public:
    Sphere(Vector center, double rad) : center(center), rad(rad) {};

    bool intersects(Vector i)
    {
      Vector d=center-i;
      if(rad>sqrtf(d.x()*d.x()+d.y()*d.y()+d.z()*d.z()))
        return true;
      else
        return false;
    }
  private:
    Vector center;
    double rad;
};

class Sphere2
{
  public:
    Sphere2(Vector center, double rad_in, double rad_out) : center(center), rad_in(rad_in), rad_out(rad_out) {};

    bool intersects(Vector i)
    {
      Vector d=center-i;
      double p=sqrtf(d.x()*d.x()+d.y()*d.y()+d.z()*d.z());
      if(rad_out>p && rad_in<p)
        return true;
      else
        return false;
    }
  private:
    Vector center;
    double rad_in,rad_out;
};
unsigned int getTotalNumFlags(vector<IntVector> flags)
{
  unsigned int num=flags.size();
  unsigned int gnum;
  MPI_Allreduce(&num,&gnum,1,MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
  return gnum;
}
unsigned int getTotalNumPatches(vector<Region> patches)
{
  unsigned int num=patches.size();
  unsigned int gnum;
  MPI_Allreduce(&num,&gnum,1,MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
  return gnum;
}
void makeFlagsList(const vector<Region> &patches, vector<CCVariable<int>*> flags, vector<IntVector> &lflags)
{
  lflags.resize(0);
  for(unsigned int p=0;p<patches.size();p++)
  {
    for(CellIterator iter(patches[p].getLow(),patches[p].getHigh());!iter.done();iter++)
    {
      if( (*flags[p])[*iter] == 1 )
      {
        lflags.push_back(*iter);
      }
    }
  }
}
void outputPatches(vector<Region> &patches, ostream& out)
{
  if(rank==0)
  {
    for(unsigned int i=0;i<patches.size();i++)
    {
      Region p=patches[i];
      
      IntVector l=p.getLow();
      IntVector h=p.getHigh();

      out << l[0] << " " << l[1] << " " << l[2] << " " << h[0] << " " << h[1] << " " << h[2] << endl;
    }
  }
}
void gatherFlags(vector<IntVector> &flags, vector<IntVector> &gflags)
{
  vector<int> num_flags(num_procs);
  int num=flags.size();

  MPI_Allgather(&num,1,MPI_INT,&num_flags[0],1,MPI_INT,MPI_COMM_WORLD);

  vector<int> counts(num_procs);
  vector<int> displ(num_procs);
 
  int total=0;
  for(int i=0;i<num_procs;i++)
  {
    counts[i]=num_flags[i]*sizeof(IntVector);
    displ[i]=total*sizeof(IntVector);
    total+=num_flags[i];
  }

  gflags.resize(total);
  MPI_Allgatherv(&flags[0],counts[rank],MPI_BYTE,&gflags[0],&counts[0],&displ[0],MPI_BYTE,MPI_COMM_WORLD);
}
void outputFlags(vector<IntVector> &flags, ostream &out)
{
  vector<IntVector> global_flags;
  gatherFlags(flags,global_flags);
  
  if(rank==0)
  {
    for(unsigned int i=0;i<global_flags.size();i++)
    {
      IntVector f=global_flags[i];

      out << f[0] << " " << f[1] << " " << f[2] <<  endl;
    }
  }
}
void gatherPatches(vector<Region> &patches, vector<Region> &global_patches)
{
  vector<int> num_patches(num_procs);
  int num=patches.size();

  MPI_Allgather(&num,1,MPI_INT,&num_patches[0],1,MPI_INT,MPI_COMM_WORLD);

  vector<int> counts(num_procs);
  vector<int> displ(num_procs);
 
  int total=0;
  for(int i=0;i<num_procs;i++)
  {
    counts[i]=num_patches[i]*sizeof(Region);
    displ[i]=total*sizeof(Region);
    total+=num_patches[i];
  }

  global_patches.resize(total);
  MPI_Allgatherv(&patches[0],counts[rank],MPI_BYTE,&global_patches[0],&counts[0],&displ[0],MPI_BYTE,MPI_COMM_WORLD);
}

void splitPatches(vector<Region> &patches, vector<Region> &split_patches, double p)
{
  list<Region> to_split_patches(patches.begin(),patches.end());
  split_patches.clear();
  long long vol=0;
  for(size_t i=0;i<patches.size();i++)
    vol+=patches[i].getVolume();

  long long total_vol;
  MPI_Allreduce(&vol,&total_vol,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);

  //if(rank==0)
  //  cout << "local vol: " << vol << " total vol: " << total_vol << endl;

  long long thresh=total_vol/num_procs*p;


  if(thresh==0)
    thresh=1;

  while(!to_split_patches.empty())
  {
    Region patch=to_split_patches.back();
    to_split_patches.pop_back();
    //cout << "thresh: " << thresh << " vol: " << patch.getVolume() << endl;
    if( patch.getVolume()>thresh)
    {
      IntVector low=patch.getLow(), high=patch.getHigh();
      IntVector size=high-low;
      int max_d=0;
      if(size[max_d]<size[1])
        max_d=1;
      if(size[max_d]<size[2])
        max_d=2;
      int mid=(high[max_d]+low[max_d])/2;
      
      Region left(low,high), right(low,high);
      left.high()[max_d]=right.low()[max_d]=mid;
      //cout << "low: " << low << " high: " << high << " max_d: " << max_d << " mid: " << mid << endl;
      //cout << "Patch: " << patch << " left: " << left << " right: " << right << endl;
      to_split_patches.push_back(left);
      to_split_patches.push_back(right);
    }
    else 
    {
      split_patches.push_back(patch);
    }
  }
  //cout << "Patches before: " << patches.size() << " patches after: " << split_patches.size() << endl;
}

#endif
