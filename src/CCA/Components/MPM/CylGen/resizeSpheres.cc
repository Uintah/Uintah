#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

bool isSphereInsideRVE(double partDia, double RVEsize,
                       double xCent, double yCent, double zCent);

bool isSphereCenterInsideRVE(double RVEsize,
                             double xCent, double yCent, double zCent);

bool doesSphereIntersectOthers(double partDia, vector<double> diaLocs,
                               double &gap,
                               double &xCent, double &yCent, double &zCent,
                               vector<double> xLocs, 
                               vector<double> yLocs,
                               vector<double> zLocs,
                               int &i_this);

#if 1
void printSphereLocs(vector<vector<double> > xLocs, vector<vector<double> > yLocs,
                     vector<vector<double> > zLocs, vector<vector<double> > diaLocs,
                     int n_bins, int RVEsize, double diam_max);
#endif

int main()
{

  // Parameters for user to change - BEGIN
  double RVEsize = 1.0;
  double targetVF = 0.77;
  double diam_min = 0.008;
  double diam_max = 0.0350;
  int n_sizes = 10;
  int n_bins  = 20;
  // Parameters for user to change - END

  double RVE_vol   =  (RVEsize*RVEsize*RVEsize);

  //Store the locations in n_bins separate vectors, so we have a smaller region
  //to search for intersections
  vector<double> xLocs;
  vector<double> yLocs;
  vector<double> zLocs;
  vector<double> diaLocs;

  //Open file to receive sphere descriptions
  string infile_name = "Position_Radius.txt";
  ifstream source(infile_name.c_str());
  if(!source){
    cerr << "File " << infile_name << " can't be opened." << endl;
  }
  double x,y,z,r,d;

  int outOfRVE=0;
  while(source >> x >> y >> z >> r){
    if(isSphereCenterInsideRVE(1.0,x,y,z)){
     xLocs.push_back(x);
     yLocs.push_back(y);
     zLocs.push_back(z);
     diaLocs.push_back(2.*r);
    } else{
      outOfRVE++;
    }
  }

  double total_sphere_volume_orig = 0.0;
  double total_sphere_volume_new  = 0.0;
  cout << xLocs.size() << endl;

  int numInts=0;
  for(int i = 0;i<xLocs.size();i++){
    d = diaLocs[i];
    total_sphere_volume_orig+=(1./6.)*M_PI*(d*d*d);
    int i_this = i;
    double gap=9.e99;
    bool spheresIntersect = doesSphereIntersectOthers(d, diaLocs, gap,
                                                   xLocs[i],yLocs[i],zLocs[i],
                                                   xLocs, yLocs, zLocs, i_this);

    if(spheresIntersect){
      //cout << "spheres intersect" << endl;
      cout << xLocs[i] << " " << yLocs[i] << " " << zLocs[i] << " " << diaLocs[i] << endl;
      cout << xLocs[i_this] << " " << yLocs[i_this] << " " << zLocs[i_this] << " " << diaLocs[i_this] << endl;
      double rad1plusrad2 = 0.5*(diaLocs[i] + diaLocs[i_this]);
      double distCentSq = (xLocs[i_this]-xLocs[i])*(xLocs[i_this]-xLocs[i]) +
                          (yLocs[i_this]-yLocs[i])*(yLocs[i_this]-yLocs[i]) +
                          (zLocs[i_this]-zLocs[i])*(zLocs[i_this]-zLocs[i]);
      cout << rad1plusrad2 << " " << sqrt(distCentSq) << endl;
      numInts++;
      //exit(1);
    }
    if(gap>9.e90){
      gap = 0.0;
    }
    diaLocs[i] += 0.5*gap;
    d = diaLocs[i];
    total_sphere_volume_new+=(1./6.)*M_PI*(d*d*d);
  }

  cout << "numInts = " << numInts << endl;
  cout << "Spheres out of RVE = " << outOfRVE << endl;
  cout << "Total sphere volume orig = " << total_sphere_volume_orig << endl;
  cout << "Total sphere volume new  = " << total_sphere_volume_new  << endl;

  vector<vector<double> > xbinLocs(n_bins);
  vector<vector<double> > ybinLocs(n_bins);
  vector<vector<double> > zbinLocs(n_bins);
  vector<vector<double> > dbinLocs(n_bins);

  for(int i = 0; i<xLocs.size(); i++){
    int index = (xLocs[i]/RVEsize)*((double) n_bins);
    xbinLocs[index].push_back(xLocs[i]);
    ybinLocs[index].push_back(yLocs[i]);
    zbinLocs[index].push_back(zLocs[i]);
    dbinLocs[index].push_back(diaLocs[i]);
  }

  printSphereLocs(xbinLocs,ybinLocs,zbinLocs,dbinLocs,n_bins,RVEsize,diam_max);

}

bool isSphereInsideRVE(double partDia, double RVEsize, 
                       double xCent, double yCent, double zCent)
{

    // Find if the particle fits in the box
    double rad = 0.5*partDia;
    double xMinPartBox = xCent-rad;
    double xMaxPartBox = xCent+rad;
    double yMinPartBox = yCent-rad;
    double yMaxPartBox = yCent+rad;
    double zMinPartBox = zCent-rad;
    double zMaxPartBox = zCent+rad;
    if (xMinPartBox >= 0.0 && xMaxPartBox <= RVEsize &&
        yMinPartBox >= 0.0 && yMaxPartBox <= RVEsize &&
        zMinPartBox >= 0.0 && zMaxPartBox <= RVEsize) {
      return true;
    }
    return false;

}

bool isSphereCenterInsideRVE(double RVEsize, 
                             double xCent, double yCent, double zCent)
{

    // Find if the particle center is in the box
    if (xCent >= 0.0 && xCent <= RVEsize &&
        yCent >= 0.0 && yCent <= RVEsize &&
        zCent >= 0.0 && zCent <= RVEsize) {
      return true;
    }
    return false;

}

bool doesSphereIntersectOthers(double partDia, vector<double> diaLocs,
                               double &gap,
                               double &xCent, double &yCent, double &zCent,
                               vector<double> xLocs, 
                               vector<double> yLocs,
                               vector<double> zLocs,
                               int &i_this)
{
  for(unsigned int i = i_this+1; i<xLocs.size(); i++){
      // Compute distance between centers
      double distCent = sqrt((xCent-xLocs[i])*(xCent-xLocs[i]) +
                             (yCent-yLocs[i])*(yCent-yLocs[i]) +
                             (zCent-zLocs[i])*(zCent-zLocs[i]));

      double sumRad = 0.5*(partDia + diaLocs[i]);
      double space = distCent - sumRad;
      gap = min(gap, space);
      if(space < -1.e-4){
        i_this = i;
        return true;
      }
  }
  // None of the spheres intersected
  return false;

#if 0
  else{
    // Adjust the sphere position by moving it away from the sphere that
    // it intersected so that at least it isn't still hitting it, and try again
    intersect = false;
    double sep = .5*(partDia + diaLocs[hitIndex1][hitPart1]) + 1.e-14;
    double vecx = xCent-xLocs[hitIndex1][hitPart1];
    double vecy = yCent-yLocs[hitIndex1][hitPart1];
    double vecz = zCent-zLocs[hitIndex1][hitPart1];
    double vec_mag = sqrt(vecx*vecx + vecy*vecy + vecz*vecz);
    xCent = xLocs[hitIndex1][hitPart1]+sep*vecx/vec_mag;
    yCent = yLocs[hitIndex1][hitPart1]+sep*vecy/vec_mag;
    zCent = zLocs[hitIndex1][hitPart1]+sep*vecz/vec_mag;

    for(int k=i_min;k<=i_max;k++){
     for(unsigned int i = 0; i<xLocs[k].size(); i++){
      // Compute distance between centers
      double distCentSq = (xCent-xLocs[k][i])*(xCent-xLocs[k][i]) +
                          (yCent-yLocs[k][i])*(yCent-yLocs[k][i]) +
                          (zCent-zLocs[k][i])*(zCent-zLocs[k][i]);

      double sumRadSq = 0.25*(partDia + diaLocs[k][i])*(partDia + diaLocs[k][i]);

      if(sumRadSq > distCentSq){
        //cout << "Still Intersected in second round :(" << endl;
        intersect = true;
        hitPart2 = i;
        hitIndex2 = k;
        break;
      }
     }
     if(intersect){
       //cout << "Intersected in second round" << endl;
       break;
     }
    }
   }
   // None of the spheres intersected
   if(!intersect){
      //cout << "Fixed intersection in second round!" << endl;
      return false;
   }
   else{
     // Adjust the sphere position by moving 
     double r_x = xLocs[hitIndex2][hitPart2] - xLocs[hitIndex1][hitPart1];
     double r_y = yLocs[hitIndex2][hitPart2] - yLocs[hitIndex1][hitPart1];
     double r_z = zLocs[hitIndex2][hitPart2] - zLocs[hitIndex1][hitPart1];
     double t1 = drand48();
     double t2 = drand48();
     double t3 = -(r_x*t1+r_y*t2)/r_z;
     double t_length = sqrt(t1*t1 + t2*t2 + t3*t3);
     t1/=t_length;
     t2/=t_length;
     t3/=t_length;

     xCent = xLocs[hitIndex1][hitPart1] + 0.5*r_x
           + t1*(max(diaLocs[hitIndex1][hitPart1],diaLocs[hitIndex2][hitPart2])
           + partDia);
     yCent = yLocs[hitIndex1][hitPart1] + 0.5*r_y 
           + t1*(max(diaLocs[hitIndex1][hitPart1],diaLocs[hitIndex2][hitPart2])
               + partDia);
     zCent = zLocs[hitIndex1][hitPart1] + 0.5*r_z 
           + t1*(max(diaLocs[hitIndex1][hitPart1],diaLocs[hitIndex2][hitPart2])
               + partDia);

     if(isSphereCenterInsideRVE(2.0, xCent, yCent, zCent)){
       for(int k=i_min;k<=i_max;k++){
        for(unsigned int i = 0; i<xLocs[k].size(); i++){
         // Compute distance between centers
         double distCentSq = (xCent-xLocs[k][i])*(xCent-xLocs[k][i]) +
                             (yCent-yLocs[k][i])*(yCent-yLocs[k][i]) +
                             (zCent-zLocs[k][i])*(zCent-zLocs[k][i]);
   
         double sumRadSq = 0.25*(partDia + diaLocs[k][i])*(partDia + diaLocs[k][i]);

         if(sumRadSq > distCentSq){
           //cout << "STILL Intersected in third round :(" << endl;
           return true;
         }
        }
       }
     }
     else{
       //cout << "STILL Intersected in third round " << endl;
       return true;
     }
     //cout << " FIXED Intersection in third round" << endl;
     return false;
   }
#endif
}

#if 1
void printSphereLocs(vector<vector<double> > xLocs, vector<vector<double> > yLocs,
                     vector<vector<double> > zLocs, vector<vector<double> > diaLocs,
                     int n_bins, int RVEsize, double diam_max)
{
  //Open file to receive sphere descriptions
  string outfile_name = "Test3D.xml";
  ofstream dest(outfile_name.c_str());
  if(!dest){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
  dest << "<Uintah_Include>" << endl;
  dest << "<union>\n\n";

  int spherecount = 0;
  for(int k=0;k<n_bins;k++){
    dest << " <intersection label = \"intersection" << k << "\">\n";
    dest << "  <box label = \"box" << k << "\">\n";
    dest << "     <min>[" << ((double) k)*(RVEsize/((double) n_bins)) - diam_max/2.0 << ",0.0,0.0]</min>" << endl;
    dest << "     <max>[" << ((double) k+1)*(RVEsize/((double) n_bins)) + diam_max/2.0 << "," << RVEsize << ", " << RVEsize << "]</max>" << endl;
    dest << "  </box>\n";
    dest << "  <union>\n";
    for(unsigned int i = 0; i<xLocs[k].size(); i++){
         dest << "    <sphere label = \"" << spherecount++ << "\">\n";
         dest << "       <origin>[" << xLocs[k][i] << ", " << yLocs[k][i] << ", " << zLocs[k][i] << "]</origin>\n";
         dest << "       <radius>" << 0.5*diaLocs[k][i] << "</radius>\n";
         dest << "    </sphere>\n";
    }
    dest << "  </union>\n\n";
    dest << " </intersection>\n\n";
  }

  dest << "</union>\n\n";
  dest << "</Uintah_Include>" << endl;

  string outfile_name2 = "Position_Radius.txt";
  ofstream dest2(outfile_name2.c_str());
  if(!dest2){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  for(int k=0;k<n_bins;k++){
   for(unsigned int i = 0; i<xLocs[k].size(); i++){
       dest2 <<  xLocs[k][i] << " " << yLocs[k][i] << " " << zLocs[k][i] << " " << 0.5*diaLocs[k][i] << "\n";
   }
  }
}
#endif
