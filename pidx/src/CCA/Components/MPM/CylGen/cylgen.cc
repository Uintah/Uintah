// This is a circle packing code that generates a distribution of cylinders
// according to a user specified size distribution and a target volume fraction.
// User can either ask for a uniform distribution of particles based on a
// min and max diameter, or can specify a non-uniform distribution.


// To compile:

//  g++ -O3 -o cylgen cylgen.cc

// To run:

// >cylgen

// The code will create a PositionRadius.txt file, and also an xml file
// called Test2D.xml which is compatible for inclusion in a Uintah Problem
// Specification (ups) file.


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

bool isCylInsideRVE(double partDia, double RVEsize,
                       double xCent, double yCent);
//
bool isCylCenterInsideRVE(double RVEsize,
                          double xCent, double yCent);
//
bool doesCylIntersectOthers(double partDia, vector<vector<double> > diaLocs,
                            double &xCent, double &yCent,
                            vector<vector<double> > xLocs, 
                            vector<vector<double> > yLocs,
                            int i_min, int i_max);
//
void printCylLocs(vector<vector<double> > xLocs,
                     vector<vector<double> > yLocs,
                     vector<vector<double> > diaLocs,
                     int n_bins, const double RVEsize, double diam_max,
                     vector<double> sizes,vector<double> TVFS,double targetVF);
int main()
{

  // Parameters for user to change - BEGIN

  // set RNG seed
  srand48(0);

  bool uniform_size_distribution = false;
  bool specified_size_distribution = true;

  // RVEsize is the edge length of the 2D sample
  double RVEsize = 0.1;

  // targetVF is the total volume fraction (area fraction, really)
  // that you wish to reach
  double targetVF = 0.65;

  // If you just want to specify a minimum and maximum grain size and
  // have a distribution in between, you only need to change diam_min, diam_max
  // and n_sizes.
  // diam_min is the minimum grain diameter
  double diam_min = 0.000625;
  // diam_max is the maximum grain diameter
  double diam_max = 0.011250;
  // How many sizes of grains do you want to use
  int n_sizes = 10;
  // Parameters for user to change - END IF AN EQUAL SIZE DISTRIBUTION IS O.K.
  //  If you want to specify the distribution more carefully, see below.

  // Part of optimizing the search for intersections
  int n_bins = RVEsize/diam_max;
  cout << "n_bins = " << n_bins << endl;

  double bin_width = RVEsize/((double) n_bins);

  double RVE_area   =  (RVEsize*RVEsize);

  double diam_inc = (diam_max - diam_min)/((double) (n_sizes-1.));
  vector<double> sizes(n_sizes);
  vector<double>  TVFS(n_sizes);
  int num_cyls_this_size[n_sizes];
  double targetArea = RVE_area*targetVF;

  if(uniform_size_distribution){
    if(specified_size_distribution){
       cerr << "Both uniform_size_distribution and specified_size_distribution" << endl;
       cerr << "can't be true.  Specify which you want in the source code and recompile." << endl;
       exit(1);
    }
    // The following assumes an equal distribution.  Specifically
    // that an equal number of cyls will be generated for each size
    // area_of_one_of_each_size  will be the sum of the areas of a single one
    // of each of the spheres
    double area_of_one_of_each_size = 0.0;
    for(int i=0;i<n_sizes;i++){
      double diam=diam_max - diam_inc*((double) i);
      sizes[i]=diam;
      area_of_one_of_each_size += 0.25*M_PI*(diam*diam);
    }
    int num_cyls_each_size = targetArea/area_of_one_of_each_size;
    cout << "Number cylinders to be created of each size = "
         << num_cyls_each_size << endl;
//    cout << "Area of one of each size = "
//         << area_of_one_of_each_size << endl;

    // Grains will be created largest to smallest.  TVFS is the 
    // cumulative volume fraction achieved after creating all of the cylinders
    // of a particular size
    double vf = 0.;
    for(int i=0;i<n_sizes;i++){
      double diam=sizes[i];
      double area_of_one_of_this_size = 0.25*M_PI*(diam*diam);
      vf += num_cyls_each_size*area_of_one_of_this_size/targetArea;
      TVFS[i]=vf;
      TVFS[i]*=targetVF;
      cout << "TVFS[" << i << "] = " << TVFS[i] << ";" << endl;
      cout << "sizes[" << i << "] = " << sizes[i] << ";" << endl;
    }
  }  // uniform_distribution

  if(specified_size_distribution){
    if(uniform_size_distribution){
       cerr << "Both uniform_size_distribution and specified_size_distribution" << endl;
       cerr << "can't be true.  Specify which you want in the source code and recompile." << endl;
       exit(1);
    }

    // In the following, the user can specify the volume fraction for each size
    // of grain (TVFS) and the actual sizes below that.
    // At this point, the TVFS is NOT cumulative (as above), this will be
    // computed automatically below.
    // At this point, TVFS is the fraction of the total FILLED area, not the
    // fraction of the total area.  Thus, the sum of all specified TVFS values
    // must add up to 1.0

    //Note that if one wants
    // more than 10 distinct sizes, they will need to increase n_sizes above
    TVFS[0]=0.000;  // Set to zero because a single cyl of this size
                    // skews the distribution for such a small sample size.
    TVFS[1]=0.010;
    TVFS[2]=0.025;
    TVFS[3]=0.025;
    TVFS[4]=0.13;
    TVFS[5]=0.13;
    TVFS[6]=0.20;
    TVFS[7]=0.20;
    TVFS[8]=0.20;
    TVFS[9]=0.08;
    double tot = 0.;
    for(int i=0;i<n_sizes;i++){
      tot+=TVFS[i];
    }
    if(tot < .999 || tot > 1.001){
      cerr << "The sum of the specified TVFS values must add up to 1.0" << endl;
      cerr << "Thus, the value given for each entry is that size's " << endl;
      cerr << "fraction of the filled space, not the total space." << endl;
      cerr << "The sum of the specified area fractions is " << tot << endl;
      exit(1);
    }

    sizes[0]=0.013750;
    sizes[1]=0.011250;
    sizes[2]=0.009375;
    sizes[3]=0.008125;
    sizes[4]=0.006875;
    sizes[5]=0.005625;
    sizes[6]=0.004375;
    sizes[7]=0.003125;
    sizes[8]=0.001875;
    sizes[9]=0.000625;

    diam_max=sizes[0];

    for(int i=0;i<n_sizes;i++){
      double diam=sizes[i];
      double area_of_one_of_this_size = 0.25*M_PI*(diam*diam);
      double total_area_of_this_size = TVFS[i]*targetArea;
      cout << "total_area_of_this_size = " << total_area_of_this_size << endl;
      num_cyls_this_size[i] = total_area_of_this_size/area_of_one_of_this_size;
      TVFS[i]*=targetVF;
      cout << "num_cyls_this_size[" << i << "] = " << num_cyls_this_size[i] << endl;
      cout << "TVFS[" << i << "] = " << TVFS[i] << endl;
    }

    cout << "TVFS[0] = " << TVFS[0] << endl;
    for(int i=1;i<n_sizes;i++){
      TVFS[i]+=TVFS[i-1];
      cout << "TVFS[ " << i << "] = " << TVFS[i] << endl;
    }
  }  // specified_distribution


  /********************************************************************
  No need to make changes below here unless you really want to get into
  the guts of the algorithm
  ********************************************************************/

  //Store the locations in n_bins separate vectors, so we have a smaller region
  //to search for intersections
  vector<vector<double> > xLocs(n_bins);
  vector<vector<double> > yLocs(n_bins);
  vector<vector<double> > diaLocs(n_bins);

  double total_cyl_area = 0.0;
  double total_cyl_VF = 0.0;

  for(int i=0;i<n_sizes;i++){
    int num_cyls_created_this_size = 0;
    int intersected = 0;
    long int total_intersections = 0;
    double num_extra_bins_d = (sizes[i]+diam_max)/(2.0*bin_width);
    int num_extra_bins = (int) num_extra_bins_d + 1;
    cout << "NEB = " << num_extra_bins_d << " " << num_extra_bins << endl;
    while(total_cyl_VF < TVFS[i] && total_intersections < 20000000){
     // Get two random numbers for the x and y and scale by RVE size
     double xCent = drand48()*RVEsize;
     double yCent = drand48()*RVEsize;

     if(isCylCenterInsideRVE(RVEsize,xCent,yCent)){
      // Figure out which bin the cylinder would be in, and grab
      // (at most) one on either side of it
      int index = (xCent/RVEsize)*((double) n_bins);

      int index_min = max(0,index-num_extra_bins);
      int index_max = min(n_bins-1,index+num_extra_bins);
      bool cylsIntersect = doesCylIntersectOthers(sizes[i], diaLocs,
                                                  xCent, yCent, xLocs, yLocs,
                                                  index_min, index_max);
      if(cylsIntersect){
        intersected++;
      } else{
        // Recompute index as particle position may have moved to a different bin
        index = (xCent/RVEsize)*((double) n_bins);
        if(index>=0 && index < n_bins){
          xLocs[index].push_back(xCent);
          yLocs[index].push_back(yCent);
          diaLocs[index].push_back(sizes[i]);
          total_cyl_area += 0.25*M_PI*(sizes[i]*sizes[i]);
          total_intersections+=intersected;
          intersected = 0;
          total_cyl_VF = total_cyl_area/RVE_area; 
          num_cyls_created_this_size++;

          int num_cyls = 0;
          for(int k = 0;k<n_bins;k++){
             num_cyls+=xLocs[k].size();
          }

          if(!(num_cyls%1000)){
            cout << "Created cyl # " << num_cyls << endl;
            cout << "Total intersections so far = " 
                 << total_intersections << endl;
            cout << "total_cyl_VF = " << total_cyl_VF << endl;
            cout << "TVFS[" << i << "] = " << TVFS[i] << endl;
            printCylLocs(xLocs, yLocs, diaLocs,n_bins,RVEsize,diam_max,
                         sizes,TVFS,targetVF);
          } // end if
        }   // if index...
      }     // else
     }      // end ifCylCenterIsInsideRVE
    }       // end while
    int numCyls = 0;
    for(int k = 0;k<n_bins;k++){
      numCyls+=xLocs[k].size();
    }
    cout << numCyls << endl;
    cout << total_cyl_area/RVE_area << endl;
  }

  printCylLocs(xLocs, yLocs, diaLocs,n_bins,RVEsize,diam_max,
               sizes,TVFS,targetVF);
}

bool isCylInsideRVE(double partDia, double RVEsize, 
                       double xCent, double yCent)
{

    // Find if the particle fits in the box
    double rad = 0.5*partDia;
    double xMinPartBox = xCent-rad;
    double xMaxPartBox = xCent+rad;
    double yMinPartBox = yCent-rad;
    double yMaxPartBox = yCent+rad;
    if (xMinPartBox >= 0.0 && xMaxPartBox <= RVEsize &&
        yMinPartBox >= 0.0 && yMaxPartBox <= RVEsize) {
      return true;
    }
    return false;
}

bool isCylCenterInsideRVE(double RVEsize, 
                          double xCent, double yCent)
{

    // Find if the particle center is in the box
    if (xCent >= 0.0 && xCent <= RVEsize &&
        yCent >= 0.0 && yCent <= RVEsize) {
      return true;
    }
    return false;
}

bool doesCylIntersectOthers(double partDia, vector<vector<double> > diaLocs,
                            double &xCent, double &yCent,
                            vector<vector<double> > xLocs, 
                            vector<vector<double> > yLocs,
                            int i_min, int i_max)
{
#if 0
  int hitPart1, hitPart2;
  int hitIndex1, hitIndex2;
#endif
  bool intersect=false;
  int tests = 0;
  for(int k=i_min;k<=i_max;k++){
    for(unsigned int i = 0; i<xLocs[k].size(); i++){
      // Compute distance between centers
      double distCentSq = (xCent-xLocs[k][i])*(xCent-xLocs[k][i]) +
                          (yCent-yLocs[k][i])*(yCent-yLocs[k][i]);

      double sumRadSq = 0.25*(partDia + diaLocs[k][i])*(partDia + diaLocs[k][i]);
      tests++;
      if(sumRadSq > distCentSq){
        intersect = true;
//        hitPart1 = i;
//        hitIndex1 = k;
        //cout << tests << endl;
        break;
      }
    }
    if(intersect){
      //cout << "Intersected in first round" << endl;
      break;
    }
  }

  // None of the cyls intersected
  if(!intersect){
    return false;
  }
  else{
    return true;
  }

#if 0
    // Adjust the cyl position by moving it away from the cyl that
    // it intersected so that at least it isn't still hitting it, and try again
    intersect = false;
    double sep = .5*(partDia + diaLocs[hitIndex1][hitPart1]) + 1.e-14;
    double vecx = xCent-xLocs[hitIndex1][hitPart1];
    double vecy = yCent-yLocs[hitIndex1][hitPart1];
    double vec_mag = sqrt(vecx*vecx + vecy*vecy);
    xCent = xLocs[hitIndex1][hitPart1]+sep*vecx/vec_mag;
    yCent = yLocs[hitIndex1][hitPart1]+sep*vecy/vec_mag;

   if(isCylCenterInsideRVE(1.0, xCent, yCent)){
    for(int k=i_min;k<=i_max;k++){
     for(unsigned int i = 0; i<xLocs[k].size(); i++){
      // Compute distance between centers
      double distCentSq = (xCent-xLocs[k][i])*(xCent-xLocs[k][i]) +
                          (yCent-yLocs[k][i])*(yCent-yLocs[k][i]);

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
   }else{
      return true;
   }
  }
   // None of the cyls intersected
   if(!intersect){
      //cout << "Fixed intersection in second round!" << endl;
      return false;
  }
  else{
     // Adjust the cyl position by moving 
     double r_x = xLocs[hitIndex2][hitPart2] - xLocs[hitIndex1][hitPart1];
     double r_y = yLocs[hitIndex2][hitPart2] - yLocs[hitIndex1][hitPart1];
     double t1 = drand48();
     double t2 = drand48();
     double t_length = sqrt(t1*t1 + t2*t2);
     t1/=t_length;
     t2/=t_length;

     xCent = xLocs[hitIndex1][hitPart1] + 0.5*r_x
           + t1*(max(diaLocs[hitIndex1][hitPart1],diaLocs[hitIndex2][hitPart2])
           + partDia);
     yCent = yLocs[hitIndex1][hitPart1] + 0.5*r_y 
           + t1*(max(diaLocs[hitIndex1][hitPart1],diaLocs[hitIndex2][hitPart2])
               + partDia);

     if(isCylCenterInsideRVE(1.0, xCent, yCent)){
       for(int k=i_min;k<=i_max;k++){
        for(unsigned int i = 0; i<xLocs[k].size(); i++){
         // Compute distance between centers
         double distCentSq = (xCent-xLocs[k][i])*(xCent-xLocs[k][i]) +
                             (yCent-yLocs[k][i])*(yCent-yLocs[k][i]);
   
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

void printCylLocs(vector<vector<double> > xLocs, vector<vector<double> > yLocs,
                  vector<vector<double> > diaLocs,
                  int n_bins, const double RVEsize, double diam_max,
                  vector<double> sizes,vector<double> TVFS, double targetVF)
{
  //Open file to receive cyl descriptions
  string outfile_name = "Test2D.xml";
  ofstream dest(outfile_name.c_str());
  if(!dest){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
  dest << "<!--" << endl;
  dest << "targetVF = " << targetVF << endl;
  for(int i=0;i<sizes.size();i++){
      dest << "sizes[" << i << "] = " << sizes[i] << endl;
  }
  for(int i=0;i<sizes.size();i++){
      dest << "TVFS[" << i << "] = " << sizes[i] << endl;
  }
  dest << "-->" << endl; 

  dest << "<Uintah_Include>" << endl;
  dest << "<union>\n\n";

  int cylcount = 0;
  for(int k=0;k<n_bins;k++){
    for(unsigned int i = 0; i<xLocs[k].size(); i++){
         dest << "    <cylinder label = \"" << cylcount++ << "\">\n";
         dest << "       <top>[" << xLocs[k][i] << ", " << yLocs[k][i] << ", " << 10000 << "]</top>\n";
         dest << "       <bottom>[" << xLocs[k][i] << ", " << yLocs[k][i] << ", " << -10000.0 << "]</bottom>\n";
         dest << "       <radius>" << 0.5*diaLocs[k][i] << "</radius>\n";
         dest << "    </cylinder>\n";
    }
  }

  dest << "</union>\n\n";
  dest << "</Uintah_Include>" << endl;

  string outfile_name2 = "Position_Radius.txt";
  ofstream dest2(outfile_name2.c_str());
  if(!dest2){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest2.precision(15);

  for(int k=0;k<n_bins;k++){
   for(unsigned int i = 0; i<xLocs[k].size(); i++){
       dest2 <<  xLocs[k][i] << " " << yLocs[k][i] << " " << 0.5*diaLocs[k][i] << "\n";
   }
  }
}
