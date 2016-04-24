/*
 * This project constitutes a work of the United States Government and is not
 * subject to domestic copyright protection under 17 USC § 105.
 *
 * However, because the project utilizes code licensed from contributors and other
 * third parties, it therefore is licensed under the MIT License.
 * http://opensource.org/licenses/mit-license.php.
 *
 * Under that license, permission is granted free of charge, to any
 * person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the conditions that any
 * appropriate copyright notices and this permission notice are
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// C++ headers
#include <cstring>					// memcpy
#include <cmath>					// sqrt
#include <algorithm>				// min, max
#include <complex>
#include <sstream>					// std::cerr, std::cout, std::endl
#include <stdexcept>				// std::runtime_error
#include <iostream>
#include <fstream>              // ifstream
#include <iomanip>                              // std::setw, std::setprecision
#include <vector>
#include <string>
#include <cassert>               // assert()
#include <stdio.h>              // printf()

// Posix Headers
#include <unistd.h>				// getpid()

// PTR headers
#include "MersenneTwister.h"    // MTRand class
#include "PortableTongeRamesh.h"
#include "PortableMieGruneisenEOSTemperature.h"
#include "Vector3.h"
#include "Matrix3x3.h"
#include "FastMatrix.h"
#include "PState.h"

// GFMS headers (Granular Flow Multi Stage)
#include "GFMS_full.h" 

namespace PTR	// Portable TongeRamesh
{
  std::string getHistVarName(const int histVarNum){
    std::string histVarName("unknown");
    if(histVarNum < PTR_FLAWHIST_OFFSET){
      histVarName = PTR_StateVarNames[histVarNum];
    } else {
      int binNum = (histVarNum-PTR_FLAWHIST_OFFSET)/PTR_NUM_FLAW_HIST_PARAM;
      int flawHistIndex = (histVarNum-PTR_FLAWHIST_OFFSET)%PTR_NUM_FLAW_HIST_PARAM;
      std::ostringstream oss;
      oss << PTR_FlawDistStateBaseNames[flawHistIndex]
          << "_" << binNum;
      histVarName = oss.str();
    }
    return histVarName;
  }
  
  std::string getMatParamName(const int paramNumber){
    std::string paramName("unknown");
    if(paramNumber<PTR_NUM_MAT_PARAMS){
      paramName = PTR_MatParamNames[paramNumber];
    }
    return paramName;
  }

  void parseFlawDistData(double flawDistParam[PTR_NUM_FLAW_DIST_PARAM],
                         const std::string inputFileName){
    std::ifstream is(inputFileName.c_str());
    std::string line;
    std::string tag;
    double val;
    if(!is.good()){
      throw std::runtime_error("File open operation failed");
    }
    bool flawDistParamSet[PTR_NUM_FLAW_DIST_PARAM] = {};
    while(std::getline(is,line)){
      std::istringstream iss(line);
      iss >> tag >> val;
      if(iss){
        for (int i=0; i<PTR_NUM_FLAW_DIST_PARAM; ++i){
          if(tag.compare(PTR_FlawDistParamNames[i]) == 0){
            flawDistParam[i] = val;
            flawDistParamSet[i] = true;
            break;
          }
        }
      }
    }
    is.close();
    bool allParamsSet=true;
    for (int i=0; i<PTR_NUM_FLAW_DIST_PARAM; ++i){
      allParamsSet = allParamsSet && flawDistParamSet[i];
      if(! flawDistParamSet[i]){
        printf("Flaw distribution Parameter: %s not set\n", PTR_FlawDistParamNames[i]);
      }
    }
    if (!allParamsSet){
      fflush(stdout);
      throw std::runtime_error("All flaw distribution parameters must be set");
    }
  }

  void parseMatParamData(double matParam[PTR_NUM_MAT_PARAMS],
                         const std::string inputFileName){
    std::ifstream is(inputFileName.c_str());
    std::string line;
    std::string tag;
    double val;
    bool matParamSet[PTR_NUM_MAT_PARAMS] = {};
    while(std::getline(is,line)){
      std::istringstream iss(line);
      iss >> tag >> val;
      if(iss){
        for (int i=0; i<PTR_NUM_MAT_PARAMS; ++i){
          if(tag.compare(PTR_MatParamNames[i]) == 0){
            matParam[i] = val;
            matParamSet[i] = true;
            break;
          }
        }
      }
    }
    bool allParamsSet=true;
    for (int i=0; i<PTR_NUM_MAT_PARAMS; ++i){
      allParamsSet = allParamsSet && matParamSet[i];
      if(! matParamSet[i]){
        printf("material Parameter: %s not set\n", PTR_MatParamNames[i]);
      }
    }
    is.close();
    if (!allParamsSet){
      fflush(stdout);
      throw std::runtime_error("All material parameters must be set");
    }
  }

  inline double pareto_PDF(const double s, const double s_min, const double s_max,
                           const double a)
  {
    return a * pow(s_min, a) * pow(s, -a-1.0) / (1-pow(s_min/s_max,a));
  }

  inline double pareto_CDF(const double s, const double s_min, const double s_max,
                           const double a)
  {
    return (1-pow(s_min/s,a))/(1-pow(s_min/s_max,a));
  }

  inline double pareto_invCDF(const double u, const double s_min, const double s_max,
                              const double a)
  {
    double HL_ratio = s_max/s_min;
    double HL_ratio_a = pow(HL_ratio, a);
    return s_max/pow(HL_ratio_a-u*(HL_ratio_a-1),1.0/a);
  }

  inline double pareto_FirstMoment(const double s_min, const double s_max, const double a)
  {
    // For a bounded Pareto distribution, the pdf for a sub section of the distribution
    // has an identical form to the parent distribution. The only difference is that the
    // max and minimum values are changed.
    if( fabs(a-1.0)>1e-8 ){
      return pow(s_min,a)/(1-pow(s_min/s_max,a))
        * a/(a-1) * (pow(s_min,1-a) - pow(s_max,1-a));
    } else {
      // Special case of a=1:
      // return s_min*(log(s_max)-log(s_min))/(1-s_min/s_max);
      return s_min*log(s_max/s_min)/(1-s_min/s_max);
    }
  }

  inline double pareto_ThirdMoment(const double s_min, const double s_max, const double a)
  {
    if( fabs(a-3.0)>1e-8 ){
      return pow(s_min,a)/(1-pow(s_min/s_max,a))
        * a/(a-3) * (pow(s_min,3-a) - pow(s_max,3-a));
    } else {
      // Special case of a=3:
      return 3.0* pow(s_min,3) * log(s_max/s_min)/(1-pow(s_min/s_max,3));
    }
  }

  inline double pareto_SixthMoment(const double s_min, const double s_max, const double a)
  {
    if( fabs(a-6.0)>1e-8 ){
      return pow(s_min,a)/(1-pow(s_min/s_max,a))
        * a/(a-6.0) * (pow(s_min,6.0-a) - pow(s_max,6.0-a));
    } else {
      // Special case of a=6:
      return 6.0* pow(s_min,6) * log(s_max/s_min) /(1-pow(s_min/s_max,6));
    }
  }

  inline double pareto_variance(const double s_min, const double s_max, const double a)
  {
    double secondMoment;
    if( fabs(a-2.0)>1e-8 ){
      double firstMoment(pareto_FirstMoment(s_min,s_max,a));
      secondMoment =  pow(s_min,a)/(1-pow(s_min/s_max,a))
        * a/(a-2) * (pow(s_min,2-a) - pow(s_max,2-a));
      secondMoment -= (firstMoment*firstMoment);
    } else {
      // Special case of a=2:
      secondMoment = s_min*s_min*log(s_max/s_min)/(1-(s_min*s_min)/(s_max*s_max));
    }
    return secondMoment;
  }

  inline int poisson_sample(MTRand &randGen, const double lambda){
    double L(exp(-lambda)), p(1.0);
    int k(0);
    do{
      ++k;
      p *= randGen.rand();
    } while (p>L);
    return k-1;
  }

  inline double logNormal_PDF(const double s, const double sig, const double mu)
  {
    double temp(log(s)-mu);
    temp *= temp;
    temp /= -(2.0*sig*sig);
    temp = exp(temp);
    temp /= (s*sig*sqrt(2*M_PI));
    return temp;
  }


  double initalizeFlawDist( double flawSize[],
                            double flawNumber[],
                            const flawDistributionData flawDistData,
                            const double dx_ave, unsigned long seedArray[],
                            unsigned long nSeedValues
                            )
  {
    MTRand randGen(seedArray, nSeedValues);
    if( flawDistData.type == "normal"){
      // Switch this based on the desired internal flaw distribution:
      // For a normal distribution
      double sd = flawDistData.stdFlawSize;
      double smin(flawDistData.minFlawSize);
      smin = smin > 0 ? smin : 0.0;
      double smax(flawDistData.maxFlawSize);

      double ln(smax-smin); // Size of the sampling interval
      double binWidth(ln/flawDistData.numCrackFamilies);
      double pdfValue(0.0);
      double mean(flawDistData.meanFlawSize);
      double s(mean);
      double eta(flawDistData.flawDensity);
      // Generate a normal distribution of internal flaws:
      for (int i = 0; i < flawDistData.numCrackFamilies; i++){
        // Calculate the flaw size for the bin:
        s = smax - (i+0.5)*binWidth;
        if(s > 0.0){
          // Calculate the flaw denstiy in the bin:
          pdfValue = exp(-(s-mean)*(s-mean)*0.5/(sd*sd))/(sd*sqrt(2*M_PI));
          // Assign values:
          flawNumber[i] = eta*pdfValue*binWidth;
          flawSize[i] = s;
        } else {
          flawNumber[i] = 0.0;
          flawSize[i] = smin;
        }
      } // End loop through flaw families
    } else if(flawDistData.type == "pareto"){
      if(flawDistData.randomizeDist){
        switch (flawDistData.randomMethod)
          {
          case 0:
            {
              // All bins have the same probability (weight). The bin centers
              // are chosen by dividing up the Cumulinitive density function
              // and selecting one value from reach segment (for 5 bins it would be
              // 5 values from 0-.2, .2-.4, .4-.6, .6-.8, .8-1.0) and then using
              // the inverse maping technique. This will reproduce the parent distribution,
              // but it tends to produce a few pockets where there is a very high density of
              // large cracks and this is probably not physical.
              double smin = flawDistData.minFlawSize;
              double smax = flawDistData.maxFlawSize;
              double eta = flawDistData.flawDensity;
              double a = flawDistData.exponent;
              double s;

              double meanEta = eta/flawDistData.numCrackFamilies;
              double stdEta = sqrt(meanEta/(dx_ave*dx_ave*dx_ave));
            
              // Generate a Pareto distribution of internal flaws:
              for (int i = 0; i < flawDistData.numCrackFamilies; i++){
                // Calculate the flaw size for the bin:
                double fmin = (flawDistData.numCrackFamilies - i - 1)
                  /flawDistData.numCrackFamilies;
                double fmax = (flawDistData.numCrackFamilies - i)
                  /flawDistData.numCrackFamilies;
                double U = fmin+(fmax-fmin) * randGen.rand53();
                s = pow(-((U-1)*pow(smax,a)-(U*pow(smin,a))) / (pow(smin*smax, a)), -1.0/a);

                double x = randGen.rand53();
                double y = randGen.rand53();
                double Z = sqrt(-2*log(x))*cos(2*M_PI*y);

                // Transform standard normal Z to the distribution that I need:
                double flawNum = Z*stdEta+meanEta; // Select from a normal dist centered at meanEta
                flawNum = flawNum<0 ? 0 : flawNum; // Do not allow negitive flaw number densties
            
                // Assign values:
                flawNumber[i] = flawNum;
                flawSize[i] = s;
              } // End loop through flaw families
              break;
            }
          case 1:
            {
              // Use bins that all have the same width in terms of s, but choose
              // s as a uniformly distributed value within the width of the bin.
              // The height of the bin still comes from the value that would be given
              // if s were located at the midpoint of the bin. Flaw densities are not
              // treated as a stoichastic quantity.
              double smin = flawDistData.minFlawSize;
              double smax = flawDistData.maxFlawSize;
              double ln   = smax-smin;
              double binWidth = ln/flawDistData.numCrackFamilies;
              double pdfValue = 0.0;
              double eta = flawDistData.flawDensity;
              double a = flawDistData.exponent;
              double s_mid;
              double s;
                                        
              // Generate a Pareto distribution of internal flaws:
              for (int i = 0; i < flawDistData.numCrackFamilies; i++){
                // Calculate the flaw size for the bin:
                s_mid = smax -(i + 0.5)*binWidth;
                s = smax - (i + randGen.rand53())*binWidth;
                // Calculate the flaw denstiy in the bin:
                pdfValue = a * pow(smin, a) * pow(s_mid, -a-1.0) / (1-pow(smin/smax,a));
            
                // Assign values:
                flawNumber[i] = eta*pdfValue*binWidth;
                flawSize[i] = s;
              } // End loop through flaw families
              
              break;
            }
          case 2:
            {
              // Use bins that all have the same width in terms of s, but choose
              // s as a uniformly distributed value within the width of the bin.
              // The height of the bin is sampled using a normal approximation
              // to a poisson distribution
              double smin = flawDistData.minFlawSize;
              double smax = flawDistData.maxFlawSize;
              double ln   = smax-smin;
              double binWidth = ln/flawDistData.numCrackFamilies;
              double pdfValue = 0.0;
              double eta = flawDistData.flawDensity;
              double a = flawDistData.exponent;
              double /*s_mid,*/ meanEta, stdEta;
              double s;

              // Generate a Pareto distribution of internal flaws:
              for (int i = 0; i < flawDistData.numCrackFamilies; i++){
                // Calculate the flaw size for the bin:
                //s_mid = smax - (i+0.5)*binWidth;
                s = smax - (i+randGen.rand53())*binWidth;
                // Calculate the flaw denstiy in the bin:
                pdfValue = a * pow(smin, a) * pow(s, -a-1.0) / (1-pow(smin/smax,a));
                meanEta = eta*pdfValue*binWidth;
                stdEta = sqrt(meanEta/(dx_ave*dx_ave*dx_ave));
                
                double x = randGen.rand53();
                double y = randGen.rand53();
                double Z = sqrt(-2*log(x))*cos(2*M_PI*y);

                // Transform standard normal Z to the distribution that I need:
                double flawNum = Z*stdEta+meanEta; // Select from a normal dist centered at meanEta
                flawNum = flawNum<0 ? 0 : flawNum; // Do not allow negitive flaw number densties

                // Assign values:
                flawNumber[i] = flawNum;
                flawSize[i] = s;
              } // End loop through flaw families
              
              break;
            }
          case 3:
            // Assign bin sizes so that each successive bin has 2x as many flaws.
            // The flaw size within the bin is selected randomly using inverse
            // sampeling. The flaw densities within the bin are deterministic.
            {
              double smin = flawDistData.minFlawSize;
              double smax = flawDistData.maxFlawSize;
              double a = flawDistData.exponent;
                
              int numFlaws = 1;
              double u_cur, u_old, delU_cur, delU_old;
              double invVol = 1.0/(dx_ave*dx_ave*dx_ave);
              double invVol_Eta = invVol/flawDistData.flawDensity;

              delU_cur = invVol_Eta;
              u_cur = 1-0.5*delU_cur;

              flawNumber[0] = invVol;
              flawSize[0] =
                pareto_invCDF( u_cur + ( randGen.rand53()- 0.5 ) * delU_cur,
                               smin, smax, a);
               
              for(int i = 1; i<flawDistData.numCrackFamilies; ++i){
                // Copy current to old:
                delU_old = delU_cur;
                u_old = u_cur;

                numFlaws *= 2;
                delU_cur = numFlaws*invVol_Eta;
                u_cur = u_old - 0.5*(delU_cur + delU_old);

                flawNumber[i] = numFlaws*invVol;
                flawSize[i] =
                  pareto_invCDF( u_cur + ( randGen.rand53()- 0.5 ) * delU_cur,
                                 smin, smax, a);
              }
              break;
            }
          case 4:
            // Assign bin sizes so that each successive bin has 2x as many flaws.
            // The flaw size within the bin is selected randomly using inverse
            // sampeling. The flaw densities within the bin are deterministic.
            {
              double smin = flawDistData.minFlawSize;
              double smax = flawDistData.maxFlawSize;
              double a = flawDistData.exponent;
                                
              int numFlaws = 1;
              double u_cur, delU_cur, s_l, s_h;
              double invVol = 1.0/(dx_ave*dx_ave*dx_ave);
              double invVol_Eta = invVol/flawDistData.flawDensity;

              delU_cur = invVol_Eta;
              u_cur = 1-delU_cur; // CDF value at the minimum flaw size for the bin.
              s_h = smax;    // Upper bound on flaw size for the bin.
              s_l = pareto_invCDF( u_cur, smin, smax, a); // Lower bound on flaw size for the bin
                
              flawNumber[0] = invVol;
              // Rescale the distribution to achieve more accurate sampeling from within
              // the bin.
              flawSize[0] = pareto_invCDF( randGen.rand53(), s_l, s_h, a);
                
              for(int i = 1; i<flawDistData.numCrackFamilies; ++i){
                // Copy current to old:
                s_h = s_l;
                numFlaws *= 2;
                delU_cur = numFlaws*invVol_Eta;
                u_cur -= delU_cur;
                s_l = pareto_invCDF( u_cur, smin, smax, a);
                flawNumber[i] = numFlaws*invVol;
                flawSize[i] = pareto_invCDF( randGen.rand53(), s_l, s_h, a);
              }
              break;
            }
          case 5:
            {
              // Use uniform bin sizes. For each bin calculate the mean number of flaws in the
              // bin. If that number is greater than poissonThreshold then use a
              // Gaussian approximation to a Poisson distribution for the number
              // of flaws in the bin and use the first moment of the Pareto
              // distribution within the bin to compute the representitive flaw size,
              // otherwise compute the number of flaws in the bin from a Poisson
              // distribution and then explicitly simulate that number of flaws
              // and compute the mean.
              double s_min = flawDistData.minFlawSize;
              double s_max = flawDistData.maxFlawSize;
              double ln   = s_max-s_min;
              double binWidth = ln/flawDistData.numCrackFamilies;
              double eta = flawDistData.flawDensity;
              double a = flawDistData.exponent;
              double s, s_l(s_max), s_h(s_max), meanBinFlaws, binOmega;
              double poissonThreshold = 20;
                                        
              // Generate a Pareto distribution of internal flaws:
              for (int i = 0; i < flawDistData.numCrackFamilies; i++){
                s_h = s_l;
                s_l = s_h-binWidth;
                meanBinFlaws = eta*(dx_ave*dx_ave*dx_ave)*
                  (pareto_CDF(s_h,s_min,s_max,a)-pareto_CDF(s_l,s_min,s_max,a));
                if (meanBinFlaws >= poissonThreshold){
                  // Make s a deterministic value. This assumption could be relaxed,
                  // and s could be drawn from an approapreate normal distribution,
                  // but this is ok for now. These are the smallest flaws and most
                  // numerious flaws in the system so it is ok if they have a
                  // deterministic size.
                  s = pareto_FirstMoment(s_l,s_h,a);
                  binOmega = floor(randGen.randNorm(meanBinFlaws,sqrt(meanBinFlaws))+0.5)/
                    (dx_ave*dx_ave*dx_ave);
                } else {
                  // Explicitly sample a Poisson distribution then explicitly calculate
                  // the mean of the number of flaws in the bin
                  double binFlaws = poisson_sample(randGen,meanBinFlaws);
                  if(binFlaws>0){
                    double sum_s = 0;
                    for(unsigned int j=0;j< binFlaws; ++j){
                      sum_s += pareto_invCDF(randGen.rand53(),s_l,s_h,a);
                    }
                    s = sum_s/binFlaws;
                    binOmega = binFlaws/(dx_ave*dx_ave*dx_ave);
                  } else {
                    s = 0.5*(s_l + s_h);
                    binOmega = 0.0;
                  }
                }
                    
                // Assign values:
                flawNumber[i] = binOmega;
                flawSize[i] = s;
              } // End loop through flaw families

              break;
            }
          case 6:
            {
              // This is identical to case 5 except the representitive
              // flaw size, when the Gaussian approximation to a Poisson
              // distribution is used, is a random variable. The argument
              // for this is the central limit. The sample mean for a collection
              // of n iid random variables is normally distributed with a mean
              // equal to the population mean and variance equal to the population
              // variance divided by the number of samples.
              double s_min = flawDistData.minFlawSize;
              double s_max = flawDistData.maxFlawSize;
              double ln   = s_max-s_min;
              double binWidth = ln/flawDistData.numCrackFamilies;
              double eta = flawDistData.flawDensity;
              double a = flawDistData.exponent;
              double s, s_l(s_max), s_h(s_max), meanBinFlaws, binOmega, binFlaws;
              double poissonThreshold = 20;
              double del_xi, xi_l;
              bool do_BinBias = fabs(flawDistData.binBias-1)>1e-8;
              bool useRationalBias = flawDistData.binBias<0;
              double biasExp = flawDistData.binBias;
              if (useRationalBias){
                xi_l = 1.0;
                biasExp = -flawDistData.binBias;
                del_xi = (pow(s_max/s_min, 1/biasExp)-1.0)
                  /flawDistData.numCrackFamilies;
              } else {
                xi_l = 1.0;
                del_xi = 1.0/flawDistData.numCrackFamilies;
              }
              // Generate a Pareto distribution of internal flaws:
              for (int i = 0; i < flawDistData.numCrackFamilies; i++){
                s_h = s_l;
                if (useRationalBias){
                  xi_l += del_xi;
                  s_l = s_max/pow(xi_l,biasExp);
                } else if(do_BinBias){
                  xi_l -= del_xi;
                  if (xi_l < 0) xi_l=0.0;
                  s_l = pow(xi_l,biasExp)*ln + s_min;
                } else {
                  s_l = s_h-binWidth;
                }
                meanBinFlaws = eta*(dx_ave*dx_ave*dx_ave)*
                  (pareto_CDF(s_h,s_min,s_max,a)-pareto_CDF(s_l,s_min,s_max,a));
                if (meanBinFlaws >= poissonThreshold){
                  binFlaws = floor(randGen.randNorm(meanBinFlaws,sqrt(meanBinFlaws))+0.5);
                } else {
                  // Explicitly sample a Poisson distribution then explicitly calculate
                  // the mean of the number of flaws in the bin
                  binFlaws = poisson_sample(randGen,meanBinFlaws);
                }
                binOmega = binFlaws/(dx_ave*dx_ave*dx_ave);
                if (meanBinFlaws >= poissonThreshold){
                  double s_mean = pareto_FirstMoment(s_l,s_h,a);
                  double s_std = sqrt(pareto_variance(s_l,s_h,a)/binFlaws);
                  // Make sure that the simulated s lies within the bin, if not choose
                  // another sample
                  int j = 0;
                  do{
                    s = randGen.randNorm(s_mean,s_std);
                    ++j;
                  } while ( !(s<=s_h && s>=s_l) && j<10 );
                  if (j>=10){
                    printf( "TongeRamesh::InitalizeCMData() bin method 6. Tries for random s exceeded 10 accepting mean for value of s\n");
                    fflush(stdout);
                    s = s_mean;
                  }
                } else if(binFlaws>0){
                  double sum_s = 0;
                  for(unsigned int j=0;j< binFlaws; ++j){
                    sum_s += pareto_invCDF(randGen.rand53(),s_l,s_h,a);
                  }
                  s = sum_s/binFlaws;
                } else {
                  s = 0.5*(s_l + s_h);
                  binOmega = 0.0;
                }
                // Assign values:
                flawNumber[i] = binOmega;
                flawSize[i] = s;
              } // End loop through flaw families

              break;
            }
          case 7:
            {
              // This is identical to case 6 except the value of s_k is
              // the cube root of the mean value of the flaw sizes cubed.
              double s_min = flawDistData.minFlawSize;
              double s_max = flawDistData.maxFlawSize;
              double ln   = s_max-s_min;
              double binWidth = ln/flawDistData.numCrackFamilies;
              double eta = flawDistData.flawDensity;
              double a = flawDistData.exponent;
              double s, s_l(s_max), s_h(s_max), meanBinFlaws, binOmega, binFlaws;
              double poissonThreshold = 20;
              double del_xi, xi_l;
              bool do_BinBias = fabs(flawDistData.binBias-1)>1e-8;
              bool useRationalBias = flawDistData.binBias<0;
              double biasExp = flawDistData.binBias;
              if (useRationalBias){
                xi_l = 1.0;
                biasExp = -flawDistData.binBias;
                del_xi = (pow(s_max/s_min, 1/biasExp)-1.0)
                  /flawDistData.numCrackFamilies;
              } else {
                xi_l = 1.0;
                del_xi = 1.0/flawDistData.numCrackFamilies;
              }
                                        
              // Generate a Pareto distribution of internal flaws:
              for (int i = 0; i < flawDistData.numCrackFamilies; i++){
                s_h = s_l;
                if (useRationalBias){
                  xi_l += del_xi;
                  s_l = s_max/pow(xi_l,biasExp);
                } else if(do_BinBias){
                  xi_l -= del_xi;
                  if (xi_l < 0) xi_l=0.0;
                  s_l = pow(xi_l,biasExp)*ln + s_min;
                } else {
                  s_l = s_h-binWidth;
                }

                meanBinFlaws = eta*(dx_ave*dx_ave*dx_ave)*
                  (pareto_CDF(s_h,s_min,s_max,a)-pareto_CDF(s_l,s_min,s_max,a));
                if (meanBinFlaws >= poissonThreshold){
                  binFlaws = floor(randGen.randNorm(meanBinFlaws,sqrt(meanBinFlaws))+0.5);
                } else {
                  // Explicitly sample a Poisson distribution then explicitly calculate
                  // the mean of the number of flaws in the bin
                  binFlaws = poisson_sample(randGen,meanBinFlaws);
                }
                binOmega = binFlaws/(dx_ave*dx_ave*dx_ave);

                if (binFlaws >= poissonThreshold){
                  double s3_mean = pareto_ThirdMoment(s_l,s_h,a);
                  double s3_std = sqrt(pareto_SixthMoment(s_l,s_h,a)/binFlaws);
                  double s3(s3_mean);
                  // Make sure that the simulated s lies within the bin, if not choose
                  // another sample
                  int j = 0;
                  do{
                    s3 = randGen.randNorm(s3_mean,s3_std);
                    ++j;
                  } while ( !( s3<=(s_h*s_h*s_h) && s3>=(s_l*s_l*s_l) ) && j<10 );
                  if (j>=10){
                    printf( "TongeRamesh::InitalizeCMData() bin method 7. Tries for random s3 exceeded 10 accepting mean for value of s3\n");
                    fflush(stdout);
                    s3 = s3_mean;
                  }
                  s = pow(s3, 1.0/3.0);
                } else if(binFlaws>0){
                  double sum_s = 0;
                  for(unsigned int j=0;j< binFlaws; ++j){
                    double ai(pareto_invCDF(randGen.rand53(),s_l,s_h,a));
                    sum_s += ai*ai*ai;
                  }
                  s = pow(sum_s/binFlaws, 1.0/3.0);

                } else {
                  s = 0.5*(s_l + s_h);
                  binOmega = 0.0;
                }

                // Assign values:
                flawNumber[i] = binOmega;
                flawSize[i] = s;
              } // End loop through flaw families

              break;
            }
          default:
            std::ostringstream desc;
            desc << "Unknown bin selection method"
                 << "File: " << __FILE__ << " Line: " << __LINE__
                 << std::endl;
            throw std::runtime_error(desc.str());
          }
      } else {
        double smin = flawDistData.minFlawSize;
        double smax = flawDistData.maxFlawSize;
        if (smax <=0 ) {
          std::ostringstream desc;
          desc << "Computed an illegal value for smax: " << smax << "\n"
               << "The initial value was: "<< flawDistData.maxFlawSize
               << "File: " << __FILE__ << " Line: " << __LINE__
               << std::endl;
          throw std::runtime_error(desc.str());
        }
        smin = smin<0 ? 1e-8 * smax : smin;
        double ln   = smax-smin;
        double binWidth = ln/flawDistData.numCrackFamilies;
        double pdfValue = 0.0;
        double eta = flawDistData.flawDensity;
        double a = flawDistData.exponent;
        double s;
        // Generate a Pareto distribution of internal flaws:
        for (int i = 0; i < flawDistData.numCrackFamilies; i++){
          // Calculate the flaw size for the bin:
          s = smax - (i+0.5)*binWidth;
          // Calculate the flaw denstiy in the bin:
          pdfValue = a * pow(smin, a) * pow(s, -a-1.0) / (1-pow(smin/smax,a));
          // Assign values:
          flawNumber[i] = eta*pdfValue*binWidth;
          flawSize[i] = s;
        } // End loop through flaw families
      }
    } else if(flawDistData.type == "delta"){
      double s = flawDistData.meanFlawSize;
      double N = (flawDistData.flawDensity) / flawDistData.numCrackFamilies;
      for (int i = 0; i < flawDistData.numCrackFamilies; i++){
        flawNumber[i] = N;
        flawSize[i] = s;
      } // End loop through flaw families
    }
    double damage = 0;
    for (int i = 0; i < flawDistData.numCrackFamilies; i++){
      double damageInc = flawSize[i];
      damageInc *= flawSize[i];
      damageInc *= flawSize[i];
      damageInc *= flawNumber[i];
      damage += damageInc;
    }
    return damage;
  }

  flawDistributionData unpackFlawDistData( const double flawDistArray[PTR_NUM_FLAW_DIST_PARAM])
  {
    flawDistributionData flawDistData;
    flawDistData.numCrackFamilies = (int)std::floor(flawDistArray[0]+0.5);
	flawDistData.meanFlawSize     = flawDistArray[1];
    flawDistData.flawDensity      = flawDistArray[2];
    flawDistData.stdFlawSize      = flawDistArray[3];
    if(fabs(flawDistArray[4]) < 0.1){
      flawDistData.type = "delta";
    } else if(fabs(flawDistArray[4]) < 1.1){
      flawDistData.type = "normal";
    } else if(fabs(flawDistArray[4]) < 2.1){
      flawDistData.type = "pareto";
    } else {
      flawDistData.type = "unknown";
    }
    flawDistData.minFlawSize = flawDistArray[5];
	flawDistData.maxFlawSize = flawDistArray[6];
	flawDistData.exponent    = flawDistArray[7];
	flawDistData.randomizeDist = std::fabs(flawDistArray[8])>0.1;
	flawDistData.randomSeed    = (int)std::floor(flawDistArray[9]+0.5);
    flawDistData.randomMethod  = (int)std::floor(flawDistArray[10]+0.5);
	flawDistData.binBias       = flawDistArray[11];
	flawDistData.useEtaField = false;
    flawDistData.etaFilename = "N/A";
	flawDistData.useSizeField = false;
	flawDistData.sizeFilename = "N/A";
    return flawDistData;
  }
  
  void unpackMatParams(const double matParamArray[PTR_NUM_MAT_PARAMS],
                       Flags *flags,
                       ModelData *initialData,
                       flawDistributionData *flawDistData,
                       BrittleDamageData *brittle_damage,
                       granularPlasticityData *gpData,
                       ArtificialViscosity *artificialViscosity,
                       CMData *eosData
                       ){
    // Flags:
	flags->useDamage                  = std::fabs(matParamArray[0])>0.1;
	flags->usePlasticity              = std::fabs(matParamArray[1])>0.1;
	flags->useGranularPlasticity      = std::fabs(matParamArray[2])>0.1;
	flags->useOldStress               = std::fabs(matParamArray[3])>0.1;
	flags->artificialViscosity        = std::fabs(matParamArray[4])>0.1;
	flags->artificialViscosityHeating = std::fabs(matParamArray[5])>0.1;
	// Unused flags:
    flags->implicit        = false;
	flags->with_color      = false;
	flags->doErosion       = false;
	flags->allowNoTension  = false;
	flags->allowNoShear    = false;
    flags->setStressToZero = false;

    // ModelData:
    initialData->Bulk        = matParamArray[6];
	initialData->tauDev      = matParamArray[7];
    initialData->rho_orig    = matParamArray[8];
    initialData->FlowStress  = matParamArray[9];
	initialData->K           = matParamArray[10];
	initialData->Alpha       = matParamArray[11];
	initialData->timeConstant= matParamArray[12];

    // flawDistributionData:
    flawDistData->numCrackFamilies = (int)std::floor(matParamArray[13]+0.5);
	flawDistData->meanFlawSize     = matParamArray[14];
    flawDistData->flawDensity      = matParamArray[15];
    flawDistData->stdFlawSize      = matParamArray[16];
    if(fabs(matParamArray[17]) < 0.1){
      flawDistData->type = "delta";
    } else if(fabs(matParamArray[17]) < 1.1){
      flawDistData->type = "normal";
    } else if(fabs(matParamArray[17]) < 2.1){
      flawDistData->type = "pareto";
    } else {
      flawDistData->type = "unknown";
    }
    flawDistData->minFlawSize = matParamArray[18];
	flawDistData->maxFlawSize = matParamArray[19];
	flawDistData->exponent    = matParamArray[20];
	flawDistData->randomizeDist = std::fabs(matParamArray[21])>0.1;
	flawDistData->randomSeed    = (int)std::floor(matParamArray[22]+0.5);
    flawDistData->randomMethod  = (int)std::floor(matParamArray[23]+0.5);
	flawDistData->binBias       = matParamArray[24];
	flawDistData->useEtaField = false;
    flawDistData->etaFilename = "N/A";
	flawDistData->useSizeField = false;
	flawDistData->sizeFilename = "N/A";
    // Brittle Damage data
    brittle_damage->printDamage = false;
	brittle_damage->KIc         = matParamArray[25];
	brittle_damage->mu          = matParamArray[26];
	brittle_damage->phi         = matParamArray[27];
	brittle_damage->cgamma      = matParamArray[28];
	brittle_damage->alpha       = matParamArray[29];
	brittle_damage->criticalDamage = matParamArray[30];
	brittle_damage->maxDamage      = matParamArray[31];
    brittle_damage->usePlaneStrain = std::fabs(matParamArray[32])>0.1;
	brittle_damage->maxDamageInc   = matParamArray[33];
    brittle_damage->useDamageTimeStep = std::fabs(matParamArray[34])>0.1;
    brittle_damage->useOldStress      = flags->useOldStress;
	brittle_damage->dt_increaseFactor = matParamArray[35];
    brittle_damage->incInitialDamage  = std::fabs(matParamArray[36])>0.1;
    brittle_damage->doFlawInteraction = std::fabs(matParamArray[37])>0.1;
    // granularPlasticityData
    gpData->timeConstant = matParamArray[38];
	gpData->JGP_loc      = matParamArray[39];
	gpData->A            = matParamArray[40];
	gpData->B            = matParamArray[41];
    gpData->yeildSurfaceType = std::fabs(matParamArray[42] + 0.5);
	gpData->Pc           = matParamArray[43];
	gpData->alpha_e      = matParamArray[44];
	gpData->Pe           = matParamArray[45];
    gpData->GPModelType  = std::fabs(matParamArray[46]-1.0)<0.1 ? PTR::SingleSurface : PTR::TwoSurface ;
    // new and imporved model
    // It would be better to unpack these using:
    //     GFMS::unpackMaterialParameters and GFMS::unpackSolutionParameters
    gpData->GFMSsolParams.absToll   = matParamArray[47];
    gpData->GFMSsolParams.relToll   = matParamArray[48];
    gpData->GFMSsolParams.maxIter   = (int)std::floor(matParamArray[49]+0.5);
    gpData->GFMSsolParams.maxLevels = (int)std::floor(matParamArray[50]+0.5);
    gpData->GFMSmatParams.bulkMod   = initialData->Bulk;
    gpData->GFMSmatParams.shearMod  = initialData->tauDev;
    gpData->GFMSmatParams.m0        = matParamArray[51];
    gpData->GFMSmatParams.m1        = matParamArray[52];
    gpData->GFMSmatParams.m2        = matParamArray[53];
    gpData->GFMSmatParams.p0        = matParamArray[54];
    gpData->GFMSmatParams.p1        = matParamArray[55];
    gpData->GFMSmatParams.p2        = matParamArray[56];
    gpData->GFMSmatParams.p3        = matParamArray[57];
    gpData->GFMSmatParams.p4        = matParamArray[58];
    gpData->GFMSmatParams.a1        = matParamArray[59];
    gpData->GFMSmatParams.a2        = matParamArray[60];
    gpData->GFMSmatParams.a3        = matParamArray[61];
    gpData->GFMSmatParams.beta      = matParamArray[62];
    gpData->GFMSmatParams.psi       = matParamArray[63];
    if(fabs(matParamArray[64] - 1.0)<0.1){
      gpData->GFMSmatParams.J3Type    = GFMS::Gudehus;
    } else if(fabs(matParamArray[64] - 2.0) < 0.1){
      gpData->GFMSmatParams.J3Type    = GFMS::WilliamWarnke;
    } else {
      gpData->GFMSmatParams.J3Type    = GFMS::DruckerPrager;
    }
    gpData->GFMSmatParams.relaxationTime = gpData->timeConstant;
    // Artificial Viscosity
    artificialViscosity->coeff1  = matParamArray[65];
    artificialViscosity->coeff2  = matParamArray[66];
    // EOS Data
    eosData->C_0      = matParamArray[67];
    eosData->Gamma_0  = matParamArray[68];
    eosData->S_1      = matParamArray[69];
    eosData->S_2      = matParamArray[70];
    eosData->S_3      = matParamArray[71];
    eosData->C_v      = matParamArray[72];
    eosData->theta_0  = matParamArray[73];
    eosData->J_min    = matParamArray[74];
    eosData->N_points = (int)std::floor(matParamArray[75]+0.5);
  }
  // int checkmatparams(const double matParamArray[PTR_NUM_MAT_PARAMS]);

  double artificialBulkViscosity(	double Dkk, 
                                    double c_bulk, 
                                    double rho,
                                    double dx,
                                    const ArtificialViscosity av
                                    )
  {
    double q = 0.0;
    if (Dkk < 0.0) {
      double A1 = av.coeff1;
      double A2 = av.coeff2;
      assert(A1>0);
      assert(A2>0);
      //double c_bulk = sqrt(K/rho);
      q = (A1*fabs(c_bulk*Dkk*dx) + A2*(Dkk*Dkk*dx*dx))*rho;
    }
    return q;
  }

  // Compute the value and derivitive of the gp yeild function in the
  // Rendulic plane
  double calc_yeildFunc_g_gs_gp(	const granularPlasticityData gpData,
                                    const double sigma_s,
                                    const double sigma_p,
                                    double *gs, 
                                    double *gp){
    double A = gpData.A;
    double B = gpData.B;
    double g;

    switch (gpData.yeildSurfaceType){
    case 1:
      {
        g = sigma_s + A*(sigma_p-B);
        *gs = 1;
        *gp = A;
        break;
      }
    case 2:
      {
        *gs = 2.0*sigma_s;
        *gp = A;
        g = sigma_s*sigma_s+A*(sigma_p-B);
        break;
      }
    default:
      std::ostringstream desc;
      desc << "An unknown value for gpData.nonAssocMethod was "
           << "given, gpData.yeildSurfaceType=" << gpData.yeildSurfaceType
           << std::endl;
      throw std::runtime_error(desc.str());
      break;
    }

    return g;
  }

  void computeIncStress(    const BrittleDamageData brittle_damage,
                            const double eta3d,
                            const double matrixStress[2],
                            double incStress[3],
                            const double wingDamage,
                            const double parentDamage,
                            const PState state)
  {
	double Dw = wingDamage;

	// Grechka and Kachanov 2006 softening for two sets of flaws:
	double shear_0 = state.initialShearModulus;
	double bulk_0 =  state.initialBulkModulus;
	double nu_0 = (3*bulk_0-2*shear_0)/(6*bulk_0+2*shear_0);
	double E_0 = 9*bulk_0*shear_0/(3*bulk_0+shear_0);

	double Z_n = 16 * (1-nu_0*nu_0)/(3*E_0);
	double Z_r = Z_n/(1-0.5*nu_0);
	double Z_c = -Z_n/8.0;

	double mat_s_1111 = 1/E_0;
	double mat_s_2222 = 1/E_0;
	double mat_s_3333 = 1/E_0;
	double mat_s_1212 = (1+nu_0)/(2*E_0);
	double mat_s_1122 = -nu_0/E_0;
	double mat_s_1133 = -nu_0/E_0;
	double mat_s_2233 = -nu_0/E_0;

	// Add the compliance from the wing cracks:
	double prefactor = 0.25*Z_r*Dw;
	mat_s_2222 += prefactor*(4.0 - 2.0*nu_0);
	mat_s_1212 += prefactor*(1.0);

	// Add the coupling term
	mat_s_1122 += Z_c * Dw;
	mat_s_2233 += Z_c * Dw;

	// Elastic properties of the inclusion
	double kappa = (3-nu_0)/(1+nu_0); // Plane Stress

	// For Plane strain
	if(brittle_damage.usePlaneStrain) {
      // Invert the stiffness matrix (only the upper 9 elements need to be inverted
      // to calculate the full 3D damaged stiffness matrix
      Matrix3x3 s_upper(mat_s_1111, mat_s_1122, mat_s_1133,
                        mat_s_1122, mat_s_2222, mat_s_2233,
                        mat_s_1133, mat_s_2233, mat_s_3333);
      Matrix3x3 c_upper = s_upper.inverse();

      // The the planar compliance tensor is calculated from the reduced
      // stiffness matrix
      double denom_red = c_upper.get(0,0)*c_upper.get(1,1)-c_upper.get(0,1)*c_upper.get(0,1);
      mat_s_1111 = c_upper.get(1,1)/denom_red;
      mat_s_2222 = c_upper.get(0,0)/denom_red;
      mat_s_1122 = -c_upper.get(0,1)/denom_red;

      kappa = 3-4*nu_0;
	}


	//  ---------- Compute the stress in the Ellipse using Junwei's update ----------

	// Define the ellipse: These equations need to be corrected for three
	// dimensional flaw densities
	double eta2d = pow(eta3d, 2.0/3.0);
	double ea = 1.4416*sqrt(0.5/eta2d);
	double eb = 1/(M_PI*ea*eta2d);

	double c = 0.5*(ea+eb)*sqrt(M_PI*eta2d);
	double d = 0.5*(ea-eb)*sqrt(M_PI*eta2d);

	// Solve for the complex material properties g1 and g2: (equation 6.9.2 Green and Zerna)
	double aa = mat_s_2222;
	double bb = -2.0*(mat_s_1122 + 2.0*mat_s_1212);
	double cc = mat_s_1111;

	double descriminant = bb*bb - 4*aa*cc;


	if(descriminant>0) {
      // There are 2 real roots so g1 and g2 are real:
      double ap1 = (-bb+sqrt(descriminant))/(2.0*aa);
      double ap2 = (-bb-sqrt(descriminant))/(2.0*aa);

      double g1((sqrt(ap1)-1.0)/(sqrt(ap1)+1.0));
      double g2((sqrt(ap2)-1.0)/(sqrt(ap2)+1.0));
      if(std::abs(g1)>1.0 || std::abs(g2)>1.0) {
        throw std::runtime_error("Both abs(g1) and abs(g2) must be less than 1 (real branch)");
      }

      if(std::abs(g1)>1e-6 || std::abs(g2)>1e-6) {
        // Compute boundary displacements (delta and rho)
        // Equation A.10 (Bhasker) Eqn. 6.9.5 Green and Zerna
        double b1(mat_s_1122-ap1*mat_s_2222);
        double b2(mat_s_1122-ap2*mat_s_2222);
        // Equation A.9
        double d1( ((1.0+g1) * b2 - (1.0-g1)*b1) );
        double d2( ((1.0+g2) * b1 - (1.0-g2)*b2) );
        double r1( ((1.0+g1) * b2 + (1.0-g1)*b1) );
        double r2( ((1.0+g2) * b1 + (1.0-g2)*b2) );

        // See lab notebook for boundary condition equations (Andy Tonge: January 14 2013):
        FastMatrix bcMat(2, 2);
        // Solve for B and B', C=C'=0;
        // Equation 1:
        bcMat(0,0) = g1;
        bcMat(0,1) = g2;
        // Equation 2:
        bcMat(1,0) = g1*g1+1;
        bcMat(1,1) = g2*g2+1;

        double bcVec[2];
        bcVec[0] = (matrixStress[0]+matrixStress[1])/8.0;
        bcVec[1] = -(matrixStress[0]-matrixStress[1])/4.0;

        bcMat.destructiveSolve(bcVec); // The solution is placed in the vector
        double B = bcVec[0];
        double B1 = bcVec[1];

        // Equation A.17 (C=C/=0
        double H1(B *(c+g1*d));
        double H2(B1*(c+g2*d));

        // Setup and solve equation B.13 from Junwei's document:
        FastMatrix AmMat(4,4);  // 4 real equaitons

        // Real Part of equation 1:
        AmMat(0,0) = (kappa-1.0)*c;
        AmMat(0,1) = -d;
        AmMat(0,2) =-state.initialShearModulus*r1;
        AmMat(0,3) =-state.initialShearModulus*r2;

        // Equation 2:
        // Real Part:
        AmMat(1,0) = (kappa-1.0)*d;
        AmMat(1,1) = -c;
        AmMat(1,2) =-state.initialShearModulus*d1;
        AmMat(1,3) =-state.initialShearModulus*d2;

        // Equation 3:
        // Real Part:
        AmMat(2,0) = 2*c;
        AmMat(2,1) = d;
        AmMat(2,2) =-1.0;
        AmMat(2,3) =-1.0;

        // Equation 4:
        AmMat(3,0) = 2*d;
        AmMat(3,1) = c;
        AmMat(3,2) = -g1;
        AmMat(3,3) = -g2;

        double BmVec[4];
        BmVec[0] = state.initialShearModulus*(d1*H1+d2*H2);
        BmVec[1] = state.initialShearModulus*(r1*H1+r2*H2);
        BmVec[2] = g1*H1+g2*H2;
        BmVec[3] = H1+H2;

        // Solve the equations:
        AmMat.destructiveSolve(BmVec);
        double A1_re = BmVec[0];
        double A2_re = BmVec[1];

        incStress[0] = 4*A1_re-2*A2_re;
        incStress[1] = 4*A1_re+2*A2_re;
        incStress[2] = 0;
      } else {
        // Damage is too low to cause any change in the stress in the ellipse accept
        // the far field stress as the stress in the ellipse:
        incStress[0] = matrixStress[0];
        incStress[1] = matrixStress[1];
        incStress[2] = 0.0;
      }
	} else {
      // The roots are complex and should be complex conjegutes of
      // eachother.
      std::complex<double> ap1,ap2;
      ap1 = std::complex<double>(-bb/(2.0*aa), sqrt(-descriminant)/(2.0*aa));
      ap2 = std::complex<double>(-bb/(2.0*aa),-sqrt(-descriminant)/(2.0*aa));

      std::complex<double> g1((sqrt(ap1)-1.0)/(sqrt(ap1)+1.0));
      std::complex<double> g2((sqrt(ap2)-1.0)/(sqrt(ap2)+1.0));
      if(std::abs(g1)>1.0 || std::abs(g2)>1.0) {
        throw std::runtime_error("Both abs(g1) and abs(g2) must be less than 1 (complex branch)");
      }

      if(std::abs(g1)>1e-6) {
        // Compute boundary displacements (delta and rho)
        // Equation A.10
        std::complex<double> b1(mat_s_1122 - mat_s_2222*ap1);
        std::complex<double> b2(mat_s_1122 - mat_s_2222*ap2);
        // Equation A.9
        std::complex<double> d1( ((1.0+g1) * b2 - (1.0-g1)*b1) );
        std::complex<double> d2( ((1.0+g2) * b1 - (1.0-g2)*b2) );
        std::complex<double> r1( conj((1.0+g1)*b2 + (1.0-g1)*b1) );
        std::complex<double> r2( conj((1.0+g2)*b1 + (1.0-g2)*b2) );
        std::complex<double> compI(0,1.0);

        // Setup and solve Equation A.18 for the boundary conditions:
        FastMatrix bcMat(2,2);
        // We know that B=B' and C=-C' for complex g1=conj(g2)
        // Real part of first equation
        bcMat(0,0) = 2*real(g1);
        bcMat(0,1) =-2*imag(g1);
        // Real part of second equation
        bcMat(1,0) = real(g1)*real(g1) -imag(g1)*imag(g1) +1;
        bcMat(1,1) = -2.0*real(g1)*imag(g1);

        double bcVec[2];
        bcVec[0] = (matrixStress[0]+matrixStress[1])/8.0;
        bcVec[1] = -(matrixStress[0]-matrixStress[1])/8.0;

        bcMat.destructiveSolve(bcVec); // The solution is placed in the vector
        double B = bcVec[0];
        double C = bcVec[1];
        double B1 = B;
        double C1 = -C;

        // Equation A.17
        std::complex<double>H1(std::complex<double>(B,C)*(c+g1*d));
        std::complex<double>H2(std::complex<double>(B1,C1)*(c+g2*d));

        // Setup and solve equation B.13 from Junwei's document:
        FastMatrix AmMat(8,8);    // Solve 2 complex equations and 2 real equations
        // The equations are broken into their imagainary and real parts
        // Real Part of equation 1:
        AmMat(0,0) = (kappa-1.0)*c;
        AmMat(0,1) = 0;
        AmMat(0,2) = -d;
        AmMat(0,3) = 0;
        AmMat(0,4) =-state.initialShearModulus*real(conj(r1));
        AmMat(0,5) = state.initialShearModulus*imag(conj(r1));
        AmMat(0,6) =-state.initialShearModulus*real(conj(r2));
        AmMat(0,7) = state.initialShearModulus*imag(conj(r2));
        // // Imaginary part of equation 1:
        AmMat(1,0) = 0;
        AmMat(1,1) = (kappa-1.0)*c;
        AmMat(1,2) = 0;
        AmMat(1,3) = -d;
        AmMat(1,4) =-state.initialShearModulus*imag(conj(r1));
        AmMat(1,5) =-state.initialShearModulus*real(conj(r1));
        AmMat(1,6) =-state.initialShearModulus*imag(conj(r2));
        AmMat(1,7) =-state.initialShearModulus*real(conj(r2));

        // Equation 2:
        // Real Part:
        AmMat(2,0) = (kappa-1.0)*d;
        AmMat(2,1) = 0;
        AmMat(2,2) = -c;
        AmMat(2,3) = 0;
        AmMat(2,4) =-state.initialShearModulus*real(d1);
        AmMat(2,5) = state.initialShearModulus*imag(d1);
        AmMat(2,6) =-state.initialShearModulus*real(d2);
        AmMat(2,7) = state.initialShearModulus*imag(d2);
        // Imaginary Part:
        AmMat(3,0) = 0;
        AmMat(3,1) = (kappa-1.0)*d;
        AmMat(3,2) = 0;
        AmMat(3,3) = -c;
        AmMat(3,4) =-state.initialShearModulus*imag((d1));
        AmMat(3,5) =-state.initialShearModulus*real((d1));
        AmMat(3,6) =-state.initialShearModulus*imag((d2));
        AmMat(3,7) =-state.initialShearModulus*real((d2));

        // Equation 3:
        // Real Part:
        AmMat(4,0) = 2*c;
        AmMat(4,1) = 0;
        AmMat(4,2) = d;
        AmMat(4,3) = 0;
        AmMat(4,4) =-1.0;
        AmMat(4,5) = 0.0;
        AmMat(4,6) =-1.0;
        AmMat(4,7) = 0.0;
        // Imaginary Part:
        AmMat(5,0) = 0.0;
        AmMat(5,1) = 2*c;
        AmMat(5,2) = 0.0;
        AmMat(5,3) = d;
        AmMat(5,4) = 0.0;
        AmMat(5,5) =-1.0;
        AmMat(5,6) = 0.0;
        AmMat(5,7) =-1.0;

        // Equation 4:
        AmMat(6,0) = 2*d;
        AmMat(6,1) = 0;
        AmMat(6,2) = c;
        AmMat(6,3) = 0;
        AmMat(6,4) =-real(g1);
        AmMat(6,5) = imag(g1);
        AmMat(6,6) =-real(g2);
        AmMat(6,7) = imag(g2);
        // Imaginary Part:
        AmMat(7,0) = 0.0;
        AmMat(7,1) = 2*d;
        AmMat(7,2) = 0.0;
        AmMat(7,3) = c;
        AmMat(7,4) =-imag(g1);
        AmMat(7,5) =-real(g1);
        AmMat(7,6) =-imag(g2);
        AmMat(7,7) =-real(g2);

        double BmVec[8];
        BmVec[0] = state.initialShearModulus*real(conj(d1*H1+d2*H2));
        BmVec[1] = state.initialShearModulus*imag(conj(d1*H1+d2*H2));
        BmVec[2] = state.initialShearModulus*real(r1*conj(H1)+r2*conj(H2));
        BmVec[3] = state.initialShearModulus*imag(r1*conj(H1)+r2*conj(H2));
        BmVec[4] = real(conj(g1*H1+g2*H2));
        BmVec[5] = imag(conj(g1*H1+g2*H2));
        BmVec[6] = real(conj(H1+H2));
        BmVec[7] = imag(conj(H1+H2));

        // Solve the equations:
        AmMat.destructiveSolve(BmVec);
        double A1_re = BmVec[0];
        double A2_re = BmVec[2];
        double A2_im = BmVec[3];

        incStress[0] = 4*A1_re-2*A2_re;
        incStress[1] = 4*A1_re+2*A2_re;
        incStress[2] = -2*A2_im;

      } else {
        // Damage is too low to cause any change in the stress in the ellipse accept
        // the far field stress as the stress in the ellipse:
        incStress[0] = matrixStress[0];
        incStress[1] = matrixStress[1];
        incStress[2] = 0.0;
      }
	} // End of if(descriminant>0)
  }


  // Compute the damage growth, returns damage at the end of the
  // increment.
  double calculateDamageGrowth(	const BrittleDamageData brittle_damage,
                                const Matrix3x3 stress,
                                const double N[],
                                const double s[],
                                const double old_L[],
                                const double currentDamage,
                                double new_L[],
                                double new_Ldot[],
                                const double dt,
                                const int Localized,
                                const PState state,
                                const int nBins
                                )
  {
    double new_damage = 0.0;
    double bulk = state.bulkModulus;
    double shear = state.shearModulus;
    double phi = brittle_damage.phi;
    double KIc = brittle_damage.KIc;
    double cgamma = brittle_damage.cgamma;
    double mu = brittle_damage.mu;

    // Compute the Rayleigh wave speed and maximum crack velocity vm:
    bulk = state.initialBulkModulus;
    shear = state.initialShearModulus;
    double nu  = (3*bulk-2*shear) / (6*bulk+2*shear);
    double e   = (9*bulk*shear) / (3*bulk+shear);
    double rho = state.initialDensity;

    double Cr = (0.862+1.14*nu) * sqrt(e/(2*(1+nu)*rho)) / (1+nu);
    double vm = Cr/brittle_damage.alpha; // Maximum crack velocity
    if (vm<0.0) {
      std::ostringstream desc;
      desc << "A negative value was computed for the maximum crack velocity \n"
           << "Value for bulk modulus: " << bulk << "\n"
           << "Shear Modulus: " << shear << "\n"
           << " rho: " << rho << "\n"
           << "Youngs Modulus: " << e << "\n"
           << " nu: " << nu << "\n"
           << " vm: " << vm << std::endl;
      throw std::runtime_error(desc.str());
    }

    double matrixStress[2];

    // Use the two extreme principal stresses:
    double sig1, sig2, sig3;
    sig1 = 0;
    sig2 = 0;
    sig3 = 0;
    int numEigenValues;
    numEigenValues = stress.getEigenValues(&sig1, &sig2, &sig3);

    // Assign the maximum principal stress to matrixStress[1]
    // and the minimum principal stress to matrixStress[0]
    if(numEigenValues == 3) {
      matrixStress[1]=sig1;
      matrixStress[0]=sig3;
    } else if(numEigenValues == 2) {
      matrixStress[1]=sig1;
      matrixStress[0]=sig2;
    } else {
      if(fabs(stress.trace()/3.0 - sig1) < 1e-3*fabs(sig1)) {
        // The stress state is hydrostatic:
        matrixStress[1]=sig1;
        matrixStress[0]=sig1;
      } else {
        // The other two eigen values are 0
        if(sig1>0) {
          matrixStress[1]=sig1;
          matrixStress[0]=0;
        } else {
          matrixStress[1]=0;
          matrixStress[0]=sig1;
        }
      }
    }

    double incStress[3] = {0};

    if(brittle_damage.doFlawInteraction) {
      double eta3d = 0;
      for(int i=0; i<nBins; ++i){
        eta3d += N[i];
      }
      computeIncStress(brittle_damage, eta3d, matrixStress, incStress, currentDamage, 0, state); // Place all of the damage in the wing cracks
    } else {
      incStress[0]=matrixStress[0];
      incStress[1]=matrixStress[1];
      incStress[2]=0.0;
    }

    // Now using the stress on the inside of the elipse calculate the crack growth:
    double s11e = incStress[0];
    double s22e = incStress[1];
    double s12e = incStress[2];

    const double cosPhi = cos(phi);
    const double sinPhi = sin(phi);
    const double cos2Phi = cos(2.0*phi);
    const double sin2Phi = sin(2.0*phi);
    
    for(int i=0; i<nBins; ++i){
      if(N[i] >0 ) {
        if( (old_L[i]+s[i])*(old_L[i]+s[i])*(old_L[i]+s[i]) < 1.0/N[i] ) {
          // Calculate the wedging force:
          double F = -2.0 * (s[i]) * ( -mu*( s11e*cosPhi*cosPhi +
                                             s22e*sinPhi*sinPhi +
                                             s12e*sin2Phi
                                             )
                                       - (-0.5*(s11e-s22e)*sin2Phi+s12e*cos2Phi)
                                       );
          if(F <= 0.0) {
            F = 0.0;
          }
          // This assumes that the wing cracking mechanism is active:
          double K1 = F/(sqrt(M_PI*(old_L[i] + 0.27 * s[i]))) + s22e*sqrt(M_PI*(old_L[i]+sinPhi*s[i]));
          // Calculate the crack growth:
          if(K1 >= KIc) {
            new_Ldot[i] = vm * pow((K1-KIc)/(K1-KIc*0.5), cgamma);
          } else {
            new_Ldot[i] = 0;
          }
          // calculate the new crack length:
          new_L[i] = std::min(old_L[i] + dt*(new_Ldot[i]), (1.0/pow(N[i], 1.0/3.0) - s[i]));
        } else {
          // set new_L to the maximum crack length for that family
          new_L[i] = (1.0/pow(N[i], 1.0/3.0) - s[i]);
        }

        // Sum the damage value:
        double damageIncrement = N[i];
        if(brittle_damage.incInitialDamage) {
          damageIncrement *=  new_L[i] + s[i];
          damageIncrement *=  new_L[i] + s[i];
          damageIncrement *=  new_L[i] + s[i];
        } else {
          damageIncrement *=  new_L[i];
          damageIncrement *=  new_L[i];
          damageIncrement *=  new_L[i];
        }
        new_damage += damageIncrement;
      } // end if(N[i]>0)
    }   // end flaw family loop
    new_damage = new_damage > brittle_damage.maxDamage ? brittle_damage.maxDamage : new_damage;
    return new_damage;
  }

  inline double calcJGPFromPressure(const PortableMieGruneisenEOSTemperature *eos,
                                    PState *state,
                                    const double pDamage,
                                    const double pJGP,
                                    const double J,
                                    const double p_target
                                    ){
    const double rho_orig((*state).initialDensity);
    Matrix3x3 identity(true);
    unsigned int stepNum=0;
    double JGP(pJGP),JEL(J/JGP);
    double p_new = computePressure(eos, identity * cbrt(JEL), *state, pDamage);

    while(std::fabs(p_new-p_target) > 1e-10*state->bulkModulus && stepNum<100) {
      stepNum++;
      JGP += JGP * (p_new-p_target) / (JEL*(*state).bulkModulus);
      // JGP = JGP * exp( (p_new - p_target)/state->bulkModulus );
      JEL = J/JGP;
      state->density = rho_orig/JEL;
      p_new = computePressure(eos, identity * cbrt(JEL), *state, pDamage);
    } // end JGP correction loop
    if(stepNum > 99){
      std::stringstream msg;
      msg << "Solver in calcJGPFromPressure() reached max number of iterations\n";
      msg << "  File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
      msg << "debugging information:\n"
          << "p_new:\t" << std::setw(20) << std::setprecision(12) << p_new << "\n"
          << "p_target:\t" << std::setw(20) << std::setprecision(12) << p_target << "\n"
          << "J:\t" << std::setw(20) << std::setprecision(12) << J << "\n"
          << "state.bulkModulus:\t" << std::setw(20) << std::setprecision(12) << (*state).bulkModulus << "\n"
          << "JGP:\t" << std::setw(20) << std::setprecision(12) << JGP << "\n"
          << "JEL:\t" << std::setw(20) << std::setprecision(12) << JEL << "\n"
          << "state.density:\t" << std::setw(20) << std::setprecision(12) << (*state).density << "\n"
          << "rho_orig:\t" << std::setw(20) << std::setprecision(12) << rho_orig << "\n"
          << "damageValue:\t" << pDamage << "\n";
      throw std::runtime_error(msg.str());
    }
    return JGP;
  }

  void calcGranularFlow( const Flags flags, // input flags
                         const BrittleDamageData brittle_damage, // Damage flags
                         const granularPlasticityData gpData,
                         const PortableMieGruneisenEOSTemperature *eos, // passed through to computePressure
                         const double delT,
                         const double pDamage_new,
                         const double J,
                         const double pGPJ_old,
                         Matrix3x3 *bElBar_new, // Input and output (deviatoric elastic strain)
                         PState *state,
                         double *pGPJ,
                         double *pGP_strain,
                         double *pGP_energy,
                         double *pdTdt
                         ){
    const double invDelT = delT > 0 ? 1.0/delT : 0.0;
    double JGP(*pGPJ);
    double JEL(J/JGP);
    double rho_orig(state->initialDensity);
    const double pTemperature(state->temperature);
    Matrix3x3 identity(true);
    // if damage is being used only activate graunlar plasticity when the
    // damage level is equal to the critical damage level
    const double gp_TQparam(1.0);
    // if damage is being used only activate graunlar plasticity when the
    // damage level is equal to the critical damage level
    bool doGPcalc(true);
    if(flags.useDamage){
      doGPcalc = pDamage_new >= brittle_damage.criticalDamage-1e-8;
    }
    if(doGPcalc){
      Matrix3x3 bElBarTrial = *bElBar_new;
      double IEl            = onethird*bElBarTrial.trace();

      // Compute the trial stress:
      Matrix3x3 tauDevTrial = (bElBarTrial - identity*IEl)*state->shearModulus;
      double sTnorm         = tauDevTrial.norm();
      // hydrostatic stress:
      double p_trial;
      if( flags.useDamage) {
        p_trial = computePressure(eos, identity * cbrt(JEL), (*state), pDamage_new);
      } else {
        p_trial = computePressure(eos, identity*cbrt(JEL), (*state), 0);
      }

      // Return algorithm discussed by Rebecca Brannon in Appendix 3 of:
      // Geometric Insight into Return Mapping Algorithms

      const double gamma_s=1.0;
      const double gamma_p=1.0;
      const double sigma_tr_p = gamma_p*sqrt(3)*p_trial;
      const double sigma_tr_s = gamma_s*sTnorm;

      double gs,gp;
      double g_tr=calc_yeildFunc_g_gs_gp(gpData, sigma_tr_s, sigma_tr_p,
                                         &gs, &gp);
      // Check for plastic loading:
      if(g_tr>0.0) {
        // Compute the rate independent limit tau_bar

        double g_test=g_tr;
        double sigma_test_p=sigma_tr_p;
        double sigma_test_s=sigma_tr_s;
        Matrix3x3 S_hat=tauDevTrial;
        if(sTnorm>1e-8){
          S_hat /= sTnorm;
        } else {
          sigma_test_p = gpData.B/(sqrt(3.0)*gamma_p);
          sigma_test_s = 0.0;
          g_test       = 0.0;
          S_hat       *= 0.0;
        }
       
        int stepNum;

        for(stepNum=0; stepNum<100; ++stepNum) {
          double M_p,M_s;

          M_p = gamma_p*gp;
          M_s = gamma_s*gs;

          double eta = 2*state->shearModulus/(3*state->bulkModulus);
          double A_p=M_p;
          double A_s=eta*M_s;
          double scaleFactor =
            sqrt( (sigma_test_p*sigma_test_p +sigma_test_s*sigma_test_s)
                  /(A_p*A_p+A_s*A_s));
          A_p *= scaleFactor;
          A_s *= scaleFactor;

          double beta_next = -g_test/(gp*gamma_p*A_p+gs*gamma_s*A_s);
          sigma_test_p += beta_next*gamma_p*A_p;
          sigma_test_s += beta_next*gamma_s*A_s;
          if(fabs(beta_next)<1e-8) {
            break;
          }
          g_test=calc_yeildFunc_g_gs_gp(gpData, sigma_test_s, sigma_test_p,
                                        &gs, &gp);
        }
        if(stepNum == 99) {
          throw std::runtime_error("Solver in TongeRamesh GP calc reached max number of iterations");
        }

        // Now we can compute the rate independent values:
        double p_bar= sigma_test_s < 0 ? gpData.B/(sqrt(3.0)*gamma_p) : sigma_test_p/(sqrt(3.0)*gamma_p);
        Matrix3x3 tauBar = sigma_test_s < 0 ? Matrix3x3(0.0) : S_hat * sigma_test_s;

        // This solution is based on equation 5.8.3 in Simo's Computational plasticity.
        // It is for the problem where strainRate_vp = 1/visc*C^-1*(sigma-sigma_bar)
        // I need to compute an effective viscosity
        double p_target(p_bar);
        Matrix3x3 tauDev(tauBar);
        if(gpData.timeConstant>0.0) {
          double inv_timeConst = 1.0/gpData.timeConstant;
          tauDev=(tauDevTrial+ tauBar*inv_timeConst*delT)/(1+delT*inv_timeConst);

          p_target = (p_trial + delT*inv_timeConst*p_bar)/(1+delT*inv_timeConst);
        } 
        // Now that I have the updated stress calculate the updated history variables:

        // switch to the cauchy stress:
        Matrix3x3 devPlasticStrainRate = (tauDevTrial-tauDev)*invDelT/(2*state->shearModulus);
        double tr_d_p = (p_trial-p_target)*invDelT/(state->bulkModulus);
        JGP = pGPJ_old*exp(delT*tr_d_p);

        *pGPJ = JGP;
        JEL = J/JGP;
        state->density = rho_orig/JEL;

        JGP = calcJGPFromPressure(eos, state, pDamage_new, JGP, J, p_target);
        JEL = J/JGP;
        state->density = rho_orig/JEL;
        *pGPJ = JGP;
        tr_d_p = log(JGP/pGPJ_old)/delT;
        double p_new = computePressure(eos, identity * cbrt(JEL), *state, pDamage_new);
        double IEl_tr(IEl);
        // update IEl based on the ratio of effective strain energies:
        if(tauDevTrial.normSquared()>tauDev.normSquared()){
          IEl = (IEl_tr - 1) * (tauDev.normSquared()/(sTnorm*sTnorm)) + 1;         
        } 
        *bElBar_new     = tauDev/state->shearModulus + identity*IEl;
        *pGP_strain    += delT*devPlasticStrainRate.norm();

        // Compute the changes in energy:
        double del_Thermal = state->specificHeat * delT * rho_orig *
          eos->computeIsentropicTemperatureRate(	pTemperature, 
                                                rho_orig,
                                                state->density,
                                                (JGP - pGPJ_old)*invDelT/(JGP)
                                                );
        double trial_U(eos->computeStrainEnergy(rho_orig,rho_orig * pGPJ_old/J));
        double del_U = trial_U - eos->computeStrainEnergy(rho_orig,rho_orig * JGP/J)+del_Thermal ;
        del_U *= state->bulkModulus/state->initialBulkModulus;
            
        double del_W = 0.5*state->shearModulus * 3.0 *(IEl_tr - IEl); // Note IEl is 1/3* tr(be)
            
        if(del_U + del_W < 0 && del_U<0){
          for (int i = 0; i<110; i++){
            tr_d_p *= 0.9;
            JGP = pGPJ_old*exp(delT*tr_d_p);
            *pGPJ = JGP;
            JEL = J/JGP;
            state->density = rho_orig/JEL;

            p_new = flags.useDamage ?
              computePressure(eos, identity * cbrt(JEL), (*state), pDamage_new) :
              computePressure(eos, identity*cbrt(JEL), (*state), 0);
						
            del_Thermal = state->specificHeat * delT * rho_orig *
              eos->computeIsentropicTemperatureRate(	pTemperature, 
                                                    rho_orig,
                                                    state->density,
                                                    (JGP - pGPJ_old)*invDelT/(JGP)
                                                    );
            del_U = trial_U - eos->computeStrainEnergy(rho_orig,rho_orig * JGP/J)+del_Thermal ;
            del_U *= state->bulkModulus/state->initialBulkModulus;
            
            del_W = 0.5*state->shearModulus * 3.0 *(IEl_tr - IEl); // Note IEl is 1/3* tr(be)

            if(del_U + del_W >= 0){
              break;
            }
            if(i>100){
              // tr_d_p = 26e-6* the original tr_d_p, set it to 0.
              del_U=0;
              tr_d_p=0;
              JGP = pGPJ_old;
              *pGPJ = JGP;
              JEL = J/JGP;
              state->density = rho_orig/JEL;
              p_new = flags.useDamage ?
                computePressure(eos, identity * cbrt(JEL), (*state), pDamage_new) :
                computePressure(eos, identity*cbrt(JEL), (*state), 0);
              break;
            }
          } // End of  for(int i = 0; i<110; i++) 
        }   // End of  if(del_U + del_W < 0 && p_trial < 0)

        if(del_U + del_W < 0){
          if( (del_W < 0.0) & (tauDevTrial.normSquared()>tauDev.normSquared()) ) {
            *bElBar_new = identity;
            tauDev = Matrix3x3(0.0);
            IEl = 1.0;
          } else {
            std::ostringstream desc;
            desc << "Negative plastic work detected: del_U+del_W=\t" << del_U+del_W << std::endl;
            desc << "Trial State:\n"
                 << "p_trial:\t" << p_trial << "\n"
                 << "J:\t" << J << "\t pGPJ_old:\t" << pGPJ_old << "\n"
                 << "tauDevTrial.Norm():\t" << tauDevTrial.norm() <<"\t tauDevTrial:" << std::endl;
            desc << tauDevTrial;
            desc << "IEl_tr:\t" << IEl_tr << std::endl;

            desc << "Bulk modulus:\t" << state->bulkModulus << "\t"
                 << "Shear modulus:\t" << state->shearModulus << std::endl;
            if(flags.useDamage){
              desc << "Damage:\t" << pDamage_new<< std::endl;
            }
              
            desc << "Return State:\n"
                 << "p_new:\t" << p_new << "\t p_target:\t" << p_target << "\n"
                 << "J:\t" << J << "\t pGPJ:\t" << JGP << "\n"
                 << "tr_d_p:\t" << tr_d_p << "\n"
                 << "tauDev.Norm():\t" << tauDev.norm() << "\t tauDev:" << std::endl;
            desc << tauDev;
            desc << "IEl:\t" << IEl << std::endl;
            desc << "del_U:\t" << del_U << "\t del_W:\t" << del_W << std::endl;
            throw std::runtime_error(desc.str());
          }
        }
        *pGP_energy +=  gp_TQparam*(del_U + del_W);
        if(p_new < 0.0){        // only allow frictional heating under compressive states
          *pdTdt += gp_TQparam*(del_U + del_W)*invDelT/(rho_orig * state->specificHeat);
        }
      } else {
        *pGPJ = JGP;
        JEL = J/JGP;
        state->density = rho_orig/JEL;
      } // End if(sigma_hat > Sigma_c)
    } else { // End damage level test
      // Leave bElBar_new alone since there is no GP flow it does not
      // need to be updated.
      *pGPJ = JGP;
      JEL = J/JGP;
      state->density = rho_orig/JEL;
    }

    // p-\alpha pore compaction model: -----------------

    // Model parameters: (Note pressure P is + compression for
    // this calculation, this is opposite the standard used in
    // the equation of state calculation).
    double Pc = gpData.Pc;
    double JGP_e = gpData.alpha_e;
    double Pe = gpData.Pe;
    double Ps;
    Ps = -computePressure(eos, identity * cbrt(JEL), (*state), pDamage_new);

    double f_phi_tr;
    double   dPs_dJGP = (state->bulkModulus - Ps/JEL)*(JEL/JGP);
    double kappa = (Pc-Pe)/(2*Pe*(JGP_e - 1.0));
    if (Ps < J*Pe) {
      f_phi_tr = Ps/(J*Pc-J*Pe) - Pe/(Pc-Pe)*exp(-kappa*(JGP - JGP_e));
    } else if(Ps < J*Pc) {
      f_phi_tr = (JGP-1.0) - (JGP_e - 1.0)* pow((J*Pc - Ps)/(J*Pc - J*Pe), 2);
    } else {
      f_phi_tr = JGP - 1.0;
    }
    double abs_toll = 1e-8;
    if (f_phi_tr>abs_toll) {
      double del_JGP_eq(0), df_dJGP, JGP_k, Ps_k(Ps);

      for (int k = 0; fabs(f_phi_tr)>abs_toll && k<200; k++) {
        JGP_k = JGP+del_JGP_eq;
        Ps_k = Ps + dPs_dJGP*del_JGP_eq;
        if(Ps_k < J*Pe) {
          f_phi_tr = Ps_k/(J*Pc-J*Pe) - Pe/(Pc-Pe)*exp(-kappa*(JGP_k - JGP_e));
          df_dJGP = dPs_dJGP/(J*Pc-J*Pe) + kappa*Pe/(Pc-Pe)*exp(-kappa*(JGP_k - JGP_e));
        } else if(Ps_k < J*Pc) {
          f_phi_tr = (JGP_k-1.0) - (JGP_e - 1.0)* pow((J*Pc - Ps_k)/(J*Pc - J*Pe), 2);
          df_dJGP = 1.0 + 2.0 * (JGP_e - 1.0)*( (J*Pc - Ps_k)/(J*Pc - J*Pe) ) *
            dPs_dJGP/(J*(Pc-Pe));
        } else {
          f_phi_tr = (JGP_k-1.0);
          df_dJGP = 1;
          del_JGP_eq = 1-JGP;
          break;
        }
        del_JGP_eq -= f_phi_tr/df_dJGP;

        if(fabs(del_JGP_eq) < abs_toll || fabs(f_phi_tr/df_dJGP) < abs_toll) {
          break;
        }

        // Check to make sure I am not using too many iterations:
        if(k == 199) {
          throw std::runtime_error("Solver in TongeRamesh GP porosity calc reached max number of iterations");
        }
      }

      if(JGP + del_JGP_eq<1.0) {
        del_JGP_eq = 1.0 - JGP;
      }

      double del_JGP_pore = del_JGP_eq;
      if(gpData.timeConstant >0) {
        double inv_timeConst = 1.0/gpData.timeConstant;
        del_JGP_pore = (JGP + delT * inv_timeConst * (JGP + del_JGP_eq))
          /(1+delT*inv_timeConst) - JGP;
      }
      JGP += del_JGP_pore;
      *pGPJ = JGP;
      JEL = J/JGP;
      state->density = rho_orig/JEL;

      // Compute the energy dissipated in the pore compaction process:
      double sigma_m_old = -Ps;
      double sigma_m_new;
      sigma_m_new = computePressure(eos, identity * cbrt(JEL), *state, pDamage_new);
      double pore_energyRate = 0.5*(sigma_m_new+sigma_m_old)*del_JGP_pore*invDelT;
      *pdTdt += pore_energyRate/ (rho_orig * state->specificHeat);
      *pGP_energy += pore_energyRate*delT;
    } // End of porosity calculation
  }

  double computePressure(const PortableMieGruneisenEOSTemperature *eos,
                         const Matrix3x3 F,
                         const PState state,
                         const double currentDamage)
  {
	double J = F.determinant();

	double damageFactor = calculateBulkPrefactor(currentDamage, state, J);

	double P_hat = J*eos->computePressure(state);

	return P_hat*damageFactor;
  }

  double calculateBulkPrefactor(	const double currentDamage,
                                    const PState state,
                                    const double J
                                    )
  {
	double shear_0 = state.initialShearModulus;
	double bulk_0 =  state.initialBulkModulus;
	double nu_0 = (3*bulk_0-2*shear_0)/(6*bulk_0+2*shear_0);
	double E_0 = 9*bulk_0*shear_0/(3*bulk_0+shear_0);

	double Z_n = 16 * (1-nu_0*nu_0)/(3*E_0);
	double Z_c = -Z_n/8.0;

	double inv_prefactor = 1 + bulk_0 * (Z_n + 4*Z_c) * currentDamage;

    return 1/inv_prefactor;
  }

  // Compute the shear modulus from the current damage level:
  double calculateShearPrefactor(	const double currentDamage,
                                    const PState state
                                    )
  {
	double shear_0 = state.initialShearModulus;
	double bulk_0 =  state.initialBulkModulus;
	double nu_0 = (3*bulk_0-2*shear_0)/(6*bulk_0+2*shear_0);
	double E_0 = 9*bulk_0*shear_0/(3*bulk_0+shear_0);

	double Z_n = 16 * (1-nu_0*nu_0)/(3*E_0);
	double Z_r = Z_n/(1-0.5*nu_0);
	double Z_c = -Z_n/8.0;

	double inv_prefactor = 1 + shear_0 * 2.0/15.0 * (3*Z_r + 2*Z_n - 4*Z_c) * currentDamage;

	return 1/inv_prefactor;
  }

  double computeStableTimestep(	const ModelData initialData,
                                const Vector3 pVelocity,
                                const Vector3 dx,
                                const double pMass,
                                const double pVolume)
  {
	// This is only called for the initial timestep - all other timesteps
	// are computed as a side-effect of computeStressTensor
	double c_dil = 0.0;
	Vector3 WaveSpeed(1.e-12);

	double mu   = initialData.tauDev;
	double bulk = initialData.Bulk;


	if(pMass > 0) {
      c_dil = sqrt((bulk + 4.*mu/3.)*pVolume/pMass);
	} else {
      c_dil = 0.0;
	}
	WaveSpeed[0] = dx.x()/std::max(c_dil+fabs(pVelocity.x()),WaveSpeed.x());
	WaveSpeed[1] = dx.y()/std::max(c_dil+fabs(pVelocity.y()),WaveSpeed.y());
	WaveSpeed[2] = dx.z()/std::max(c_dil+fabs(pVelocity.z()),WaveSpeed.z());
	return WaveSpeed.min();
  }

  void ComputeStressTensorInnerLoop(
                                    // Data Structures
                                    const Flags flags,
                                    const ModelData initialData,
                                    const flawDistributionData flawDistData,
                                    const BrittleDamageData brittle_damage,
                                    const granularPlasticityData gpData,
                                    const ArtificialViscosity artificialViscosity,
                                    const PortableMieGruneisenEOSTemperature *eos,
                
                                    // Input Matrices
                                    const Matrix3x3 pDefGrad,
                                    const Matrix3x3 pDefGrad_new,
                                    const Matrix3x3 pVelGrad,

                                    // Output Matrix
                                    Matrix3x3 *pDeformRate,
                                    Matrix3x3 *bElBar,
                                    Matrix3x3 *pStress,
                                    Matrix3x3 *pStress_qs,

                                    // Input Vector3
                                    const Vector3 pVelocity,
                                    const Vector3 dx,

                                    // Output Vector3
                                    Vector3 *WaveSpeed,

                                    // Input double
                                    const double pGPJ_old,
                                    const double RoomTemperature,
                                    const double pTemperature,
                                    const double rho_orig,
                                    const double pVolume_new,
                                    const double pMass,
                                    const double SpecificHeat,
                                    const double pDamage,
                                    const double K,
                                    const double flow,
                                    const double delT,
                
                                    // Output double
                                    double *pGP_strain,
                                    double *pPlasticStrain,
                                    double *pPlasticEnergy,
                                    double *pDamage_new,
                                    double *pGPJ,
                                    double *pGP_energy,
                                    double *pEnergy_new,
                                    double *damage_dt,
                                    double *p_q,
                                    double *se,
                                    double *pdTdt,
                                    double *pepsV,
                                    double *pgam,
                                    double *pepsV_qs,
                                    double *pgam_qs,

                                    // Input int
                                    const int pLocalized,
                                    const long long pParticleID,
                
                                    // Output int
                                    long long *totalLocalizedParticle,
                                    int *pLocalized_new,
                
                                    // Input std::vector
                                    const std::vector<double> *pWingLength_array,
                                    const std::vector<double> *pFlawNumber_array,
                                    const std::vector<double> *pflawSize_array,

                                    // Output std::vector
                                    std::vector<double> *pWingLength_array_new
                                    )
  {
    const Matrix3x3 identity(true);
    const Matrix3x3 pDefGradInc = identity + pVelGrad*delT + (pVelGrad*pVelGrad)*(0.5*delT*delT);
    *pDeformRate = (pVelGrad + pVelGrad.transpose())*0.5;
    const double Jinc = pDefGradInc.determinant();
    const Matrix3x3 defGrad = pDefGrad_new;
    const double J = pDefGrad_new.determinant();
    const double J_old = pDefGrad.determinant();
    const Matrix3x3 pStress_old = *pStress;
    const std::vector<double>::size_type nBins = pWingLength_array->size();

    bool inputValid(true);
    const double defGradMax(1e8);
    const double velGradMax(1e12);
    const double stressMax(1e24);
    const double paramMax(1e37);
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
        if(!(std::fabs(pDefGrad.get(i,j))     < defGradMax)) inputValid=false;
        if(!(std::fabs(pDefGrad_new.get(i,j)) < defGradMax)) inputValid=false;
        if(!(std::fabs(pVelGrad.get(0,0))     < velGradMax)) inputValid=false;
        if(!(std::fabs(bElBar->get(0,0))      < defGradMax)) inputValid=false;
        if(!(std::fabs(pStress_qs->get(0,0))  < stressMax))  inputValid=false;
        if(!(std::fabs(pStress->get(0,0))     < stressMax))  inputValid=false;
      }
    }
    if(!(J > 0 && J_old > 0 && Jinc > 0)) inputValid=false;
    // Input doubles:
    if( !(pGPJ_old > 0.0) || !(pGPJ_old<defGradMax) )  inputValid=false;
    if( !(RoomTemperature >= 0.0) || !(RoomTemperature < paramMax) )  inputValid=false;
    if( !(pTemperature >= 0.0) || !(pTemperature < paramMax) )  inputValid=false;
    if( !(rho_orig > 0.0) || !(rho_orig<paramMax) )  inputValid=false;
    if( !(pVolume_new > 0.0) || !(pVolume_new<paramMax) )  inputValid=false;
    if( !(pMass > 0.0) || !(pMass<paramMax) )  inputValid=false;
    if( !(SpecificHeat >= 0.0) || !(SpecificHeat<paramMax) )  inputValid=false;
    if( !(pDamage >= 0.0) || !(pDamage<paramMax) )  inputValid=false;
    if( !(K >= 0.0) || !(K<paramMax) )  inputValid=false;
    if( !(flow >= 0.0) || !(flow<paramMax) )  inputValid=false;
    if( !(delT >= 0.0) || !(delT<paramMax) )  inputValid=false;
    // Input/Output doubles:
    if( !(*pGP_strain >= 0.0) || !(*pGP_strain<paramMax) )  inputValid=false;
    if( !(*pPlasticStrain >= 0.0) || !(*pPlasticStrain<paramMax) )  inputValid=false;
    if( !(*pPlasticEnergy >= 0.0) || !(*pPlasticEnergy<paramMax) )  inputValid=false;
    if( !(*pDamage_new >= 0.0) || !(*pDamage_new<paramMax) )  inputValid=false;
    if( !(*pGPJ > 0.0) || !(*pGPJ<paramMax) )  inputValid=false;
    if( !(*pGP_energy >= 0.0) || !(*pGP_energy<paramMax) )  inputValid=false;
    if( !(std::fabs(*pEnergy_new)<paramMax) )  inputValid=false;
    if( !(*damage_dt >= 0.0) || !(*damage_dt<paramMax) )  inputValid=false;
    if( !(*p_q >= 0.0) || !(*p_q<paramMax) )  inputValid=false;
    if( !(std::fabs(*se)<paramMax) )  inputValid=false;
    if( !(std::fabs(*pdTdt)<paramMax) )  inputValid=false;
    if( !(std::fabs(*pepsV)<paramMax) )  inputValid=false;
    if( !(*pgam >= 0.0) || !(*pgam<paramMax) )  inputValid=false;
    if( !(std::fabs(*pepsV_qs)<paramMax) )  inputValid=false;
    if( !(*pgam_qs >= 0.0) || !(*pgam_qs<paramMax) )  inputValid=false;

    if(!inputValid){
      std::stringstream msg;
      msg << "Invalid input to ComputeStressTensorInnerLoop() detected:\n";
      msg << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
      msg << "Debugging information:\n"
          << "F_old = " << pDefGrad << "\n"
          << "J_old = " << J_old << "\n"
          << "velGrad = " << pVelGrad << "\n"
          << "F_new = " << pDefGrad_new << "\n"
          << "J = " << J << "\n"
          << "Jinc = " << Jinc << "\n"
          << "bElBar = " << *bElBar << "\n"
          << "stress = " << *pStress << "\n"
          << "stress_qs = " << *pStress_qs << "\n"
          << "pGPJ_old:\t" << std::setw(20) << std::setprecision(12) << pGPJ_old << "\n"
          << "RoomTemperature:\t" << std::setw(20) << std::setprecision(12) << RoomTemperature << "\n"
          << "pTemperature:\t" << std::setw(20) << std::setprecision(12) << pTemperature << "\n"
          << "rho_orig:\t" << std::setw(20) << std::setprecision(12) << rho_orig << "\n"
          << "pVolume_new:\t" << std::setw(20) << std::setprecision(12) << pVolume_new << "\n"
          << "pMass:\t" << std::setw(20) << std::setprecision(12) << pMass << "\n"
          << "SpecificHeat:\t" << std::setw(20) << std::setprecision(12) << SpecificHeat << "\n"
          << "pDamage:\t" << std::setw(20) << std::setprecision(12) << pDamage << "\n"
          << "K:\t" << std::setw(20) << std::setprecision(12) << K << "\n"
          << "flow:\t" << std::setw(20) << std::setprecision(12) << flow << "\n"
          << "delT:\t" << std::setw(20) << std::setprecision(12) << delT << "\n"
          << "pGP_strain:\t" << std::setw(20) << std::setprecision(12) << *pGP_strain << "\n"
          << "pPlasticStrain:\t" << std::setw(20) << std::setprecision(12) << *pPlasticStrain << "\n"
          << "pPlasticEnergy:\t" << std::setw(20) << std::setprecision(12) << *pPlasticEnergy << "\n"
          << "pDamage_new:\t" << std::setw(20) << std::setprecision(12) << *pDamage_new << "\n"
          << "pGPJ:\t" << std::setw(20) << std::setprecision(12) << *pGPJ << "\n"
          << "pGP_energy:\t" << std::setw(20) << std::setprecision(12) << *pGP_energy << "\n"
          << "pEnergy_new:\t" << std::setw(20) << std::setprecision(12) << *pEnergy_new << "\n"
          << "damage_dt:\t" << std::setw(20) << std::setprecision(12) << *damage_dt << "\n"
          << "p_q:\t" << std::setw(20) << std::setprecision(12) << *p_q << "\n"
          << "se:\t" << std::setw(20) << std::setprecision(12) << *se << "\n"
          << "pdTdt:\t" << std::setw(20) << std::setprecision(12) << *pdTdt << "\n"
          << "pepsV:\t" << std::setw(20) << std::setprecision(12) << *pepsV << "\n"
          << "pepsV_qs:\t" << std::setw(20) << std::setprecision(12) << *pepsV_qs << "\n"
          << "pgam:\t" << std::setw(20) << std::setprecision(12) << *pgam << "\n"
          << "pgam_qs:\t" << std::setw(20) << std::setprecision(12) << *pgam_qs << "\n";
      if( !(pDamage >= 0.0) || !(pDamage<paramMax) ){
        msg.setf(std::ios::dec|std::ios::scientific);
        msg << "pDamage was invalid, initial crack parameters:\n"
            << "Flaw densities:\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pFlawNumber_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Initial flaw sizes:\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pflawSize_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Wing crack lengths (input):\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pWingLength_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n";
      }
    
      throw std::runtime_error(msg.str());
    }
    
    double c_dil;
    double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
    double pIEl(bElBar->trace()/3.0);
    *pLocalized_new = pLocalized;
    *pDamage_new = flags.useDamage ? pDamage : 0.0;
    advanceTimeSigmaL( // Data Structures
                      flags,
                      initialData,
                      flawDistData,
                      brittle_damage,
                      gpData,
                      artificialViscosity,
                      eos,
                      // Input Matrix:
                      pVelGrad,
                      // Input/OutputMatrix:
                      pStress,
                      pStress_qs,
                      // Input double
                      delT,
                      J_old,
                      J,
                      pTemperature,
                      rho_orig,
                      dx_ave, /* mean spatial dimension for art visc calc */
                      &pIEl,
                      pPlasticStrain,
                      pPlasticEnergy,
                      pDamage_new,
                      pGPJ,
                      pGP_strain,
                      pGP_energy,
                      pEnergy_new,
                      damage_dt,
                      pepsV,
                      pgam,
                      pepsV_qs,
                      pgam_qs,
                      pLocalized_new,
                      // Output only double:
                      p_q,
                      pdTdt,
                      &c_dil,
                      pWingLength_array,
                      pFlawNumber_array,
                      pflawSize_array,
                      pWingLength_array_new
                       );
    if(*pLocalized_new) ++(*totalLocalizedParticle);
    // Compute wave speed at each particle, store the maximum
    Vector3 pvel = pVelocity;
    (*WaveSpeed)[0] = std::max(c_dil+fabs(pvel.x()),WaveSpeed->x());
    (*WaveSpeed)[1] = std::max(c_dil+fabs(pvel.y()),WaveSpeed->y());
    (*WaveSpeed)[2] = std::max(c_dil+fabs(pvel.z()),WaveSpeed->z());

    // compute bElBar from pStress and pIEl:
    PState state;
    state.pressure = -pStress->trace()*onethird;
    state.temperature = pTemperature;
    state.initialTemperature = RoomTemperature;
    state.initialDensity = rho_orig;
    state.density = rho_orig*(*pGPJ)/J;
    state.initialVolume = pVolume_new/J;
    state.volume = pVolume_new;
    state.specificHeat = SpecificHeat;
    state.energy = (state.temperature-state.initialTemperature)*state.specificHeat; // This is the internal energy do
    state.initialBulkModulus = eos->computeBulkModulus(state.initialDensity, state.density);
    state.initialShearModulus = initialData.tauDev;
    state.bulkModulus  = calculateBulkPrefactor(*pDamage_new, state, J/(*pGPJ))*state.initialBulkModulus;
    state.shearModulus = calculateShearPrefactor(*pDamage_new, state)*state.initialShearModulus;
    double p = pStress->trace()/3.0;
    *bElBar = (*pStress-identity*p)*J/state.shearModulus + identity*pIEl;

    // Compute the strain energy in the volume:
    *se    += pVolume_new* (*pEnergy_new);

    bool outputValid(true);
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
        if(!(std::fabs(bElBar->get(0,0))       < defGradMax)) outputValid=false;
        if(!(std::fabs(pVelGrad.get(0,0))      < velGradMax)) outputValid=false;
        if(!(std::fabs(pStress_qs->get(0,0))   < stressMax))  outputValid=false;
        if(!(std::fabs(pStress->get(0,0))      < stressMax))  outputValid=false;
      }
    }
    // Input doubles:
    if( !(pGPJ_old > 0.0) || !(pGPJ_old<defGradMax) )  outputValid=false;
    if( !(RoomTemperature >= 0.0) || !(RoomTemperature < paramMax) )  outputValid=false;
    if( !(pTemperature >= 0.0) || !(pTemperature < paramMax) )  outputValid=false;
    if( !(rho_orig > 0.0) || !(rho_orig<paramMax) )  outputValid=false;
    if( !(pVolume_new > 0.0) || !(pVolume_new<paramMax) )  outputValid=false;
    if( !(pMass > 0.0) || !(pMass<paramMax) )  outputValid=false;
    if( !(SpecificHeat >= 0.0) || !(SpecificHeat<paramMax) )  outputValid=false;
    if( !(pDamage >= 0.0) || !(pDamage<paramMax) )  outputValid=false;
    if( !(K >= 0.0) || !(K<paramMax) )  outputValid=false;
    if( !(flow >= 0.0) || !(flow<paramMax) )  outputValid=false;
    if( !(delT >= 0.0) || !(delT<paramMax) )  outputValid=false;
    // Input/Output doubles:
    if( !(*pGP_strain >= 0.0) || !(*pGP_strain<paramMax) )  outputValid=false;
    if( !(*pPlasticStrain >= 0.0) || !(*pPlasticStrain<paramMax) )  outputValid=false;
    if( !(*pPlasticEnergy >= 0.0) || !(*pPlasticEnergy<paramMax) )  outputValid=false;
    if( !(*pDamage_new >= 0.0) || !(*pDamage_new<paramMax) )  outputValid=false;
    if( !(*pGPJ > 0.0) || !(*pGPJ<paramMax) )  outputValid=false;
    if( !(*pGP_energy >= 0.0) || !(*pGP_energy<paramMax) )  outputValid=false;
    if( !(std::fabs(*pEnergy_new)<paramMax) )  outputValid=false;
    if( !(*damage_dt >= 0.0) || !(*damage_dt<paramMax) )  outputValid=false;
    if( !(*p_q >= 0.0) || !(*p_q<paramMax) )  outputValid=false;
    if( !(std::fabs(*se)<paramMax) )  outputValid=false;
    if( !(std::fabs(*pdTdt)<paramMax) )  outputValid=false;
    if( !(std::fabs(*pepsV)<paramMax) )  outputValid=false;
    if( !(*pgam >= 0.0) || !(*pgam<paramMax) )  outputValid=false;
    if( !(std::fabs(*pepsV_qs)<paramMax) )  outputValid=false;
    if( !(*pgam_qs >= 0.0) || !(*pgam_qs<paramMax) )  outputValid=false;

    if(!outputValid){
      std::stringstream msg;
      msg << "Invalid output from ComputeStressTensorInnerLoop() detected:\n";
      msg << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
      msg << "Debugging information:\n"
          << "velGrad = " << pVelGrad << "\n"
          << "bElBar = " << *bElBar << "\n"
          << "stress = " << *pStress << "\n"
          << "stress_qs = " << *pStress_qs << "\n"
          << "pGPJ_old:\t" << std::setw(20) << std::setprecision(12) << pGPJ_old << "\n"
          << "RoomTemperature:\t" << std::setw(20) << std::setprecision(12) << RoomTemperature << "\n"
          << "pTemperature:\t" << std::setw(20) << std::setprecision(12) << pTemperature << "\n"
          << "rho_orig:\t" << std::setw(20) << std::setprecision(12) << rho_orig << "\n"
          << "pVolume_new:\t" << std::setw(20) << std::setprecision(12) << pVolume_new << "\n"
          << "pMass:\t" << std::setw(20) << std::setprecision(12) << pMass << "\n"
          << "SpecificHeat:\t" << std::setw(20) << std::setprecision(12) << SpecificHeat << "\n"
          << "pDamage:\t" << std::setw(20) << std::setprecision(12) << pDamage << "\n"
          << "K:\t" << std::setw(20) << std::setprecision(12) << K << "\n"
          << "flow:\t" << std::setw(20) << std::setprecision(12) << flow << "\n"
          << "delT:\t" << std::setw(20) << std::setprecision(12) << delT << "\n"
          << "pGP_strain:\t" << std::setw(20) << std::setprecision(12) << *pGP_strain << "\n"
          << "pPlasticStrain:\t" << std::setw(20) << std::setprecision(12) << *pPlasticStrain << "\n"
          << "pPlasticEnergy:\t" << std::setw(20) << std::setprecision(12) << *pPlasticEnergy << "\n"
          << "pDamage_new:\t" << std::setw(20) << std::setprecision(12) << *pDamage_new << "\n"
          << "pGPJ:\t" << std::setw(20) << std::setprecision(12) << *pGPJ << "\n"
          << "pGP_energy:\t" << std::setw(20) << std::setprecision(12) << *pGP_energy << "\n"
          << "pEnergy_new:\t" << std::setw(20) << std::setprecision(12) << *pEnergy_new << "\n"
          << "damage_dt:\t" << std::setw(20) << std::setprecision(12) << *damage_dt << "\n"
          << "p_q:\t" << std::setw(20) << std::setprecision(12) << *p_q << "\n"
          << "se:\t" << std::setw(20) << std::setprecision(12) << *se << "\n"
          << "pdTdt:\t" << std::setw(20) << std::setprecision(12) << *pdTdt << "\n"
          << "pepsV:\t" << std::setw(20) << std::setprecision(12) << *pepsV << "\n"
          << "pepsV_qs:\t" << std::setw(20) << std::setprecision(12) << *pepsV_qs << "\n"
          << "pgam:\t" << std::setw(20) << std::setprecision(12) << *pgam << "\n"
          << "pgam_qs:\t" << std::setw(20) << std::setprecision(12) << *pgam_qs << "\n";
      if( !(*pDamage_new >= 0.0) || !(*pDamage_new<paramMax) ){
        msg << "pDamage_new was invalid, initial crack parameters:\n"
            << "Flaw densities:\n";
        msg.setf(std::ios::dec|std::ios::scientific);
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pFlawNumber_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Initial flaw sizes:\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pflawSize_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Wing crack lengths (input):\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pWingLength_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Wing crack lengths (output):\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pWingLength_array_new)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n";
      }
      throw std::runtime_error(msg.str());
    }
  }

  void advanceTimeSigmaL(
                         // Data Structures
                         const Flags flags,
                         const ModelData initialData,
                         const flawDistributionData flawDistData,
                         const BrittleDamageData brittle_damage,
                         const granularPlasticityData gpData,
                         const ArtificialViscosity artificialViscosity,
                         const PortableMieGruneisenEOSTemperature *eos,
                         // Input Matrix:
                         const Matrix3x3 velGrad,
                         // Input/OutputMatrix:
                         Matrix3x3 *pStress,
                         Matrix3x3 *pStress_qs,
                         // Input double
                         const double delT,
                         const double J_old,
                         const double J,
                         const double pTemperature,
                         const double rho_orig,
                         const double dx_ave, /* mean spatial dimension for art visc calc */
                         // Input/Output double
                         double *pIEl,
                         double *pPlasticStrain,
                         double *pPlasticEnergy,
                         double *pDamage,
                         double *pGPJ,
                         double *pGP_strain,
                         double *pGP_energy,
                         double *pEnergy,
                         double *damage_dt,
                         double *pepsV,
                         double *pgam,
                         double *pepsV_qs,
                         double *pgam_qs,
                         int *pLocalized,
                         // Output only double:
                         double *p_q_out,
                         double *pdTdt_out,
                         double *c_dil_out,
                         // Input std::vector
                         const std::vector<double> *pWingLength_array,
                         const std::vector<double> *pFlawNumber_array,
                         const std::vector<double> *pflawSize_array,
                         // Output std::vector
                         std::vector<double> *pWingLength_array_new,
                         const bool assumeRotatedTensors
                         )
  {
    const std::vector<double>::size_type nBins = pWingLength_array->size();
    const Matrix3x3 identity(true);
    const Matrix3x3 pDefGradInc = identity + velGrad*delT + (velGrad*velGrad)*(0.5*delT*delT);
    const Matrix3x3 pDeformRate = (velGrad + velGrad.transpose())*0.5;
    Matrix3x3 Rinc(true), pStress_tmp(*pStress), Vinc_tmp(pDefGradInc);
    // If the incomming tensors are rotated, then the incomming velocity gradient
    // should actually be the rate of deformation. 
    if(!assumeRotatedTensors){
      pDefGradInc.polarRotationRMB(&Rinc);               // Compute the incremental rotation
      *pStress_qs  = Rinc*((*pStress_qs)*Rinc.transpose()); // Rotate the reference stress forward.
      Vinc_tmp    = pDefGradInc*Rinc.transpose();
      pStress_tmp = Rinc*((*pStress)*Rinc.transpose());
    }
    const Matrix3x3 pStress_old(pStress_tmp);
    const Matrix3x3 Vinc(Vinc_tmp);

    *pdTdt_out = 0.0;
    *p_q_out   = 0.0;
    *c_dil_out = 0.0;
#ifndef NDEBUG
    const double Jinc = J/J_old;
    bool inputValid(true);
    const double velGradMax(1e12);
    const double stressMax(1e24);
    const double paramMax(1e37);
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
        if(!(std::fabs(velGrad.get(0,0))      < velGradMax)) inputValid=false;
        if(!(std::fabs(pStress_qs->get(0,0))   < stressMax))  inputValid=false;
        if(!(std::fabs(pStress->get(0,0))      < stressMax))  inputValid=false;
      }
    }
    if( !(J > 0 && J_old > 0 && Jinc > 0)) inputValid=false;
    if( !(pTemperature >= 0.0) || !(pTemperature < paramMax) )  inputValid=false;
    if( !(rho_orig > 0.0) || !(rho_orig<paramMax) )  inputValid=false;
    if( !(*pDamage >= 0.0) || !(*pDamage<paramMax) )  inputValid=false;
    if( !(delT >= 0.0) || !(delT<paramMax) )  inputValid=false;
    // Input/Output doubles
    if( !(*pIEl-1.0   >= 0.0) || !(*pIEl < paramMax) ){
      if (*pIEl-1.0 > -1.0e-8) *pIEl = 1.0;
      else inputValid=false;
    }
    if( !(*pGP_strain >= 0.0) || !(*pGP_strain<paramMax) )  inputValid=false;
    if( !(*pPlasticStrain >= 0.0) || !(*pPlasticStrain<paramMax) )  inputValid=false;
    if( !(*pPlasticEnergy >= 0.0) || !(*pPlasticEnergy<paramMax) )  inputValid=false;
    if( !(*pGPJ > 0.0) || !(*pGPJ<paramMax) )  inputValid=false;
    if( !(*pGP_energy >= 0.0) || !(*pGP_energy<paramMax) )  inputValid=false;
    if( !(*damage_dt >= 0.0) || !(*damage_dt<paramMax) )  inputValid=false;
    if( !(*p_q_out >= 0.0) || !(*p_q_out<paramMax) )  inputValid=false;
    if( !(std::fabs(*pdTdt_out)<paramMax) )  inputValid=false;
    if( !(std::fabs(*pepsV)<paramMax) )  inputValid=false;
    if( !(*pgam >= 0.0) || !(*pgam<paramMax) )  inputValid=false;
    if( !(std::fabs(*pepsV_qs)<paramMax) )  inputValid=false;
    if( !(*pgam_qs >= 0.0) || !(*pgam_qs<paramMax) )  inputValid=false;
    if(!inputValid){
      std::stringstream msg;
      msg << "Invalid input to advanceTimeSigmaL() detected:\n";
      msg << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
      msg << "Debugging information:\n"
          << "J_old = "          << J_old << "\n"
          << "velGrad = "        << velGrad << "\n"
          << "J = "              << J << "\n"
          << "Jinc = "           << Jinc << "\n"
          << "stress = "         << *pStress << "\n"
          << "stress_qs = "      << *pStress_qs << "\n"
          << "pTemperature:\t"   << std::setw(20) << std::setprecision(12) << pTemperature << "\n"
          << "rho_orig:\t"       << std::setw(20) << std::setprecision(12) << rho_orig << "\n"
          << "pDamage:\t"        << std::setw(20) << std::setprecision(12) << *pDamage << "\n"
          << "delT:\t"           << std::setw(20) << std::setprecision(12) << delT << "\n"
          << "pGP_strain:\t"     << std::setw(20) << std::setprecision(12) << *pGP_strain << "\n"
          << "pPlasticStrain:\t" << std::setw(20) << std::setprecision(12) << *pPlasticStrain << "\n"
          << "pPlasticEnergy:\t" << std::setw(20) << std::setprecision(12) << *pPlasticEnergy << "\n"
          << "pIEl - 1.0:\t"     << std::setw(20) << std::setprecision(12) << *pIEl-1.0 << "\n"
          << "pGPJ:\t"           << std::setw(20) << std::setprecision(12) << *pGPJ << "\n"
          << "pGP_energy:\t"     << std::setw(20) << std::setprecision(12) << *pGP_energy << "\n"
          << "damage_dt:\t"      << std::setw(20) << std::setprecision(12) << *damage_dt << "\n"
          << "p_q:\t"            << std::setw(20) << std::setprecision(12) << *p_q_out << "\n"
          << "pdTdt:\t"          << std::setw(20) << std::setprecision(12) << *pdTdt_out << "\n"
          << "pepsV:\t"          << std::setw(20) << std::setprecision(12) << *pepsV << "\n"
          << "pepsV_qs:\t"       << std::setw(20) << std::setprecision(12) << *pepsV_qs << "\n"
          << "pgam:\t"           << std::setw(20) << std::setprecision(12) << *pgam << "\n"
          << "pgam_qs:\t"        << std::setw(20) << std::setprecision(12) << *pgam_qs << "\n";
      if( !(*pDamage >= 0.0) || !(*pDamage<paramMax) ){
        msg.setf(std::ios::dec|std::ios::scientific);
        msg << "pDamage was invalid, initial crack parameters:\n"
            << "Flaw densities:\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pFlawNumber_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Initial flaw sizes:\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pflawSize_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Wing crack lengths (input):\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pWingLength_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n";
      }
    
      throw std::runtime_error(msg.str());
    }
#endif
    const double invDelT = delT>0 ? 1.0/delT : 0.0;
    const double pDamage_old(*pDamage);
    const CMData eosData = eos->getEOSData(); // needed for specific heat and reference temperature
    double JGP;
    double JEL;
    if(flags.useGranularPlasticity) {
      JGP = *pGPJ;
      JEL = J/JGP;
    } else {
      JEL = J;
      JGP = 1.0;
    }

    const double pGPJ_old = JGP;
    double rho_cur  = rho_orig/JEL;
    // Set up the PlasticityState (for t_n+1) --------------------------------
    PState state;
    state.pressure = -pStress->trace()*onethird;
    state.temperature = pTemperature;
    state.initialTemperature = eosData.theta_0;
    state.density = rho_cur;
    state.initialDensity = rho_orig;
    state.initialVolume = 1.0;
    state.volume = state.initialVolume*J;

    state.specificHeat = eosData.C_v;
    state.energy = (state.temperature-state.initialTemperature)*state.specificHeat; // This is the internal energy do
    // to the temperature
    // Set the moduli:
    state.initialBulkModulus = eos->computeBulkModulus(rho_orig, rho_cur);
    state.bulkModulus = state.initialBulkModulus;
    // The shear modulus is easy b/c it does not depend on the deformation
    state.shearModulus = initialData.tauDev; // This is changed later if there is damage
    state.initialShearModulus = initialData.tauDev;

    // Compute bElBar from the old stress:
    state.bulkModulus  = calculateBulkPrefactor(pDamage_old, state, JEL)*state.initialBulkModulus;
    state.shearModulus = calculateShearPrefactor(pDamage_old, state)*state.initialShearModulus;
    double p_old = pStress_old.trace()/3.0;
    const double IEl_old = *pIEl;
    // bElBar has been rotated to the current frame:
    const Matrix3x3 bElBar = (pStress_old-identity*p_old)*J_old/state.shearModulus + identity*IEl_old; 

    state.bulkModulus         = state.initialBulkModulus;
    state.initialShearModulus = initialData.tauDev;

    // End PlasticityState setup -----------------------------------------

    // Step 0: Compute input parameters for constitutive update:
    Matrix3x3 bElBarTrial;
    double IEl;
    {
      // Get the volume preserving part of the deformation gradient increment
      Matrix3x3 fBar = Vinc/cbrt(Vinc.determinant()); // Because the old stress was already rotated
      // Compute the trial elastic part of the volume preserving
      // part of the left Cauchy-Green deformation tensor
      bElBarTrial = fBar*bElBar*fBar.transpose();
      IEl   = onethird*bElBarTrial.trace();
      if(IEl < 1.0){
        bElBarTrial -= identity*IEl;
        double bElBarDevNorm = (bElBar-identity*IEl_old).norm();
        IEl = bElBarDevNorm > 0.0 ?
          (IEl_old-1.0)*(bElBarTrial.norm())/bElBarDevNorm + 1.0 :
          1.0 + bElBarTrial.norm();
        bElBarTrial += identity*IEl;
      }
    }
    Matrix3x3 bElBar_new = bElBarTrial;
    
    // Step 1: Compute Plastic flow --------------------------------------
    if(flags.usePlasticity) {
      double muBar = IEl*state.shearModulus;
      Matrix3x3 tauDevTrial = (bElBarTrial - identity*IEl)*state.shearModulus;
      double sTnorm      = tauDevTrial.norm();

      // Check for plastic loading
      double K(initialData.K);
      double alpha  = *pPlasticStrain;
      double flow(initialData.FlowStress+K*alpha);
      if( !(flow > 0.0) ){
        flow = 0.0;
        K    = 0.0;
        *pLocalized |= 4;
      }
      double fTrial = sTnorm - (flow);
      if (fTrial > 0.0) {
        // plastic
        // Compute increment of slip in the direction of flow (with viscoplastic effects
        double delgamma = (fTrial/(2.0*muBar)) / (initialData.timeConstant*invDelT + 1.0 + (K/(3.0*muBar)));
        double delTauNorm = 2.0*muBar*delgamma;
        if( delTauNorm < sTnorm ){
          Matrix3x3 normal   = tauDevTrial/sTnorm;
          // The actual shear stress
          tauDevTrial -= normal*2.0*muBar*delgamma;
          double IEl_tr = IEl;
          IEl = (IEl_tr - 1) * (tauDevTrial.normSquared()/(sTnorm*sTnorm)) + 1;
          bElBar_new      = tauDevTrial/state.shearModulus + identity*IEl;
        } else {
          // Do not let the stress cross zero
          IEl         = 1.0;
          delgamma    = sTnorm/(2.0*muBar);
          bElBar_new  = identity;
          tauDevTrial = Matrix3x3(0.0);
          *pLocalized |= 4;
        }
        // Deal with history variables
        *pPlasticStrain = alpha + delgamma;
        *pPlasticEnergy += delgamma*tauDevTrial.norm();
        *pdTdt_out      += delgamma*tauDevTrial.norm()*invDelT/(rho_orig * state.specificHeat);
      }
    }
    
    // Step 2: Compute damage growth -------------------------------------
    if(flags.useDamage) {
      // if using damage calculate the new bulk and shear modulus
      // based on the particle damage: Update the damage based on the
      // stress from the previous timestep. Then calculate the current
      // stiffness and then update the stress, and plastic terms.

      // copy the flaw size, wing crack length, and number of flaws in the
      // bin to a std::vector:

      double oldDamage = pDamage_old; // Damage at the beginning of the step
      double damageTrial = oldDamage;
      double currentDamage;
      std::vector<double> L_old(*pWingLength_array);
      std::vector<double> L_new(*pWingLength_array);
      std::vector<double> L_dot_new(nBins,0);
      std::vector<double> s(*pflawSize_array);
      std::vector<double> N(*pFlawNumber_array);

      state.bulkModulus  = calculateBulkPrefactor(oldDamage, state, JEL)*state.initialBulkModulus;
      state.shearModulus = calculateShearPrefactor(oldDamage, state)*state.initialShearModulus;

      // Use the predicted stress instead of the stress from the previous
      // increment:
      double JEL_old = J_old/JGP;
      Matrix3x3 oldStress = pStress_old/J_old;
      if (oldDamage > brittle_damage.maxDamage - brittle_damage.maxDamage*1e-8) {
        currentDamage = brittle_damage.maxDamage;
        L_new = L_old;
      } else if(brittle_damage.useOldStress) {
        // basic forward euler, I only used information from the previous timestep
        // to compute the new values.
        currentDamage =  calculateDamageGrowth(brittle_damage, oldStress,
                                               N.data(), s.data(), L_old.data(),
                                               oldDamage, L_new.data(), L_dot_new.data(),
                                               delT, *pLocalized, state,
                                               nBins);
      } else {
        // Try to use a more consistant formulation, so that the current stress
        // is used to calculate the damage growth rate.
        double JEL_new = JEL;
        double p_new = computePressure(eos, identity*pow(JEL_new, onethird), state, oldDamage);
        Matrix3x3 bElBar_test = bElBar_new;
        double IEl_new = onethird*bElBar_test.trace();
        // Calculate the trial stress using the current strain, and the old damage:
        Matrix3x3 stress_trial = (bElBar_test-identity*IEl_new)*state.shearModulus + identity*p_new;
        // Update the damage state:
        damageTrial =  calculateDamageGrowth(brittle_damage, stress_trial, N.data(), s.data(),
                                             L_old.data(), oldDamage,
                                             L_new.data(), L_dot_new.data(), delT, *pLocalized, state,
                                             nBins);
        // Compute the increment in damage:
        double damageInc = damageTrial - oldDamage;
        if((damageInc > brittle_damage.maxDamageInc) && (*pLocalized < 1) && (delT > 0.0)) {
          // The timesteps to resolve the damage process need to be smaller
          double numSteps = ceil(damageInc/brittle_damage.maxDamageInc);
          double invNumSteps = 1.0/numSteps;

          // Compute the strain increment per sub step:
          Matrix3x3 devStrain_new = bElBar_test-identity*IEl_new;
          Matrix3x3 devStrain_old = bElBar-identity*IEl_old;
          Matrix3x3 devStrain_inc = (devStrain_new-devStrain_old)*invNumSteps;

          // Increment in the volumetric strain:
          double volStrain_old  = 0.5*(JEL_old - 1.0/JEL_old);
          double volStrain_new  = 0.5*(JEL_new - 1.0/JEL_new);
          double volStrain_inc  = (volStrain_new-volStrain_old)*invNumSteps;

          // Sub increment in time:
          double dt = delT*invNumSteps;

          // initialze:
          devStrain_new = devStrain_old;
          volStrain_new = volStrain_old;
          currentDamage = oldDamage;

          // subloop:
          for(int i = 0; i<numSteps; i++) {
            // Compute the stress:
            state.bulkModulus  = calculateBulkPrefactor(currentDamage, state, JEL)
              *state.initialBulkModulus;
            state.shearModulus = calculateShearPrefactor(currentDamage, state)
              *state.initialShearModulus;
            double J_oneThird = powf(1.0/(1-volStrain_new), onethird);
            double pressure = computePressure(eos, identity*J_oneThird, state, currentDamage);
            stress_trial = devStrain_new*state.shearModulus+identity*pressure;
            currentDamage =  calculateDamageGrowth(brittle_damage, stress_trial,
                                                   N.data(), s.data(), L_old.data(), currentDamage,
                                                   L_new.data(), L_dot_new.data(), dt, *pLocalized, state,
                                                   nBins);
            L_old = L_new;
            volStrain_new += volStrain_inc;
            devStrain_new += devStrain_inc;
          }
        } else {
          currentDamage = damageTrial;
        }
      } // End if(useOldStress)

      // Compute the desired timestep based on the damage evolution:
      if (currentDamage-oldDamage > brittle_damage.maxDamageInc) {
        *damage_dt = std::min(*damage_dt, (brittle_damage.maxDamageInc /
                                         (currentDamage - oldDamage)) * delT);
      }

      // Set the damage, and wing crack sizes at the end of the step:
      *pWingLength_array_new = L_new;
      *pDamage = currentDamage;
      if(currentDamage >= brittle_damage.criticalDamage){
        *pLocalized |= 1;
      }
      // Update the moduli:
      state.bulkModulus  = calculateBulkPrefactor(currentDamage, state, JEL)*state.initialBulkModulus;
      state.shearModulus = calculateShearPrefactor(currentDamage, state)*state.initialShearModulus;
    } // end if(flags.useDamage)

    // End damage computation -------------------------------------------

    // Step 3: Calculate the flow due to Granular Plasticity:
    if(flags.useGranularPlasticity) {
      switch(gpData.GPModelType){
      case TwoSurface:
        calcGranularFlow(flags, brittle_damage, gpData, eos,
                         delT, *pDamage, J, pGPJ_old,
                         &bElBar_new, &state, pGPJ, pGP_strain,
                         pGP_energy, pdTdt_out
                         );
        JGP = *pGPJ;
        JEL = J/JGP;
        break;
      case SingleSurface:
        // This model is written for the Cauchy stress
        
        // Call the granular flow model that I have been working on:
        // Assign history variables and material parameters:
        double JEL_old = J_old/pGPJ_old;
        PState state_old(state);
        state_old.density      = state.initialDensity/JEL_old;
        state_old.bulkModulus  = calculateBulkPrefactor(pDamage_old, state, JEL)*state.initialBulkModulus;
        state_old.shearModulus = calculateShearPrefactor(pDamage_old, state)*state.initialShearModulus;
        Matrix3x3 oldStress = pStress_old;
        Matrix3x3 bElBarTrial = bElBar_new; // This may have been modified in the Plasticity section
        double IEl_tr         = onethird*bElBarTrial.trace();
        Matrix3x3 tauDevTrial = (bElBarTrial - identity*IEl_tr)*state.shearModulus;
        double p_trial;
        p_trial = computePressure(eos, identity * cbrt(JEL), state, *pDamage);

        GFMS::matParam GFMatParams(gpData.GFMSmatParams);
        GFMatParams.bulkMod  =
          state.bulkModulus/state.initialBulkModulus *
          eos->computeIsentropicBulkModulus(state.initialDensity, state.density, state.temperature);
        GFMatParams.shearMod = state.shearModulus;
        GFMatParams.bulkMod  /= J; // It appears that the bulk modulus is also for the Kirchoff stress
        GFMatParams.shearMod /= J; // The shear modulus in state is for the Kirchoff stress

        // Compute the starting stress:
        SymMat3::SymMatrix3 sigma_tr,sigma_0,sigma_0qs,sigma_1,sigma_1qs, D;
        sigma_tr.set(0,0,tauDevTrial.get(0,0) + p_trial);
        sigma_tr.set(1,1,tauDevTrial.get(1,1) + p_trial);
        sigma_tr.set(2,2,tauDevTrial.get(2,2) + p_trial);
        sigma_tr.set(1,2,0.5*(tauDevTrial.get(1,2) + tauDevTrial.get(2,1)));
        sigma_tr.set(0,2,0.5*(tauDevTrial.get(0,2) + tauDevTrial.get(2,0)));
        sigma_tr.set(0,1,0.5*(tauDevTrial.get(1,0) + tauDevTrial.get(0,1)));
        sigma_tr /= J;          // Convert to the Cauchy stress

        sigma_0.set(0,0,oldStress.get(0,0));
        sigma_0.set(1,1,oldStress.get(1,1));
        sigma_0.set(2,2,oldStress.get(2,2));
        sigma_0.set(1,2,0.5*(oldStress.get(1,2) + oldStress.get(2,1)));
        sigma_0.set(0,2,0.5*(oldStress.get(0,2) + oldStress.get(2,0)));
        sigma_0.set(0,1,0.5*(oldStress.get(1,0) + oldStress.get(0,1)));

        SymMat3::SymMatrix3 dSigma(sigma_tr-sigma_0);
        D = (dSigma.isotropic()/(3.0*GFMatParams.bulkMod) + dSigma.deviatoric()/(2.0*GFMatParams.shearMod))*invDelT;
        
        sigma_0qs.set(0,0,pStress_qs->get(0,0));
        sigma_0qs.set(1,1,pStress_qs->get(1,1));
        sigma_0qs.set(2,2,pStress_qs->get(2,2));
        sigma_0qs.set(1,2,0.5*(pStress_qs->get(1,2) + pStress_qs->get(2,1)));
        sigma_0qs.set(0,2,0.5*(pStress_qs->get(0,2) + pStress_qs->get(2,0)));
        sigma_0qs.set(0,1,0.5*(pStress_qs->get(1,0) + pStress_qs->get(0,1)));

        if(flags.useDamage && !(*pDamage > brittle_damage.criticalDamage) ){
          // Carry forward:
          *pepsV_qs   = 0.0;
          *pgam_qs    = 0.0;
          *pepsV      = 0.0;
          *pgam       = 0.0;
          *pGP_energy = 0.0;
          *pGPJ       = 1.0;
          *pGP_strain = 0.0;
          pStress_qs->set(0,0, 0.0);
          pStress_qs->set(0,1, 0.0);
          pStress_qs->set(0,2, 0.0);
          pStress_qs->set(1,0, 0.0);
          pStress_qs->set(1,1, 0.0);
          pStress_qs->set(1,2, 0.0);
          pStress_qs->set(2,0, 0.0);
          pStress_qs->set(2,1, 0.0);
          pStress_qs->set(2,2, 0.0);
        } else {
          if(flags.useDamage && pDamage_old < brittle_damage.criticalDamage){
            // For this one step take the granular flow model from 0 to sigma_tr:
            D = (sigma_tr.isotropic()/(3.0*GFMatParams.bulkMod) + sigma_tr.deviatoric()/(2.0*GFMatParams.shearMod))*invDelT;
            sigma_0 *= 0;
          }
          double epsv_1,epsv_0(*pepsV), gam_1, gam_0(*pgam);
          double epsv1_qs,epsv0_qs(*pepsV_qs), gam1_qs, gam0_qs(*pgam_qs);
          double GFMSIncPlasWork =
            GFMS::integrateRateDependent(delT, &D, &sigma_0, &sigma_0qs,
                                         epsv_0, gam_0, epsv0_qs, gam0_qs,
                                         &GFMatParams, &gpData.GFMSsolParams,
                                         &sigma_1, &sigma_1qs, &epsv_1, &gam_1,
                                         &epsv1_qs, &gam1_qs
                                         );
          *pepsV    = epsv_1;
          *pgam     = gam_1;
          *pepsV_qs = epsv1_qs;
          *pgam_qs  = gam1_qs;

          JGP = exp(epsv_1);
          // JGP = exp(epsv_1-epsv_0)*pGPJ_old;
          // double p_target = sigma_1.trace()*J/3.0;
          // JGP = calcJGPFromPressure(eos, &state, *pDamage, JGP, J, p_target);
          JEL = J/JGP;
          state.density = rho_orig/JEL;
          *pGPJ = JGP;
          *pGP_strain += (gam_1-gam_0);
          // update bElBar_new:
          Matrix3x3 tauDev(0.0);
          for (int i = 0; i<3; ++i){
            tauDev.set(i,i, sigma_1.get(i,i)-sigma_1.trace()/3.0);
            pStress_qs->set(i,i, sigma_1qs.get(i,i));
            if(i<2){
              for (int j = i+1; j<3; ++j){
                tauDev.set(i,j, sigma_1.get(i,j));
                tauDev.set(j,i, sigma_1.get(i,j));
                pStress_qs->set(i,j, sigma_1qs.get(i,j));
                pStress_qs->set(j,i, sigma_1qs.get(i,j));
              }
            }
          }
          tauDev *= J;

          double IEl(IEl_tr);
          if(tauDevTrial.normSquared()>tauDev.normSquared()){
            IEl = (IEl_tr-1.0)*(tauDev.normSquared()/tauDevTrial.normSquared()) + 1.0;
          } 
          bElBar_new = (tauDev)/(state.shearModulus) + identity*IEl;
          // Compute the changes in energy:
          double delEnergy = GFMSIncPlasWork*J;
          if ( !( (delEnergy >= 0.0) || (GFMSIncPlasWork >= 0.0) ) ){
            double extWork = (sigma_1+sigma_0).Contract(D)*(J*0.5)*delT;
            if( !( (delEnergy >= -std::fabs(extWork)*gpData.GFMSsolParams.relToll) ||
                   (GFMSIncPlasWork >= -std::fabs(extWork)*gpData.GFMSsolParams.relToll)
                   ) ){
              const SymMat3::SymMatrix3 symI(true);
              SymMat3::SymMatrix3 delSigma = sigma_tr-sigma_1;
              SymMat3::SymMatrix3 DpInc    =
                delSigma.isotropic()/(3.0*GFMatParams.bulkMod) +
                delSigma.deviatoric()/(2.0*GFMatParams.shearMod);
              SymMat3::SymMatrix3 Dp = DpInc*invDelT;
              std::stringstream msg;
              msg << "Negative increment in plastic work due to granular flow detected:\n";
              msg << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
              msg << "delEnergy: " << delEnergy << " GFMSIncPlasWork: " << GFMSIncPlasWork << "\n"
                  << "ExternalWork= " << extWork << "\n";
              msg << "Debugging information:\n"
                  << " ////// Begin C++ input //////// \n"
                  << "GFMS::matParam GFMatParams; \n"
                  << "GFMatParams.bulkMod= " <<  GFMatParams.bulkMod << ";\n"
                  << "GFMatParams.shearMod= " << GFMatParams.shearMod << ";\n"
                  << "GFMatParams.m0= " <<  GFMatParams.m0 << ";\n"
                  << "GFMatParams.m1= " <<  GFMatParams.m1 << ";\n"
                  << "GFMatParams.m2= " << GFMatParams.m2 << ";\n"
                  << "GFMatParams.p0= " << GFMatParams.p0 << ";\n"
                  << "GFMatParams.p1= " << GFMatParams.p1 << ";\n"
                  << "GFMatParams.p2= " << GFMatParams.p2 << ";\n"
                  << "GFMatParams.p3= " << GFMatParams.p3 << ";\n"
                  << "GFMatParams.p4= " << GFMatParams.p4 << ";\n"
                  << "GFMatParams.a1= " << GFMatParams.a1 << ";\n"
                  << "GFMatParams.a2= " << GFMatParams.a2 << ";\n"
                  << "GFMatParams.a3= " << GFMatParams.a3 << ";\n"
                  << "GFMatParams.beta= " << GFMatParams.beta << ";\n"
                  << "GFMatParams.psi= " << GFMatParams.psi  << ";\n"
                  << "GFMatParams.J3Type= " << GFMatParams.J3Type  << ";\n"
                  << "GFMatParams.relaxationTime= " << GFMatParams.relaxationTime  << ";\n"
                  << "GFMS::solParam GFMSsolParams; \n"
                  << "GFMSsolParams.absToll= " << gpData.GFMSsolParams.absToll << ";\n"
                  << "GFMSsolParams.relToll= " << gpData.GFMSsolParams.relToll << ";\n"
                  << "GFMSsolParams.maxIter= " << gpData.GFMSsolParams.maxIter << ";\n"
                  << "GFMSsolParams.maxLevels= " << gpData.GFMSsolParams.maxLevels << ";\n"
                  << "const double J(" << std::setprecision(24) << J << ");\n"
                  << "const double delT(" << std::setprecision(24) << delT << ");\n"
                  << "const double invDelT(" << std::setprecision(24) << invDelT << ");\n"
                  << "SymMat3::SymMatrix3 sigma_0;\n"
                  << "sigma_0.set(0,0," << std::setprecision(24) << sigma_0.get(0,0) << ");\n"
                  << "sigma_0.set(1,1," << std::setprecision(24) << sigma_0.get(1,1) << ");\n"
                  << "sigma_0.set(2,2," << std::setprecision(24) << sigma_0.get(2,2) << ");\n"
                  << "sigma_0.set(1,2," << std::setprecision(24) << sigma_0.get(1,2) << ");\n"
                  << "sigma_0.set(0,2," << std::setprecision(24) << sigma_0.get(0,2) << ");\n"
                  << "sigma_0.set(0,1," << std::setprecision(24) << sigma_0.get(0,1) << ");\n"
                  << "SymMat3::SymMatrix3 sigma_tr;\n"
                  << "sigma_tr.set(0,0," << std::setprecision(24) << sigma_tr.get(0,0) << ");\n"
                  << "sigma_tr.set(1,1," << std::setprecision(24) << sigma_tr.get(1,1) << ");\n"
                  << "sigma_tr.set(2,2," << std::setprecision(24) << sigma_tr.get(2,2) << ");\n"
                  << "sigma_tr.set(0,1," << std::setprecision(24) << sigma_tr.get(0,1) << ");\n"
                  << "sigma_tr.set(0,2," << std::setprecision(24) << sigma_tr.get(0,2) << ");\n"
                  << "sigma_tr.set(1,2," << std::setprecision(24) << sigma_tr.get(1,2) << ");\n"
                  << "SymMat3::SymMatrix3 sigma_0qs;\n"
                  << "sigma_0qs.set(0,0," << std::setprecision(24) << sigma_0qs.get(0,0) << ");\n"
                  << "sigma_0qs.set(0,1," << std::setprecision(24) << sigma_0qs.get(0,1) << ");\n"
                  << "sigma_0qs.set(0,2," << std::setprecision(24) << sigma_0qs.get(0,2) << ");\n"
                  << "sigma_0qs.set(1,0," << std::setprecision(24) << sigma_0qs.get(1,0) << ");\n"
                  << "sigma_0qs.set(1,1," << std::setprecision(24) << sigma_0qs.get(1,1) << ");\n"
                  << "sigma_0qs.set(1,2," << std::setprecision(24) << sigma_0qs.get(1,2) << ");\n"
                  << "sigma_0qs.set(2,0," << std::setprecision(24) << sigma_0qs.get(2,0) << ");\n"
                  << "sigma_0qs.set(2,1," << std::setprecision(24) << sigma_0qs.get(2,1) << ");\n"
                  << "sigma_0qs.set(2,2," << std::setprecision(24) << sigma_0qs.get(2,2) << ");\n"
                  << "SymMat3::SymMatrix3 sigma_1,sigma_1qs,D;\n"
                  << "double epsv_0(" << std::setprecision(24) << epsv_0 <<");\n"
                  << "double gam_0("  << std::setprecision(24) << gam_0  <<");\n"
                  << "double epsv0_qs(" << std::setprecision(24) << epsv0_qs <<");\n"
                  << "double gam0_qs("  << std::setprecision(24) << gam0_qs << ");\n"
                  << "double epsv_1,  gam_1,  epsv1_qs, gam1_qs;\n"
                  << " ////// End C++ input //////// \n"
                  << "Time step output values:\n"
                  << "sigma_1= \n"
                  << sigma_1.get(0,0) << "\t"<< sigma_1.get(0,1) << "\t" << sigma_1.get(0,2) << "\n"
                  << sigma_1.get(1,0) << "\t"<< sigma_1.get(1,1) << "\t" << sigma_1.get(1,2) << "\n"
                  << sigma_1.get(2,0) << "\t"<< sigma_1.get(2,1) << "\t" << sigma_1.get(2,2) << "\n"
                  << "sigma_1qs= \n"
                  << sigma_1qs.get(0,0) << "\t"<< sigma_1qs.get(0,1) << "\t" << sigma_1qs.get(0,2) << "\n"
                  << sigma_1qs.get(1,0) << "\t"<< sigma_1qs.get(1,1) << "\t" << sigma_1qs.get(1,2) << "\n"
                  << sigma_1qs.get(2,0) << "\t"<< sigma_1qs.get(2,1) << "\t" << sigma_1qs.get(2,2) << "\n"
                  << "Dp= \n"
                  << Dp.get(0,0) << "\t"<< Dp.get(0,1) << "\t" << Dp.get(0,2) << "\n"
                  << Dp.get(1,0) << "\t"<< Dp.get(1,1) << "\t" << Dp.get(1,2) << "\n"
                  << Dp.get(2,0) << "\t"<< Dp.get(2,1) << "\t" << Dp.get(2,2) << "\n"
                  << "D= \n"
                  << D.get(0,0) << "\t"<< D.get(0,1) << "\t" << D.get(0,2) << "\n"
                  << D.get(1,0) << "\t"<< D.get(1,1) << "\t" << D.get(1,2) << "\n"
                  << D.get(2,0) << "\t"<< D.get(2,1) << "\t" << D.get(2,2) << "\n"
                  << "epsv_1= " << epsv_1 <<"\n"
                  << "gam_1 = " << gam_1  <<"\n"
                  << "epsv1_qs= " << epsv1_qs <<"\n"
                  << "gam1_qs = " << gam1_qs  <<"\n";
              throw std::runtime_error(msg.str());
            } else {
              delEnergy = 0.0;
            }
          }
          if(delEnergy > 0.0){
            *pGP_energy += delEnergy;
            *pdTdt_out  += delEnergy*invDelT/(rho_orig *state.specificHeat);
          }
        }
        break;
      }
      if ( *pGPJ >= gpData.JGP_loc ){
        *pLocalized |= 2;
      }
    }   // End if(flags.useGranularPlasticity)

    // End Granular Plasticity Calculation:

    // Compute the pressure ---------------------------------------

    // get the hydrostatic part of the stress
    double p;
    p = computePressure(eos, identity*cbrt(JEL), state, *pDamage);

    // Assign the new total stress -------------------------------
    IEl = bElBar_new.trace()/3.0;
    Matrix3x3 tauDev = (bElBar_new - identity*IEl)*state.shearModulus;
    *pIEl = IEl;

    // compute the total stress (volumetric + deviatoric)
    *pStress = (identity*p + tauDev)/J;
    
    // Compute the increment in strain energy due to the deviatoric
    double U = 0;
    double W = 0;
    U = eos->computeStrainEnergy(rho_orig,state.density);
    U *= state.bulkModulus / state.initialBulkModulus;
    W = 0.5*state.shearModulus*(bElBar_new.trace() - 3.0);

    // Compute temperature rise ----------------------------------

    double heatRate = 0.0;
    // Isentropic from EOS
    // Compute the elastic rate of volume change.
    double tr_d_el = velGrad.trace();
    if(flags.useGranularPlasticity) {
      // Correct for volume change associated with granular plasticity
      tr_d_el -= ((*pGPJ-pGPJ_old)/(*pGPJ)) * invDelT;
    }
    heatRate += (state.bulkModulus / state.initialBulkModulus) *
      eos->computeIsentropicTemperatureRate(pTemperature, 
                                            rho_orig,
                                            state.density,
                                            tr_d_el
                                            );

    *pdTdt_out += heatRate;
    *pEnergy = U+W;    // This is the strain energy density stored in the particle

    // Compute the local sound speed -------------------------------------------------
    if(flags.useDamage && pLocalized != 0) {
      *c_dil_out = 0;             // Localized particles should not contribute to the stable timestep calculation
    } else {
      if(flags.useGranularPlasticity) {
        // Make sure that I correct for the volumetric expansion, rho_cur gets corrupted and
        // represents the elastic density and does not include the effect of the GPJ.
        *c_dil_out = sqrt((state.bulkModulus + 4.*state.shearModulus/3.)/rho_orig);
      } else {
        *c_dil_out = sqrt((state.bulkModulus + 4.*state.shearModulus/3.)/rho_cur);
      }
    }

    // Compute artificial viscosity term --------------------
    if (flags.artificialViscosity) {
      double c_bulk = sqrt(state.bulkModulus/rho_cur);
      double Dkk = pDeformRate.trace();
      *p_q_out = artificialBulkViscosity(	Dkk, c_bulk,
                                        rho_cur, dx_ave,
                                        artificialViscosity);
      // Include the heating from artificial viscosity:
      if (flags.artificialViscosityHeating) {
        *pdTdt_out +=  J*Dkk*(-(*p_q_out))/(rho_orig*state.specificHeat);
      }
    } else {
      *p_q_out = 0.;
    }

#ifndef NDEBUG
    bool outputValid(true);
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
        if(!(std::fabs(pStress_qs->get(0,0))   < stressMax))  outputValid=false;
        if(!(std::fabs(pStress->get(0,0))      < stressMax))  outputValid=false;
      }
    }
    // Input/Output doubles:
    if( !(*pIEl-1.0 >= 0.0) || !(*pIEl<paramMax) ){
      if(*pIEl-1.0 > -1.0e-8) *pIEl=1.0;
      else outputValid=false;
    }
    if( !(*pPlasticStrain >= 0.0) || !(*pPlasticStrain<paramMax) )  outputValid=false;
    if( !(*pPlasticEnergy >= 0.0) || !(*pPlasticEnergy<paramMax) )  outputValid=false;
    if( !(*pDamage >= 0.0) || !(*pDamage<paramMax) )  outputValid=false;
    if( !(*pGPJ > 0.0) || !(*pGPJ<paramMax) )  outputValid=false;
    if( !(*pGP_strain >= 0.0) || !(*pGP_strain<paramMax) )  outputValid=false;
    if( !(*pGP_energy >= 0.0) || !(*pGP_energy<paramMax) )  outputValid=false;
    if( !(std::fabs(*pEnergy)<paramMax) )  outputValid=false;
    if( !(*damage_dt >= 0.0) || !(*damage_dt<paramMax) )  outputValid=false;
    if( !(std::fabs(*pepsV)<paramMax) )  outputValid=false;
    if( !(*pgam >= 0.0) || !(*pgam<paramMax) )  outputValid=false;
    if( !(std::fabs(*pepsV_qs)<paramMax) )  outputValid=false;
    if( !(*pgam_qs >= 0.0) || !(*pgam_qs<paramMax) )  outputValid=false;
    if( !(*p_q_out >= 0.0) || !(*p_q_out<paramMax) )  outputValid=false;
    if( !(std::fabs(*pdTdt_out)<paramMax) )  outputValid=false;
    if( !(*c_dil_out >= 0.0 && *c_dil_out<paramMax) )  outputValid=false;

    if(!outputValid){
      std::stringstream msg;
      msg << "Invalid output from advanceTimeSigmaL() detected:\n";
      msg << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
      msg << "Debugging information:\n"
          << "brittle_damage.criticalDamage: " << brittle_damage.criticalDamage << "\n"
          << "flags.useDamage: " << flags.useDamage << "\n"
          << "J_old = " << std::setw(20) << std::setprecision(12) << J_old << "\n"
          << "velGrad = " << velGrad << "\n"
          << "J = " << std::setw(20) << std::setprecision(12) << J << "\n"
          << "Jinc = " << std::setw(20) << std::setprecision(12) << Jinc << "\n"
          << "IEl_old-1 = "<< std::setw(20) << std::setprecision(12) << IEl_old-1 << "\n"
          << "IEl-1 = "<< std::setw(20) << std::setprecision(12) << IEl-1 << "\n"
          << "stress = " << *pStress << "\n"
          << "stress_old = " << pStress_old << "\n"
          << "Mean stress = "<< std::setw(20) << std::setprecision(12) << p/J << "\n"
          << "Old mean stress = "<< std::setw(20) << std::setprecision(12) << p_old << "\n"
          << "SigmaDev_old.norm() = " << std::setw(20) << std::setprecision(12) << (pStress_old-identity*pStress_old.trace()*onethird).norm() << "\n"
          << "SigmaDev.norm() = " << std::setw(20) << std::setprecision(12) << (*pStress-identity*pStress->trace()*onethird).norm() << "\n"
          << "stress_qs = " << *pStress_qs << "\n"
          << "pTemperature:\t" << std::setw(20) << std::setprecision(12) << pTemperature << "\n"
          << "rho_orig:\t" << std::setw(20) << std::setprecision(12) << rho_orig << "\n"
          << "pDamage_old:\t" << std::setw(20) << std::setprecision(12) << pDamage_old << "\n"
          << "delT:\t" << std::setw(20) << std::setprecision(12) << delT << "\n"
          << "pGP_strain:\t" << std::setw(20) << std::setprecision(12) << *pGP_strain << "\n"
          << "pPlasticStrain:\t" << std::setw(20) << std::setprecision(12) << *pPlasticStrain << "\n"
          << "pPlasticEnergy:\t" << std::setw(20) << std::setprecision(12) << *pPlasticEnergy << "\n"
          << "pDamage:\t" << std::setw(20) << std::setprecision(12) << *pDamage << "\n"
          << "pGPJ:\t" << std::setw(20) << std::setprecision(12) << *pGPJ << "\n"
          << "pGP_energy:\t" << std::setw(20) << std::setprecision(12) << *pGP_energy << "\n"
          << "damage_dt:\t" << std::setw(20) << std::setprecision(12) << *damage_dt << "\n"
          << "p_q:\t" << std::setw(20) << std::setprecision(12) << *p_q_out << "\n"
          << "pdTdt:\t" << std::setw(20) << std::setprecision(12) << *pdTdt_out << "\n"
          << "pepsV:\t" << std::setw(20) << std::setprecision(12) << *pepsV << "\n"
          << "pepsV_qs:\t" << std::setw(20) << std::setprecision(12) << *pepsV_qs << "\n"
          << "pgam:\t" << std::setw(20) << std::setprecision(12) << *pgam << "\n"
          << "pgam_qs:\t" << std::setw(20) << std::setprecision(12) << *pgam_qs << "\n";
      if( !(*pDamage >= 0.0) || !(*pDamage<paramMax) ){
        msg << "pDamage was invalid, initial crack parameters:\n"
            << "Flaw densities:\n";
        msg.setf(std::ios::dec|std::ios::scientific);
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pFlawNumber_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Initial flaw sizes:\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pflawSize_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Wing crack lengths (input):\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pWingLength_array)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n"
            << "Wing crack lengths (output):\n";
        for (unsigned int i = 0; i < nBins; i++) {
          msg << std::setw(10) << std::setprecision(6) << (*pWingLength_array_new)[i];
          if (i%5 == 4){
            msg << "\n";
          } else {
            msg << "\t";
          }
        }
        msg << "\n";
      }
    
      throw std::runtime_error(msg.str());
    }
#endif
  }

  void postAdvectionFixup(
                         // Data Structures
                         const Flags flags,
                         const ModelData initialData,
                         const flawDistributionData flawDistData,
                         const BrittleDamageData brittle_damage,
                         const granularPlasticityData gpData,
                         const ArtificialViscosity artificialViscosity,
                         const PortableMieGruneisenEOSTemperature *eos,
                         // Input/OutputMatrix:
                         Matrix3x3 *pStress, // Recalculate
                         Matrix3x3 *pStress_qs, // Unused
                         const double J,        // input
                         const double pTemperature, // input
                         const double rho_orig,     // input
                         // Input/Output double
                         double *pIEl, // update
                         double *pPlasticStrain, // Unused
                         double *pPlasticEnergy, 
                         double *pDamage,
                         double *pGPJ,
                         double *pGP_strain,
                         double *pGP_energy,
                         double *pEnergy,
                         double *damage_dt,
                         double *pepsV,
                         double *pgam,
                         double *pepsV_qs,
                         double *pgam_qs,
                         int *pLocalized,
                         // Input/Output std::vector
                         std::vector<double> *pWingLength_array,
                         std::vector<double> *pFlawNumber_array,
                         std::vector<double> *pflawSize_array
                         )
  {
    const std::vector<double>::size_type nBins = pWingLength_array->size();
    const Matrix3x3 identity(true);
    // Compute the current damage:
    if(flags.useDamage){
      double newDamage(0.0);
      for(unsigned int i=0; i<nBins; ++i){
        double binDensity = (*pFlawNumber_array)[i];
        if (binDensity<0){
          binDensity = 0;
          (*pFlawNumber_array)[i] = binDensity;
        }
        double s = (*pflawSize_array)[i];
        if (s<0){
          s=flawDistData.minFlawSize*1e-3;
          (*pflawSize_array)[i] = s;
        }
        double L = (*pWingLength_array)[i];
        if(L<0){
          L=0;
          (*pWingLength_array)[i] = L;
        }
        double damageIncrement = binDensity;
        if(brittle_damage.incInitialDamage) {
          damageIncrement *=  (L+s)*(L+s)*(L+s);
        } else {
          damageIncrement *=  L*L*L;
        }
        newDamage += damageIncrement;
      } // end loop over flaw bins
      *pDamage = newDamage;
    } else {
      *pDamage = 0.0;
    } // end if(flags.useDamage)

    // Update the hydrostatic stress:
    const CMData eosData = eos->getEOSData(); // needed for specific heat and reference temperature
    double JGP;
    double JEL;
    if(flags.useGranularPlasticity && (!flags.useDamage || *pDamage>=brittle_damage.criticalDamage) ) {
      JGP = *pGPJ;
      JEL = J/JGP;
      *pGP_strain = *pGP_strain >= 0.0 ? *pGP_strain : 0.0;
      *pGP_energy = *pGP_energy >= 0.0 ? *pGP_energy : 0.0;
      *pgam       = *pgam       >= 0.0 ? *pgam       : 0.0;
      *pgam_qs    = *pgam_qs    >= 0.0 ? *pgam_qs    : 0.0;
    } else {
      *pGPJ = 1.0;
      *pStress_qs *= 0.0;
      *pGP_strain = 0.0;
      *pGP_energy = 0.0;
      *pepsV = 0.0;
      *pgam  = 0.0;
      *pepsV_qs = 0.0;
      *pgam_qs  = 0.0;
      JEL = J;
      JGP = 1.0;
    }

    double rho_cur  = rho_orig/JEL;

    PState state;
    state.pressure = -pStress->trace()*onethird;
    state.temperature = pTemperature;
    state.initialTemperature = eosData.theta_0;
    state.density = rho_cur;
    state.initialDensity = rho_orig;
    state.initialVolume = 1.0;
    state.volume = state.initialVolume*J;
    state.specificHeat = eosData.C_v;
    state.energy = (state.temperature-state.initialTemperature)*state.specificHeat;
    state.initialBulkModulus = eos->computeBulkModulus(rho_orig, rho_cur);
    state.bulkModulus = state.initialBulkModulus;
    state.shearModulus = initialData.tauDev; // This is changed later if there is damage
    state.initialShearModulus = initialData.tauDev;
    state.bulkModulus  = calculateBulkPrefactor(*pDamage, state, JEL)*state.initialBulkModulus;
    state.shearModulus = calculateShearPrefactor(*pDamage, state)*state.initialShearModulus;
    
    double p = computePressure(eos, identity*cbrt(JEL), state, *pDamage);
    SymMat3::SymMatrix3 bElBarDev(false);
    double sigma_m_in = pStress->trace()/3.0;
    bElBarDev.set(0,0, pStress->get(0,0)-sigma_m_in);
    bElBarDev.set(1,1, pStress->get(1,1)-sigma_m_in);
    bElBarDev.set(2,2, pStress->get(2,2)-sigma_m_in);
    bElBarDev.set(1,2, 0.5*(pStress->get(2,1)+pStress->get(1,2)));
    bElBarDev.set(0,2, 0.5*(pStress->get(0,2)+pStress->get(2,0)));
    bElBarDev.set(0,1, 0.5*(pStress->get(0,1)+pStress->get(1,0)));
    bElBarDev *= (J/state.shearModulus);
    // SymMat3::SymMatrix3 er(false), ez(false), etheta(false);
    double J2 = 0.5*bElBarDev.normSquared();
    double J3 = bElBarDev.determinant();
    double r = sqrt(2.0*J2);
    // Compute IElBar from the constraint that det(bElBar)=1:
    double s3theta(0.0);
    if(J2 > 0.0){
      double A = 3.0/J2;
      s3theta = 0.5*J3*sqrt(A*A*A);
      s3theta = s3theta <= 1.0 ? s3theta : 1.0;
      s3theta = s3theta >= -1.0 ? s3theta : -1.0;
    }
    double c3thetaSq = 1.0-(s3theta*s3theta);
    double rcu = r*r*r;
    double C1 = sqrt(6.0)*(rcu)*s3theta;
    double C2 = sqrt(6.0)*sqrt(54.0 - (rcu * rcu)*(c3thetaSq) - 6.0*sqrt(6.0)*(rcu)*s3theta);
    double z = pow(( 18.0 - C1 + C2 ), 1.0/3.0) / (pow(2.0, 2.0/3.0) * pow(3, 1.0/6.0) )
      + ( (pow(3.0, 1.0/6.0)*(r*r) ) / pow(36.0 - 2*C1 + 2*C2, 1.0/3.0));
    *pIEl = z/sqrt(3.0);
    if(*pIEl < 1.0){
      *pIEl = 1.0;
      bElBarDev = SymMat3::SymMatrix3(false);
    }
    bElBarDev *= state.shearModulus;
    pStress->set(0,0, bElBarDev.get(0,0) + p);
    pStress->set(1,1, bElBarDev.get(1,1) + p);
    pStress->set(2,2, bElBarDev.get(2,2) + p);
    pStress->set(0,1, bElBarDev.get(0,1));
    pStress->set(1,0, bElBarDev.get(0,1));
    pStress->set(0,2, bElBarDev.get(0,2));
    pStress->set(2,0, bElBarDev.get(0,2));
    pStress->set(1,2, bElBarDev.get(1,2));
    pStress->set(2,1, bElBarDev.get(1,2));
    *pStress /= J;


    // Energy calculation:
    double U = 0;
    double W = 0;
    if(*pLocalized==0) {
      U = eos->computeStrainEnergy(rho_orig,state.density);
      U *= state.bulkModulus / state.initialBulkModulus;
      W = 1.5*state.shearModulus*(*pIEl - 1.0);
    }
    *pEnergy = U+W;

    // Check history parameters:
    *pPlasticStrain = *pPlasticStrain >= 0.0 ? *pPlasticStrain : 0.0;
    *pPlasticEnergy = *pPlasticEnergy >= 0.0 ? *pPlasticEnergy : 0.0;
    *pGP_strain     = *pGP_strain >= 0.0 ? *pGP_strain : 0.0;
    *pGP_energy     = *pGP_energy >= 0.0 ? *pGP_energy : 0.0;
    *pgam           = *pgam >= 0.0 ? *pgam : 0.0;
    *pgam_qs        = *pgam_qs >= 0.0 ? *pgam_qs : 0.0;

    // Recompute sigma_qs from pStress:
    if( (!flags.useGranularPlasticity) ||
        (flags.useDamage && *pDamage<brittle_damage.criticalDamage) ||
        !(gpData.timeConstant > 0.0) )
      {                      //  set pStress_qs to 0
        for (int i = 0; i<3; ++i){
          pStress_qs->set(i,i, 0.0);
          if(i<2){
            for (int j = i+1; j<3; ++j){
              pStress_qs->set(i,j, 0.0);
              pStress_qs->set(j,i, 0.0);
            }
          }
        }
      } else
      {
        GFMS::matParam GFMatParams(gpData.GFMSmatParams);
        GFMatParams.bulkMod  =
          state.bulkModulus/state.initialBulkModulus *
          eos->computeIsentropicBulkModulus(state.initialDensity, state.density, state.temperature);
        GFMatParams.shearMod = state.shearModulus;
        GFMatParams.bulkMod  /= J; // It appears that the bulk modulus is also for the Kirchoff stress
        GFMatParams.shearMod /= J; // The shear modulus in state is for the Kirchoff stress
        SymMat3::SymMatrix3 sigma_tr,sigma_1qs;
        double a,b;
        sigma_tr.set(0,0,pStress->get(0,0));
        sigma_tr.set(1,1,pStress->get(1,1));
        sigma_tr.set(2,2,pStress->get(2,2));
        sigma_tr.set(1,2,0.5*(pStress->get(1,2) + pStress->get(2,1)));
        sigma_tr.set(0,2,0.5*(pStress->get(0,2) + pStress->get(2,0)));
        sigma_tr.set(0,1,0.5*(pStress->get(1,0) + pStress->get(0,1)));
        GFMS::doReturnOuter(&sigma_tr, *pepsV, *pgam, &GFMatParams, &gpData.GFMSsolParams, &sigma_1qs, &a, &b);
        for (int i = 0; i<3; ++i){
          pStress_qs->set(i,i, sigma_1qs.get(i,i));
          if(i<2){
            for (int j = i+1; j<3; ++j){
              pStress_qs->set(i,j, sigma_1qs.get(i,j));
              pStress_qs->set(j,i, sigma_1qs.get(i,j));
            }
          }
        }
      } // done calculation of pStress_qs

  } // end postAdvectionFixup()

}	// End namespace PTR
