/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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



#include <Core/Grid/Variables/NCVariable.h>
#include <CCA/Components/Models/HEChem/Common.h>
#include <Core/Exceptions/InvalidValue.h>

#include <iostream>
namespace Uintah {

  //______________________________________________________________________
  double computeSurfaceArea(Vector &normDensityGrad, Vector &dx){


    if( normDensityGrad.length() == 0 ){  // possibly inside of a solid
      return 0.0;
    }

    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();

    double normDensityGradX = fabs(normDensityGrad.x());
    double normDensityGradY = fabs(normDensityGrad.y());
    double normDensityGradZ = fabs(normDensityGrad.z());

    double max = normDensityGradX;
    if(normDensityGradY > max){
      max = normDensityGradY;
    }
    if(normDensityGradZ > max){
      max = normDensityGradZ;
    }

    double coeff = pow(1.0/max, 1.0/3.0);
    //coeff = 1.0;

    double TmpX = delX * normDensityGradX;
    double TmpY = delY * normDensityGradY;
    double TmpZ = delZ * normDensityGradZ;

    double surfArea  = delX*delY*delZ / ( (TmpX + TmpY + TmpZ) * coeff);

    //__________________________________
    //    Bulletproofing
    if ( std::isnan(surfArea) || std::isinf(surfArea) ){
      std::ostringstream warn;
      warn << "Models::computeSurfaceArea: surfArea (" << surfArea << ") is either a Nan or Inf."
           <<  "\n normDensityGrad: " << normDensityGrad << " coeff: " << coeff <<  " max: "<< max;
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
    return surfArea;
  }

  //______________________________________________________________________
  //    This returns the normalized gradient of the density
  Vector computeDensityGradientVector(IntVector *nodeIdx,
                                      constNCVariable<double> &NCsolidMass,
                                      constNCVariable<double> &NC_CCweight,
                                      Vector &dx){
    double gradRhoX = 0.25 * (           // xminus
                              (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]] +
                               NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]] +
                               NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]] +
                               NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                              -          // xplus
                              (NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]] +
                               NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]] +
                               NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]] +
                               NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                              )/dx.x();

    double gradRhoY = 0.25 * (           // yminus
                              (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]] +
                               NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]] +
                               NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]] +
                               NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                              -          // yplus
                              (NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]] +
                               NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]] +
                               NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]] +
                               NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                             )/dx.y();

    double gradRhoZ = 0.25 * (           // zminus
                              (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]] +
                               NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]] +
                               NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]] +
                               NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                              -          // zplus
                              (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]] +
                               NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]] +
                               NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]] +
                               NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                              )/dx.z();

    // add d_SMALL_NUM to prevent inf or NaNs that will occur inside a solid matl
    double magDensityGrad = sqrt(gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ ) + d_SMALL_NUM;

    Vector normGradRho = Vector(gradRhoX/magDensityGrad, gradRhoY/magDensityGrad, gradRhoZ/magDensityGrad);

    //__________________________________
    //  Bulletproofing
    if (normGradRho.Vector::isnan() || normGradRho.Vector::isinf() ){
      std::ostringstream warn;
      warn << "Models::computeDensityGradientVector:  norm density gradient" << nodeIdx[0] << " is either a Nan or Inf."
           <<  "\n magDensityGrad: " << magDensityGrad << " normGradRho: " << normGradRho
           <<  "\n gradRhoX: " << gradRhoX << " gradRhoY: " << gradRhoY << " gradRhoZ: " << gradRhoZ;
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }

    return normGradRho;
  }
}
