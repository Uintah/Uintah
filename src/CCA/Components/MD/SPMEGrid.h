/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_MD_SPME_GRID_H
#define UINTAH_MD_SPME_GRID_H

#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/MD/MapPoint.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <complex>

namespace Uintah {

class Matrix;

typedef SimpleGrid<std::complex<double > > cdGrid;

template<class T> class SPMEGrid {

    public:
      // Functions to take care of mapping from point to grid and back again
      SPMEGrid& mapChargeToGrid(const SimpleGrid<std::vector<MapPoint<T> > > gridMap, const ParticleSubset& globalParticleList);

      SPMEGrid& mapForceFromGrid(const SimpleGrid<std::vector<MapPoint<T> > > gridMap, ParticleSubset& globalParticleList) const;

      SPMEGrid& multiplyInPlace(const cdGrid& GridIn) { Q *= GridIn; FieldValid=false; return *this; }; // Multiply Q * GridIn

      double calculateEnergyAndStress(const vector<double>& M1,
                                    const vector<double>& M2,
                                    const vector<double>& M3,
                                    Matrix StressTensor,
                                    const cdGrid& StressPrefactor, const cdGrid& ThetaRecip); // Returns local energy of charge grid;

      SPMEGrid& CalculateField();  // if (FieldValid == false) { calculate field grid; set FieldValid == true; return *this; }

      SPMEGrid& inPlaceFFT_RealToFourier(/*Data for FFT routine goes here */); // Transforms Q from real to fourier space;
      SPMEGrid& inPlaceFFT_FourierToReal(/*Data for FFT routine goes here */); // Transforms Q' from fourier to real space;

    private:
        IntVector          Extent; // See below
        IntVector          Offset; // See below
        cdGrid             Q;
        SimpleGrid<Vector> Field;
        bool FieldValid; // =false;
        // Alan - Must deal with ghosts here in the same way as in SimpleGrid;  Alternatively we can get rid of extent/offset for SPME_Grid
        //        entirely and pass through extent/offset calls to the underlying SimpleGrid data type.  If we do this, then we need to ensure
        //        registration of Q and Field extent/offset on construction.  This probably isn't a bad idea, since ideally we will construct
        //        an SPME_Grid instance with either Q set and Field zero'ed out, or both Q and Field zero'd out, so we can just construct
        //        Field as a subprocess of constructing the SPME_Grid object by extracting the relevant data from the Q grid and passing it to
        //        Field's constructor.  This is just more complex construction than I'm used to, so I'm not going to even take a stab at that here.


};

}  // End namespace Uintah

#endif
