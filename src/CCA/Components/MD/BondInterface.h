/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_BONDINTERFACE_H
#define UINTAH_MD_BONDINTERFACE_H

namespace Uintah {

  /**
   *  @class Bond
   *  @ingroup MD
   *  @author Justin Hooper and Alan Humphrey
   *  @date   September, 2013
   *
   *  @brief
   *
   */
  class Bond {

    public:

      Bond();

      virtual ~Bond() = 0;

      double getDistance();
      Generic3Vector getOffset();

      virtual double getEnergy() = 0;                      // override for specific bond implementations
      virtual Generic3Vector getForce(unsigned int) = 0;   // override for specific bond implementations
      virtual Generic3Vector getVirial(unsigned int) = 0;  // override for specific bond implementations

      inline unsigned int getFirstAtom() const
      {
        return firstAtomID;
      }

      inline unsigned int getSecondAtom() const
      {
        return secondAtomID;
      }


    private:

      // reference to components which hold related, but not bond specific distance
      unsigned int firstAtomID;
      unsigned int secondAtomID;

      // store bond specific information
      bool offsetValid;
      Generic3Vector bondOffset;
      double distance;

      unsigned int moleculeID;
      unsigned int bondID;

      bool energyValid;
      double bondEnergy;

      bool forceValid;
      Generic3Vector forceFirst;
      Generic3Vector forceSecond;

      void calculateOffset();
      virtual void calculateEnergy() = 0;  // override for specific bond type
      virtual void calculateForce() = 0;   //override for specific bond type

      inline void invalidate() const
      {
        forceValid = false;
        energyValid = false;
      }

  };

}  // End namespace Uintah

#endif // UINTAH_MD_BONDINTERFACE_H
