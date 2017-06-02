/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#ifndef UINTAH_HOMEBREW_VarLabel_H
#define UINTAH_HOMEBREW_VarLabel_H

#include <string>
#include <iosfwd>
#include <Core/Util/RefCounted.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

  class TypeDescription;
  class Patch;

    /**************************************
      
      CLASS
        VarLabel
      
        Short Description...
      
      GENERAL INFORMATION
      
        VarLabel.h
      
        Steven G. Parker
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
      KEYWORDS
        VarLabel
      
      DESCRIPTION
        Long description...
      
      WARNING
      
      ****************************************/
    
  class VarLabel : public RefCounted {
  public:
    enum VarType {
      Normal,
      PositionVariable
    };

    // Ensure the uniqueness of VarLabel names (same name, same object).
    static VarLabel* create( const std::string       & name,
                             const TypeDescription   * type_description,
                             const Uintah::IntVector & boundaryLayer = Uintah::IntVector(0,0,0),
                                   VarType             vartype = Normal );

    static bool destroy( const VarLabel * label );

    inline const std::string & getName() const { return d_name;  }
    std::string getFullName( int matlIndex, const Patch * patch ) const;

    bool isPositionVariable() const {
      return d_vartype == PositionVariable;
    }

    const TypeDescription * typeDescription() const {
      return d_td;
    }

    Uintah::IntVector getBoundaryLayer() const {
      return d_boundaryLayer;
    }

    void allowMultipleComputes();

    bool allowsMultipleComputes() const
      { return d_allowMultipleComputes; }
    
    static VarLabel* find(const std::string& name);
    static VarLabel* particlePositionLabel();
    static void setParticlePositionName(const std::string& pPosName){d_particlePositionName = pPosName;}
    static std::string& getParticlePositionName(){return d_particlePositionName;}
    
    class Compare {
    public:
      inline bool operator()(const VarLabel* v1, const VarLabel* v2) const {
        // because of uniqueness, we can use pointer comparisons
        //return v1 < v2;
        // No we cannot, because we need the order to be the same on different processes
        if(v1 == v2)
          return false;
        return v1->getName() < v2->getName();
      }
    private:
    };
    
    bool equals(const VarLabel* v2) const {
      // because of uniqueness, we can use pointer comparisons
      return this == v2;
      /* old way
         if(this == v2)
         return true;
         return getName() == v2->getName();
      */
    }

    void setCompressionMode(std::string compressionMode)
      { d_compressionMode = compressionMode; }
    
    const std::string& getCompressionMode() const {
      return (d_compressionMode == "default") ?
        d_defaultCompressionMode : d_compressionMode;
    }
     
    static void setDefaultCompressionMode( const std::string & compressionMode ) {
      d_defaultCompressionMode = compressionMode;
    }

    static void printAll(); // for debugging
     
    std::string d_name;

    friend std::ostream & operator<<( std::ostream & out, const Uintah::VarLabel & vl );

  private:
    // You must use VarLabel::create.
    VarLabel(const std::string&, const TypeDescription*,
             const Uintah::IntVector& boundaryLayer, VarType vartype);
    // You must use destroy.
    ~VarLabel();   
     
    const   TypeDescription   * d_td;
            Uintah::IntVector   d_boundaryLayer;
            VarType             d_vartype;
    mutable std::string         d_compressionMode;
    static  std::string         d_defaultCompressionMode;
    static  std::string         d_particlePositionName;
    // Allow a variable of this label to be computed multiple
    // times in a TaskGraph without complaining.
    bool                        d_allowMultipleComputes;
     
    VarLabel( const VarLabel & );
    VarLabel & operator=( const VarLabel & );
  };
} // End namespace Uintah

#endif
