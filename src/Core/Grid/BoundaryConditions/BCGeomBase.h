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

#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/BaseIterator.h>
#include <Core/Util/DebugStream.h>

#include <vector>
#include <typeinfo>
#include <iterator>


namespace Uintah {

  /*!

  \class BCGeomBase

  \ brief Base class for the boundary condition geometry types.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */

   

  class BCGeomBase {
  public:

    /*  \struct ParticleBoundarySpec
     *  \author Tony Saad
     *  \date   August 2014
     *  \brief  Struct that holds information about particle boundary conditions
     */
    struct ParticleBndSpec {
      
      /*
       *  \brief Particle boundary type wall or inlet
       */
      enum ParticleBndTypeEnum {
        WALL, INLET, NOTSET
      };

      /*
       *  \brief Wall boundary type: Elastic, Inelastic, etc...
       */
      enum ParticleWallTypeEnum {
        ELASTIC, INELASTIC, PARTIALLYELASTIC
      };
      
      // Default constructor
      ParticleBndSpec()
      {
        ParticleBndSpec(ParticleBndSpec::NOTSET, ParticleBndSpec::ELASTIC, 0.0, 0.0);
      }

      // Constructor
      ParticleBndSpec(ParticleBndTypeEnum bndTypet, ParticleWallTypeEnum wallTypet, const double restitution, const double pPerSec)
      {
        bndType         = bndTypet;
        wallType        = wallTypet;
        restitutionCoef = restitution;
        particlesPerSec = pPerSec;
      }
      
      // Copy constructor
      ParticleBndSpec(const ParticleBndSpec& rhs)
      {
        bndType         = rhs.bndType;
        wallType        = rhs.wallType;
        restitutionCoef = rhs.restitutionCoef;
        particlesPerSec = rhs.particlesPerSec;
      }
      
      /*
       *  \brief Checks whether a particle boundary condition has been specified on this BCGeometry
       */
      bool hasParticleBC() const
      {
        return( bndType != ParticleBndSpec::NOTSET );
      }
      
      ParticleBndTypeEnum bndType;
      ParticleWallTypeEnum wallType;
      double restitutionCoef;
      double particlesPerSec;
    };

    /// Constructor
    BCGeomBase();

    /// Copy constructor
    BCGeomBase(const BCGeomBase& rhs);

    /// Assignment operator
    BCGeomBase& operator=(const BCGeomBase& rhs);

    /// Destructor
    virtual ~BCGeomBase();    

    /// Equality test
    virtual bool operator==(const BCGeomBase&) const = 0;

    /// Make a clone
    virtual BCGeomBase* clone() = 0;

    /// Get the boundary condition data
    virtual void getBCData(BCData& bc) const = 0;

    /// For old boundary conditions
    virtual void addBCData(BCData& bc)  = 0;

    /// For old boundary conditions
    virtual void addBC(BoundCondBase* bc)  = 0;

    /// Allows a component to add a boundary condition, which already has an iterator.
    //  This method is exactly the same as addBC, except it applies to two additional geometries,
    //  differences and unions.  Since these geometries are not "atomic" and can consist of other
    //  sub-geometries the function addBC intentionally will not add boundary conditions to these objects. 
    //  Hence, this function forces the boundary conditions to be set regardless of the inherited object's
    //  special properties. 
    virtual void sudoAddBC(BoundCondBase* bc)  = 0;

    void getCellFaceIterator(Iterator& b_ptr);

    void getNodeFaceIterator(Iterator& b_ptr);

    bool hasIterator(){return (d_cells.size() > 0);}

    /// Determine if a point is inside the geometry where the boundary
    /// condition is applied.
    virtual bool inside(const Point& p) const = 0;

    /// Print out the type of boundary condition -- debugging
    virtual void print() = 0;

    /// Determine the cell centered boundary and node centered boundary
    /// iterators.
    virtual void determineIteratorLimits( const Patch::FaceType      face, 
                                          const Patch              * patch, 
                                          const std::vector<Point> & test_pts );
    
    /*
     \Author  Tony Saad
     \Date    September 2014
     \brif    Determine the iterator associated with this geometry when it is used as an interior boundary.
     */
    virtual void determineInteriorBndIteratorLimits( const Patch::FaceType      face,
                                         const Patch              * patch );

    /// Print out the iterators for the boundary.
    void printLimits() const;

    /// Get the name for this boundary specification
    std::string getBCName(){ return d_bcname; }
    void setBCName( std::string bcname ){ d_bcname = bcname; }

    /// Get the type for this boundary specification (type is usually associated with a user-friendly
    /// boundary type such as Wall, Inlet, Outflow...
    std::string getBndType(){ return d_bndtype; }
    void setBndType( std::string bndType ){ d_bndtype = bndType; }
    
    // Particle-related functionality
    ParticleBndSpec getParticleBndSpec(){return d_particleBndSpec;}
    void setParticleBndSpec(const ParticleBndSpec pBndSpec){ d_particleBndSpec = pBndSpec; }
    bool hasParticleBC(){ return d_particleBndSpec.hasParticleBC(); }

    double surfaceArea(){ return d_surfaceArea; }
    Point getOrigin() { return d_origin;}

  protected:
    Iterator          d_cells;
    Iterator          d_nodes;
    std::string       d_bcname;
    std::string       d_bndtype;
    ParticleBndSpec   d_particleBndSpec;
    double            d_surfaceArea;
    Point             d_origin;

    // Can only be one and copying is possible.
    static DebugStream BC_dbg;
  };

  template<class T> 
  class cmp_type {
    public:
    bool operator()(const BCGeomBase* p) {
      return (typeid(T) == typeid(*p));
    }
  };

  template<class T>
  class not_type {
    public:
    bool operator()(const BCGeomBase* p) {
      return (typeid(T) != typeid(*p));
    }
  };

  template<typename T>
  class delete_object {
  public:
    void operator() (T* ptr) {
      delete ptr;
    }
  };

} // End namespace Uintah

#endif
