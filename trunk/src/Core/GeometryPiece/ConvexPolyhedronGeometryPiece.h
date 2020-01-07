#ifndef __CONVEX_POLYHEDRON_H__
#define __CONVEX_POLYHEDRON_H__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/PolyPlane.h>
#include <Core/Grid/Box.h>
#include <Core/Util/DebugStream.h>

#include <Core/GeometryPiece/GeometryPiece.h>
namespace Uintah {

  /*******************************
CLASS
  ConvexPolyhedronGeometryPiece

  Creates the corresponding polyhedron from a center location and a list of
  vertex vectors which represent the planes bounding the polyhedron.  The
  list is assumed to represent a complex polyhedron when read; checking for
  convexity is not enacted.

GENERAL INFORMATION
  ConvexPolyhedronGeometryPiece.h

  Justin B. Hooper
  Dept. of Materials Science and Engineering
  University of Utah

KEYWORDS
  ConvexPolyhedronGeometryPiece BoundingBox inside

DESCRIPTION
  Creates a convex polyhedron from an XML input file description containing
  the center of the polyhedron and a list of vectors representing the offset
  of the center to the bounding planes of the polyhedron.

  The input format is:
    <polyhedron>
      <center> [ 0.,   0.,   0.] </center>
      <bounding_planes type = "relative">
        <anchor>  [ 0.5,  0.5,  0.5] </anchor>
        <anchor>  [-0.5,  0.5,  0.5] </anchor>
        <anchor>  [ 0.5, -0.5,  0.5] </anchor>
        <anchor>  [ 0.5,  0.5, -0.5] </anchor>
        <anchor>  [-0.5, -0.5,  0.5] </anchor>
        <anchor>  [-0.5,  0.5, -0.5] </anchor>
        <anchor>  [ 0.5, -0.5, -0.5] </anchor>
        <anchor>  [-0.5, -0.5, -0.5] </anchor>
      </bounding_planes>
    </polyhedron>

    The above input would generate an octahedral convex polyhedron.
*******/

    class ConvexPolyhedronGeometryPiece : public GeometryPiece {

      public:
        // Constructor that takes a ProblemSpecP argument.
        ConvexPolyhedronGeometryPiece(ProblemSpecP&);

        // Constructor that takes a center point and a list of plane locations.
        //   The plane normal is always pointed toward the center point.
        ConvexPolyhedronGeometryPiece(const Point               & p1    ,
                                      const std::vector<PolyPlane>  & bounds  );

        // Destructor
        virtual ~ConvexPolyhedronGeometryPiece();

        static const std::string TYPE_NAME;
        virtual      std::string getType() const { return TYPE_NAME; }

        // Make a clone
        virtual GeometryPieceP clone() const;

        // Determine whether a point is inside the box.
        virtual bool inside(const Point &p, const bool dummy = false) const;

        // Return the bounding box surrounding the polyhedrun (axis aligned)
        virtual Box getBoundingBox() const;

      private:
        virtual void outputHelper( ProblemSpecP & ps) const;

        void printVertices() const;
        void constructBoundingBox();
        void findEdges();
        void findVertices();
        void findVerticesNew();

        Point                   m_centerPoint;
        std::vector<PolyPlane>  m_boundaryPlanes;
        std::vector<Vector>     m_edges;
        std::vector<Point>      m_vertices;
        Box                     m_boundingBox;
        bool                    f_isBoundingBoxValid;
    };
}  // End namespace Uintah

#endif // __CONVEX_POLYHEDRON_H__
