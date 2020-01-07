/*
 * ConvexPolyhedronGeometryPiece.cc
 *
 *  Created on: Jan 7, 2019
 *      Author: jbhooper
 */
#include <Core/GeometryPiece/ConvexPolyhedronGeometryPiece.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Malloc/Allocator.h>

#include <Core/Util/DOUT.hpp>

#include <sstream>

namespace Uintah {

  const std::string ConvexPolyhedronGeometryPiece::TYPE_NAME = "polyhedron";

  ConvexPolyhedronGeometryPiece::ConvexPolyhedronGeometryPiece( ProblemSpecP & polyhedron_ps)
  {
    bool debugConstruction = false;
    std::ostringstream debugConstructionOut;
    debugConstructionOut << "ConvexPolyhedronGeometryPiece:\t";

    name_ = "Unnamed " + TYPE_NAME + " from PS";

    std::string polyhedronLabel = "";
    polyhedron_ps->getAttribute("label",polyhedronLabel);

    polyhedron_ps->require("center", m_centerPoint);

    // Debug
    debugConstructionOut << "Polyhedron: << " << polyhedronLabel 
                         << " Read center: " << m_centerPoint << "\n";
    // Debug

    ProblemSpecP planes_ps = polyhedron_ps->findBlock("bounds");
    if (!planes_ps) {
      // Throw geometry exception because no bounding planes are defined
      std::ostringstream geomError;
      geomError << " INPUT GEOMETRY ERROR:  Cannot find bounding planes for polyhedron with center at: "
                << m_centerPoint << "\n";
      SCI_THROW(ProblemSetupException(geomError.str(), __FILE__,__LINE__));
    }

    for (ProblemSpecP halfSpace_ps = planes_ps->findBlock("half_space"); halfSpace_ps; halfSpace_ps = halfSpace_ps->findNextBlock("half_space")) {
      std::string anchor_type = "absolute";
      halfSpace_ps->getAttribute("type",anchor_type);

      if (!(anchor_type == "absolute" || anchor_type == "relative")) {
        std::ostringstream anchorError;
        anchorError << " INPUT GEOMETRY ERROR:  Unknown bounding plane relation for convex polyhedron with"
                    << " center at: " << m_centerPoint << "\n";
        SCI_THROW(ProblemSetupException(anchorError.str(), __FILE__, __LINE__));
      }
      
      Vector planePoint, planeNormal, centerOffset;
      // If specifying the anchor type as absolute, the plane is denoted by a point on the plane and a normal which
      //   contains the centeral point of the polyhedron on the side of the plane from which the normal emanates.
      if (anchor_type == "absolute") {
        halfSpace_ps->require("planePoint", planePoint);
        halfSpace_ps->require("normal", planeNormal);
      }
      else
      {
        halfSpace_ps->require("offset",centerOffset);
        planeNormal = -centerOffset;
        planePoint = m_centerPoint.asVector() + centerOffset;
      }
      PolyPlane currentPlane(planePoint.asPoint(),planeNormal);
      m_boundaryPlanes.push_back(currentPlane);
    }


    f_isBoundingBoxValid = false;
    // Debug
    debugConstructionOut << "\t\t\tNow finding vertices.\n";
    DOUT(debugConstruction,debugConstructionOut.str());
    // Debug
    findVertices();
    // Debug
    debugConstructionOut.clear();
    debugConstructionOut << "\t\t\tFound " << m_vertices.size() << " vertices.\n";
    DOUT(debugConstruction,debugConstructionOut.str());
    // Debug
    // Bulletproofing
    if (m_vertices.size() < 4) { // Need at least four vertices to enclose a volume.
      std::ostringstream vertexError;
      vertexError << "INPUT GEOMETRY ERROR:  "
                  << "Convex polyhedron \"" << polyhedronLabel
                  << "\" does not have enough valid vertices.\n";
      SCI_THROW(ProblemSetupException(vertexError.str(), __FILE__, __LINE__));
    }
    // Debug
    debugConstructionOut.clear();
    debugConstructionOut << "\t\t\tNow finding bounding box.\n";
    DOUT(debugConstruction,debugConstructionOut.str());
    // Debug
    if (m_vertices.size() == 0) std::cout << "Taking the bounding box of a zero size polygon.\n";
    constructBoundingBox();
    // Debug
    debugConstructionOut.clear();
    debugConstructionOut << "\t\t\tBounding Box: " << m_boundingBox << "\n";
    debugConstructionOut << "\t\t\tNow polyhedron theoretically constructed.\n";
    // Debug
  } // Construct from PS

  ConvexPolyhedronGeometryPiece::ConvexPolyhedronGeometryPiece( const Point                   & center ,
                                                                const std::vector<PolyPlane>  & bounds  )
                                                              :m_centerPoint(center),
                                                               m_boundaryPlanes(bounds)
  {
   findVertices();
   constructBoundingBox();
  }

  ConvexPolyhedronGeometryPiece::~ConvexPolyhedronGeometryPiece()
  {

  }

  void ConvexPolyhedronGeometryPiece::outputHelper(ProblemSpecP & ps) const
  {

  }

  GeometryPieceP ConvexPolyhedronGeometryPiece::clone() const
  {
    return scinew ConvexPolyhedronGeometryPiece(*this);
  }

//  void ConvexPolyhedronGeometryPiece::findVerticesNew() {
//    size_t numPlanes = m_boundaryPlanes.size();
//  }

  void ConvexPolyhedronGeometryPiece::findVerticesNew() {
    size_t numPlanes = m_boundaryPlanes.size();

    for(size_t first = 0; first < numPlanes -2; ++first) {
      const PolyPlane & plane1 = m_boundaryPlanes[first];
      Vector norm1;
      double distance1;
      std::tie(norm1, distance1) = plane1.getDirectionAndDistance();

      for(size_t second = first + 1; second < numPlanes -1; ++second) {
        const PolyPlane & plane2 = m_boundaryPlanes[second];
        Vector norm2;
        double distance2;
        std::tie(norm2, distance2) = plane2.getDirectionAndDistance();
        double n1Dotn2 = Dot(norm1,norm2);
        if (std::abs(1.0-n1Dotn2) > PolyPlane::NEAR_ZERO) {
          Vector n1Crossn2 = Cross(norm1,norm2);
          for (size_t third = second + 1; third < numPlanes; ++third) {
            const PolyPlane & plane3 = m_boundaryPlanes[third];
            Vector norm3;
            double distance3;
            std::tie(norm3, distance3) = plane3.getDirectionAndDistance();
            Vector n2Crossn3 = Cross(norm2,norm3);
            double denom = Dot(norm1, n2Crossn3);
            if (std::abs(denom) > PolyPlane::NEAR_ZERO) {
              Vector n3Crossn1 = Cross(norm3,norm1);
              Vector vertex = (-distance1*n2Crossn3 - distance2*n3Crossn1 - distance3*n1Crossn2)/denom;
              // Check to see if it's actually a vertex
              bool isInterior = true;
              size_t planeIndex = 0;
              while (isInterior && planeIndex < numPlanes) {
                const PolyPlane& checkPlane = m_boundaryPlanes[planeIndex];
                isInterior = isInterior && checkPlane.pointInterior(vertex.asPoint());
                ++planeIndex;
              }
              if (isInterior) {
                m_vertices.push_back(vertex.asPoint());
              }
            }
          }
        }

      }

    }
  }

  void ConvexPolyhedronGeometryPiece::findVertices() {
    //  Debug spew
    bool debugVertices = false;
    bool debugInside = false;
    std::ostringstream debugVerticesOut;
    std::stringstream debugInsideOut;
    debugVerticesOut.clear();
    debugInsideOut.clear();

    Vector n1, n2;
    Point  p1, p2;
    //  Debug spew
    debugVerticesOut << "\nCenter: " << m_centerPoint;

    size_t numPlanes = m_boundaryPlanes.size();
    for (size_t first = 0; first < numPlanes - 2; ++first) {
      const PolyPlane & plane1 = m_boundaryPlanes[first];

      // --- Debug spew
      debugVerticesOut << "\nChecking plane " << first;
      std::tie(n1,p1) = plane1.getDirectionAndOffset();
      debugVerticesOut << " n:  " << n1 << " l = " << n1.length() << ") at point " << p1 << "\n";
      // --- Debug spew

      for (size_t second = first + 1; second < numPlanes - 1; ++second) {
        const   PolyPlane & plane2 = m_boundaryPlanes[second];
        bool    intersect12;
        Vector  edge12Direction;
        Point   pointOnEdge12;

        // --- Debug spew
        debugVerticesOut << "  Against plane " << second;
        std::tie(n2,p2) = plane2.getDirectionAndOffset();
        debugVerticesOut << " (n:  " << n2 << " l = " << n2.length() << ") at point " << p2;
        // --- Debug spew

        std::tie(intersect12,edge12Direction,pointOnEdge12) = plane1.intersectWithPlane(plane2);
        if (intersect12) { // Plane 1 and 2 intersect, so check 3

          // --- Debug spew
          debugVerticesOut << "\n    " << first << " , " << second << " intersect. Dir: " << edge12Direction << " point: " << pointOnEdge12;
          // --- Debug spew


          for (size_t third = second + 1; third < numPlanes; ++third) {
            const PolyPlane & plane3 = m_boundaryPlanes[third];
            bool  intersectionExists;
            Point resultPoint;
            std::tie(intersectionExists,resultPoint)
                  = plane3.intersectWithLine(edge12Direction,pointOnEdge12);

//            bool pointIsInside = false;
            // --- Debug spew
            if (intersectionExists) {
              debugInsideOut << "\n      Line " << first << "-" << second
                             << " crosses plane " << third << " @ "
                             << resultPoint;

              if (inside(resultPoint) ) {
                debugInsideOut << " Point IS INTERIOR.";
              }
            } // Temporary loop to output for debugging.
            // --- Debug spew

//            if (intersectionExists) {
//              bool pointInterior = inside(resultPoint);
//              if (pointInterior) {
//                size_t  numVertices    = m_vertices.size();
//                size_t  currentVertex  = 0;
//                bool    isUnique  = true;
//              }
//            }

            if ((intersectionExists) && inside(resultPoint)) {

                size_t numVertices = m_vertices.size();
                size_t currentVertex = 0;
                bool isUnique = true;
                // --- Debug spew
                debugVerticesOut << "Found vertex: " << resultPoint << "\n";
                // --- Debug spew

                while (isUnique && currentVertex < numVertices) {
                  Vector difference = resultPoint - m_vertices[currentVertex];

                  // --- Debug spew
                  debugVerticesOut << "Comparing to vertex: " << currentVertex << m_vertices[currentVertex];
                  debugVerticesOut << " difference: " << difference << " of length " << difference.length() << "\n";
                  // --- Debug spew

                  if (difference.length() < 1.0e-10) {
                    isUnique = false;
                  }
                  ++currentVertex;
                }

                if (isUnique) m_vertices.push_back(resultPoint);
                // point exists, is new, and is interior to all planes
                // FIXME:  At some point we should put edge calculation in
                //   here, since a polygon edge occurs between any two 3-plane
                //   intersection points that share 2 common indices, and we
                //   have all that information here already.
            }
          } // Find intersection of edge 12 with plane 3

          // --- Debug spew
          if (debugInside) {
            if (debugVertices) {
              debugVerticesOut << debugInsideOut.str();
            } else {
              DOUT(true,debugInsideOut.str());
            }
            debugInsideOut.clear();
          }
          // --- Debug spew

        } // Planes 1 and 2 intersect in a unique line
      } // Loop over Plane 2

      // --- Debug spew
      DOUT(debugVertices,debugVerticesOut.str());
      debugVerticesOut.clear();
      // --- Debug spew

    } // Loop over first plane
  } // Find vertices

  bool ConvexPolyhedronGeometryPiece::inside(const Point & pointIn ,
                                             const bool    dummy    ) const
  {
    // Shift input point to be relative to the center of the polyhedron.
    if (f_isBoundingBoxValid && !m_boundingBox.contains(pointIn)) {
      // std::cout << "Bouncing due to bounding box.\n";
      return false;
    }

    bool isInside = true;
    size_t checkedPlane = 0;
    size_t numPlanes = m_boundaryPlanes.size();

//    std::cerr << "\n\t\t\tPoint IS inside planes ";
    while (isInside && (checkedPlane < numPlanes)) {
      isInside = m_boundaryPlanes[checkedPlane].isInside(pointIn);
      if (isInside) {
//        std::cerr << " " << checkedPlane;
        ++checkedPlane;
      }
    }
    if (checkedPlane < numPlanes) {
//      std::cerr << "\n\t\t\tIt is NOT inside plane " << checkedPlane << "\n";
      Vector normal;
      Point  anchor;
      std::tie(normal,anchor) = m_boundaryPlanes[checkedPlane].getDirectionAndOffset();
//      std::cerr << " Test point: " << pointIn << " Excluded plane normal: " << normal << " and anchor " << anchor << "\n";
    } else {
//      std::cerr << "\n";
    }
    return (isInside);
  }

  void ConvexPolyhedronGeometryPiece::constructBoundingBox() {
    Vector boxLow   = m_vertices[0].asVector();
    Vector boxHigh  = boxLow;
    for (size_t index=1; index < m_vertices.size(); ++index) {
      boxLow = Min(boxLow,m_vertices[index].asVector());
      boxHigh = Max(boxHigh,m_vertices[index].asVector());
    }

    m_boundingBox = Box((boxLow).asPoint(),(boxHigh).asPoint());

    f_isBoundingBoxValid = true;

  }

  void ConvexPolyhedronGeometryPiece::printVertices() const {
    size_t numVertices = m_vertices.size();
    for (size_t vertex = 0; vertex < numVertices; ++vertex) {
      std::cout << "Vertex " << vertex << ": " << m_vertices[vertex] << "\n";
    }
  }

  Box ConvexPolyhedronGeometryPiece::getBoundingBox() const {
    return m_boundingBox;
  }

}



