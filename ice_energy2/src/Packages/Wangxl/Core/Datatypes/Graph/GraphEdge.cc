#include "NodeData.h"
#include "EdgeData.h"

#include <iostream>

namespace Wangxl {

using std::ostream;

GraphEdge::GraphEdge() : data(0) {}

ostream& operator<< (ostream& os, const GraphEdge& edge) {
    return os << edge.source() << "-->" << edge.target();
}

GraphNode GraphEdge::source() const
{
    return data->nodes[0];
}

GraphNode GraphEdge::target() const
{
    return data->nodes[1];
}

int GraphEdge::id()
{
  return data->id;
}
bool GraphEdge::isHidden()
{
  return data->hidden;
}

bool operator==(GraphEdge edge0, GraphEdge edge1)
{
    return edge0.data == edge1.data;
}

bool operator!=(GraphEdge edge0, GraphEdge edge1)
{
    return edge0.data != edge1.data;
}

bool operator<(GraphEdge edge0, GraphEdge edge1)
{
    return edge0.data < edge1.data;
}

}


