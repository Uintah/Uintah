#include <CCA/Components/Builder/BridgeConnection.h>

void BridgeConnection::drawShape(QPainter& p)
{
  QPointArray par(6);
  for(int i=0;i<6;i++)	
    par[i]=(points()[i]+points()[11-i])/2;
  QPen pen(color,4);
  pen.setStyle(DotLine);
  p.setPen(pen);
  p.setBrush(blue);
  p.drawPolyline(par);
}

std::string BridgeConnection::getConnectionType()
{
  return "BridgeConnection";
}


