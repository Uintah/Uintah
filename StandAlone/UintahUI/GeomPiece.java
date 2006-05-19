//**************************************************************************
// Program : GeomPiece.java
// Purpose : Define the Geometry Piece objects
// Author  : Biswajit Banerjee
// Date    : 05/14/2006
// Mods    :
//**************************************************************************

import java.io.PrintWriter;

//**************************************************************************
// Class   : GeomPiece
// Purpose : Creates a GeomPiece 
//**************************************************************************
public abstract class GeomPiece extends Object {

  // Common data
  protected String d_name;
  
  // Common Methods
  public abstract String getName();
  public abstract void writeUintah(PrintWriter pw, String tab);
  public abstract void print();


}

