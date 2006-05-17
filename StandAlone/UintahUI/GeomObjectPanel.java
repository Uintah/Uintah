//**************************************************************************
// Class   : GeomObjectPanel
// Purpose : Panel to enter data on geometry objects
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.util.Random;
import java.util.Vector;
import java.io.*;
import javax.swing.*;

public class GeomObjectPanel extends JPanel 
                             implements ActionListener {

  // Data
  private boolean d_usePartDist = false;
  private ParticleList d_partList = null;
  private Vector d_mpmMat = null;
  private Vector d_iceMat = null;
  private Vector d_geomObj = null;
  private Vector d_geomPiece = null;
  private CreateGeomObjectPanel d_parent = null;

  // Components
  private JTextField nameEntry = null;
  private JComboBox matIDComB = null;
  private WholeNumberField xresEntry = null;
  private WholeNumberField yresEntry = null;
  private WholeNumberField zresEntry = null;
  private DecimalField xvelEntry = null;
  private DecimalField yvelEntry = null;
  private DecimalField zvelEntry = null;
  private DecimalField tempEntry = null;
  private DecimalField presEntry = null;
  private DecimalField rhoEntry = null;
  private JComboBox geomPieceComB = null;
  private JButton acceptButton = null;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------
  public GeomObjectPanel(boolean usePartDist,
                         ParticleList partList,
                         Vector mpmMat,
                         Vector iceMat,
                         Vector geomObj,
                         Vector geomPiece,
                         CreateGeomObjectPanel parent) {

    // Initialize
    d_usePartDist = false;
    d_partList = partList;
    d_mpmMat = mpmMat;
    d_iceMat = iceMat;
    d_geomObj = geomObj;
    d_geomPiece = geomPiece;

    // Save parent
    d_parent = parent;

    // Create a grid bag for the components
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Geom object name
    JLabel nameLabel = new JLabel("Name");
    UintahGui.setConstraints(gbc, 0, 0);
    gb.setConstraints(nameLabel, gbc);
    add(nameLabel);
  
    nameEntry = new JTextField("Box", 10);
    UintahGui.setConstraints(gbc, 1, 0);
    gb.setConstraints(nameEntry, gbc);
    add(nameEntry);

    // Geom object materials
    JLabel matIDLabel = new JLabel("Material");
    UintahGui.setConstraints(gbc, 2, 0);
    gb.setConstraints(matIDLabel, gbc);
    add(matIDLabel);

    matIDComB = new JComboBox();
    int mpmMatSize = d_mpmMat.size();
    if (mpmMatSize > 0) {
      for (int ii = 0; ii < mpmMatSize; ++ii) {
	String matName = (String) d_mpmMat.elementAt(ii);
	matIDComB.addItem(matName);
      }
    }
    int iceMatSize = d_iceMat.size();
    if (iceMatSize > 0) {
      for (int ii = 0; ii < iceMatSize; ++ii) {
	String matName = (String) d_iceMat.elementAt(ii);
	matIDComB.addItem(matName);
      }
    }
    UintahGui.setConstraints(gbc, 3, 0);
    gb.setConstraints(matIDComB, gbc);
    add(matIDComB);
  
    // Resolution
    JLabel resLabel = new JLabel("Resolution:");
    UintahGui.setConstraints(gbc, 0, 1);
    gb.setConstraints(resLabel, gbc);
    add(resLabel);

    JLabel xresLabel = new JLabel("x");         
    UintahGui.setConstraints(gbc, 1, 1);
    gb.setConstraints(xresLabel, gbc);
    add(xresLabel); 

    xresEntry = new WholeNumberField(0,5);
    UintahGui.setConstraints(gbc, 2, 1);
    gb.setConstraints(xresEntry, gbc);
    add(xresEntry);

    JLabel yresLabel = new JLabel("y");         
    UintahGui.setConstraints(gbc, 3, 1);
    gb.setConstraints(yresLabel, gbc);
    add(yresLabel); 

    yresEntry = new WholeNumberField(0,5);
    UintahGui.setConstraints(gbc, 4, 1);
    gb.setConstraints(yresEntry, gbc);
    add(yresEntry);

    JLabel zresLabel = new JLabel("z");         
    UintahGui.setConstraints(gbc, 5, 1);
    gb.setConstraints(zresLabel, gbc);
    add(zresLabel); 

    zresEntry = new WholeNumberField(0,5);
    UintahGui.setConstraints(gbc, 6, 1);
    gb.setConstraints(zresEntry, gbc);
    add(zresEntry);

    // Velocity
    JLabel velLabel = new JLabel("Velocity:");
    UintahGui.setConstraints(gbc, 0, 2);
    gb.setConstraints(velLabel, gbc);
    add(velLabel);

    JLabel xvelLabel = new JLabel("x");         
    UintahGui.setConstraints(gbc, 1, 2);
    gb.setConstraints(xvelLabel, gbc);
    add(xvelLabel); 

    xvelEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbc, 2, 2);
    gb.setConstraints(xvelEntry, gbc);
    add(xvelEntry);

    JLabel yvelLabel = new JLabel("y");         
    UintahGui.setConstraints(gbc, 3, 2);
    gb.setConstraints(yvelLabel, gbc);
    add(yvelLabel); 

    yvelEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbc, 4, 2);
    gb.setConstraints(yvelEntry, gbc);
    add(yvelEntry);

    JLabel zvelLabel = new JLabel("z");         
    UintahGui.setConstraints(gbc, 5, 2);
    gb.setConstraints(zvelLabel, gbc);
    add(zvelLabel); 

    zvelEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbc, 6, 2);
    gb.setConstraints(zvelEntry, gbc);
    add(zvelEntry);

    // Temperature
    JLabel tempLabel = new JLabel("Temperature");
    UintahGui.setConstraints(gbc, 0, 3);
    gb.setConstraints(tempLabel, gbc);
    add(tempLabel);

    tempEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbc, 1, 3);
    gb.setConstraints(tempEntry, gbc);
    add(tempEntry);

    // Density
    JLabel rhoLabel = new JLabel("Density");
    UintahGui.setConstraints(gbc, 2, 3);
    gb.setConstraints(rhoLabel, gbc);
    add(rhoLabel);

    rhoEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbc, 3, 3);
    gb.setConstraints(rhoEntry, gbc);
    add(rhoEntry);

    // Pressure
    JLabel presLabel = new JLabel("Pressure");
    UintahGui.setConstraints(gbc, 4, 3);
    gb.setConstraints(presLabel, gbc);
    add(presLabel);

    presEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbc, 5, 3);
    gb.setConstraints(presEntry, gbc);
    add(presEntry);

    // Geometry Piece
    JLabel geomPieceLabel = new JLabel("Geometry Pieces");
    UintahGui.setConstraints(gbc, 0, 4);
    gb.setConstraints(geomPieceLabel, gbc);
    add(geomPieceLabel);
    geomPieceComB = new JComboBox();
    UintahGui.setConstraints(gbc, 1, 4);
    gb.setConstraints(geomPieceComB, gbc);
    add(geomPieceComB);

    if (d_usePartDist) {
      geomPieceLabel.setEnabled(false);
      geomPieceComB.setEnabled(false);
    } else {
      geomPieceLabel.setEnabled(true);
      geomPieceComB.setEnabled(true);
    }

    // Accept button
    acceptButton = new JButton("Accept");
    acceptButton.setActionCommand("accept");
    acceptButton.addActionListener(this);
    UintahGui.setConstraints(gbc, 0, 5);
    gb.setConstraints(acceptButton, gbc);
    add(acceptButton);
  }

  //---------------------------------------------------------------------
  // Update the usePartDist flag
  //---------------------------------------------------------------------
  public void usePartDist(boolean flag) {
    d_usePartDist = flag;
  }

  //---------------------------------------------------------------------
  // Actions performed when an item is selected
  //---------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    if (e.getActionCommand() == "accept") {

      if (d_usePartDist) {

	// Create a geometry object for the particles
	GeomObject particles = new GeomObject();
	particles.setName(new String("particles"));
	particles.setResolution(xresEntry.getValue(), yresEntry.getValue(),
			       zresEntry.getValue());
	particles.setVelocity(xvelEntry.getValue(), yvelEntry.getValue(),
			     zvelEntry.getValue());
	particles.setTemperature(tempEntry.getValue());
	particles.setDensity(rhoEntry.getValue());
	particles.setPressure(presEntry.getValue());

	int numPart = d_partList.size();
	for (int ii=0; ii < numPart; ++ii) {
	  particles.addGeomPiece((GeomPiece) d_geomPiece.elementAt(ii));
	}
	d_geomObj.addElement(particles);

	// Create a geometry object for the difference
	GeomObject remainder = new GeomObject();
	remainder.setName(new String("rest_of_domain"));
	remainder.setResolution(xresEntry.getValue(), yresEntry.getValue(),
			       zresEntry.getValue());
	remainder.setVelocity(xvelEntry.getValue(), yvelEntry.getValue(),
			     zvelEntry.getValue());
	remainder.setTemperature(tempEntry.getValue());
	remainder.setDensity(rhoEntry.getValue());
	remainder.setPressure(presEntry.getValue());
	remainder.addGeomPiece((GeomPiece) d_geomPiece.elementAt(numPart+1));
        d_geomObj.addElement(remainder);
        
      } else {
      }
        
    }

  }

  //---------------------------------------------------------------------
  // Update the components
  //---------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }
}
