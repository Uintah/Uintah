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
import java.util.Vector;
import javax.swing.*;
import javax.swing.event.*;

public class GeomObjectPanel extends JPanel 
                             implements ActionListener {

  // Data
  private boolean d_usePartList = false;
  private Vector d_geomObj = null;
  private Vector d_geomPiece = null;
  private CreateGeomObjectPanel d_parent = null;
  private int d_numLocalGeomObject = 0;
  private int d_localGeomObjectStartIndex = 0;
  private Vector d_localGeomPiece = null;
  private double d_T = 0.0;
  private double d_p = 0.0;
  private double d_rho = 0.0;

  // Components
  private JTextField nameEntry = null;
  private IntegerVectorField resEntry = null;
  private DecimalVectorField velEntry = null;
  private DecimalField tempEntry = null;
  private DecimalField presEntry = null;
  private DecimalField rhoEntry = null;
  private JList geomPieceList = null;
  private DefaultListModel geomPieceListModel = null;
  private JScrollPane geomPieceSP = null;
  private JButton acceptButton = null;

  private JLabel nameLabel = null;
  private JLabel resLabel = null;
  private JLabel velLabel = null;
  private JLabel tempLabel = null;
  private JLabel rhoLabel = null;
  private JLabel presLabel = null;
  private JLabel geomPieceLabel = null;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------
  public GeomObjectPanel(boolean usePartList,
                         ParticleList partList,
                         Vector geomObj,
                         Vector geomPiece,
                         CreateGeomObjectPanel parent) {

    // Initialize
    d_usePartList = usePartList;
    d_geomObj = geomObj;
    d_geomPiece = geomPiece;
    d_numLocalGeomObject = 0;
    d_localGeomObjectStartIndex = 0;
    d_localGeomPiece = new Vector();

    // Save parent
    d_parent = parent;

    // Create a grid bag for the components
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);
    int fill = GridBagConstraints.BOTH;
    int xgap = 5;
    int ygap = 0;

    // Geom object name
    nameLabel = new JLabel("Name");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 0);
    gb.setConstraints(nameLabel, gbc);
    add(nameLabel);
  
    nameEntry = new JTextField("Geometry Object", 10);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 0);
    gb.setConstraints(nameEntry, gbc);
    add(nameEntry);

    // Resolution
    resLabel = new JLabel("Particles/Cell");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 1);
    gb.setConstraints(resLabel, gbc);
    add(resLabel);

    resEntry = new IntegerVectorField(2, 2, 2, 5);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 1);
    gb.setConstraints(resEntry, gbc);
    add(resEntry);

    // Velocity
    velLabel = new JLabel("Velocity (m/s)");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 2);
    gb.setConstraints(velLabel, gbc);
    add(velLabel);

    velEntry = new DecimalVectorField(0.0, 0.0, 0.0, 5);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 2);
    gb.setConstraints(velEntry, gbc);
    add(velEntry);

    // Set up the default values (SI units)
    d_T = 300.0;
    d_p = 101325.0;
    double gamma = 1.4;
    double cv = 716;
    d_rho = d_p/((gamma-1.0)*cv*d_T);

    // Temperature
    tempLabel = new JLabel("Temperature (K)");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 3);
    gb.setConstraints(tempLabel, gbc);
    add(tempLabel);

    tempEntry = new DecimalField(d_T, 5);
    tempEntry.addActionListener(this);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 3);
    gb.setConstraints(tempEntry, gbc);
    add(tempEntry);

    // Pressure
    presLabel = new JLabel("Pressure (Pa)");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 4);
    gb.setConstraints(presLabel, gbc);
    add(presLabel);

    presEntry = new DecimalField(d_p, 5);
    presEntry.addActionListener(this);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 4);
    gb.setConstraints(presEntry, gbc);
    add(presEntry);

    // Density
    rhoLabel = new JLabel("Density (kg/m3)");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 5);
    gb.setConstraints(rhoLabel, gbc);
    add(rhoLabel);

    rhoEntry = new DecimalField(d_rho, 5);
    rhoEntry.addActionListener(this);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 5);
    gb.setConstraints(rhoEntry, gbc);
    add(rhoEntry);

    // Geometry Piece
    geomPieceLabel = new JLabel("Geometry Pieces");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 6);
    gb.setConstraints(geomPieceLabel, gbc);
    add(geomPieceLabel);
    geomPieceListModel = new DefaultListModel();
    geomPieceList = new JList(geomPieceListModel);
    geomPieceList.setVisibleRowCount(4);
    geomPieceSP = new JScrollPane(geomPieceList);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 6);
    gb.setConstraints(geomPieceSP, gbc);
    add(geomPieceSP);

    if (d_usePartList) {
      geomPieceLabel.setEnabled(false);
      geomPieceSP.setEnabled(false);
    } else {
      geomPieceLabel.setEnabled(true);
      geomPieceSP.setEnabled(true);
    }

    // Accept button
    acceptButton = new JButton("Accept");
    acceptButton.setActionCommand("accept");
    acceptButton.addActionListener(this);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 7);
    gb.setConstraints(acceptButton, gbc);
    add(acceptButton);
  }

  //---------------------------------------------------------------------
  // Actions performed when an item is selected
  //---------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    Object source = e.getSource();
    if (source.equals(tempEntry)) {
      double gamma = 1.4;
      double cv = 716;
      d_T = tempEntry.getValue();
      d_p = presEntry.getValue();
      d_rho = d_p/((gamma-1.0)*cv*d_T);
      rhoEntry.setValue(d_rho);
    } else if (source.equals(presEntry)) {
      double gamma = 1.4;
      double cv = 716;
      d_T = tempEntry.getValue();
      d_p = presEntry.getValue();
      d_rho = d_p/((gamma-1.0)*cv*d_T);
      rhoEntry.setValue(d_rho);
    } else if (source.equals(rhoEntry)) {
      double gamma = 1.4;
      double cv = 716;
      d_T = tempEntry.getValue();
      d_rho = rhoEntry.getValue();
      d_p = (gamma-1.0)*cv*d_T*d_rho;
      presEntry.setValue(d_p);
    } else {
      String command = e.getActionCommand();
      if (command.equals(new String("accept"))) {
        if (d_usePartList) {
          // Create a geometry object for each of the particles
          createPartListGeomObject();
        } else {
          GeomObject go = new GeomObject();
          go.setName(nameEntry.getText());
          go.setResolution(resEntry.x(), resEntry.y(), resEntry.z());
          go.setVelocity(velEntry.x(), velEntry.y(), velEntry.z());
          go.setTemperature(tempEntry.getValue());
          go.setDensity(rhoEntry.getValue());
          go.setPressure(presEntry.getValue());
          go.addGeomPiece((GeomPiece) d_geomPiece.elementAt(0));
          d_geomObj.addElement(go);
          d_numLocalGeomObject = 1;
          d_localGeomObjectStartIndex = 0;
        }
      }
    }

  }

  //---------------------------------------------------------------------
  // Create a geometry object for each of the particles
  //---------------------------------------------------------------------
  private void createPartListGeomObject() {

    if (d_numLocalGeomObject > 0) {
      for (int ii = d_localGeomObjectStartIndex; ii < d_numLocalGeomObject;
           ++ii) {
        d_geomObj.removeElementAt(ii);
        GeomObject go = new GeomObject();
        go.setName((String) geomPieceListModel.elementAt(ii));
        go.setResolution(resEntry.x(), resEntry.y(), resEntry.z());
        go.setVelocity(velEntry.x(), velEntry.y(), velEntry.z());
        go.setTemperature(tempEntry.getValue());
        go.setDensity(rhoEntry.getValue());
        go.setPressure(presEntry.getValue());
        go.addGeomPiece((GeomPiece) d_localGeomPiece.elementAt(ii));
        d_geomObj.add(ii, go);
      }
    } else {
      d_localGeomObjectStartIndex = d_geomObj.size();
      int numGeomPiece = d_localGeomPiece.size();
      for (int ii = 0; ii < numGeomPiece; ++ii) {
        GeomObject go = new GeomObject();
        go.setName((String) geomPieceListModel.elementAt(ii));
        go.setResolution(resEntry.x(), resEntry.y(), resEntry.z());
        go.setVelocity(velEntry.x(), velEntry.y(), velEntry.z());
        go.setTemperature(tempEntry.getValue());
        go.setDensity(rhoEntry.getValue());
        go.setPressure(presEntry.getValue());
        go.addGeomPiece((GeomPiece) d_localGeomPiece.elementAt(ii));
        d_geomObj.addElement(go);
      }
      d_numLocalGeomObject = numGeomPiece;
    }
  }

  //---------------------------------------------------------------------
  // Update the usePartList flag
  //---------------------------------------------------------------------
  public void usePartList(boolean flag) {
    d_usePartList = flag;
  }

  //---------------------------------------------------------------------
  // Add a geometry piece to the panel
  //---------------------------------------------------------------------
  public void addGeomPiece(GeomPiece geomPiece) {
    d_localGeomPiece.addElement(geomPiece);
    geomPieceListModel.addElement(geomPiece.getName());
  }

  //---------------------------------------------------------------------
  // Delete a geometry piece from the panel
  //---------------------------------------------------------------------
  public void removeGeomPiece(GeomPiece geomPiece) {

    for (int ii = 0; ii < d_localGeomPiece.size(); ++ii) {
      GeomPiece gp = (GeomPiece) d_localGeomPiece.elementAt(ii);
      if (gp.equals(geomPiece)) {
        d_localGeomPiece.removeElementAt(ii);
        geomPieceListModel.removeElementAt(ii);
      }
    }
  }

  //---------------------------------------------------------------------
  // Select all geometry pieces
  //---------------------------------------------------------------------
  public void selectAllGeomPiece() {
    int numGeomPiece = geomPieceListModel.size();
    geomPieceList.setSelectionInterval(0, numGeomPiece);
  }

  //---------------------------------------------------------------------
  // Update the components
  //---------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }
}
