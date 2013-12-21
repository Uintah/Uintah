//**************************************************************************
// Class   : CreateGeomObjectPanel
// Purpose : Panel to add and modify geometry objects
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import javax.swing.*;

public class CreateGeomObjectPanel extends JPanel 
                                   implements ActionListener {

  // Data
  private boolean d_usePartList = false;
  private ParticleList d_partList = null;
  private Vector d_geomObj = null;
  private Vector d_geomPiece = null;
  private InputGeometryPanel d_parent = null;

  // Components
  private JButton addButton = null;
  private JButton delButton = null;
  private JTabbedPane geomObjectTabPane = null;

  //-------------------------------------------------------------------------
  // Constructor
  //-------------------------------------------------------------------------
  public CreateGeomObjectPanel(boolean usePartList,
                               ParticleList partList,
                               Vector geomObj,
                               Vector geomPiece,
                               InputGeometryPanel parent) {

    // Initialize
    d_usePartList = false;
    d_partList = partList;
    d_geomObj = geomObj;
    d_geomPiece = geomPiece;

    // Save the arguments
    d_parent = parent;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create a panel for the buttons and the buttons
    JPanel panel = new JPanel(new GridLayout(1,0));

    addButton = new JButton("Create Geom Object");
    addButton.setActionCommand("add");
    addButton.addActionListener(this);
    panel.add(addButton);
      
    delButton = new JButton("Delete Geom Object");
    delButton.setActionCommand("del");
    delButton.addActionListener(this);
    panel.add(delButton);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(panel, gbc);
    add(panel);

    // Create a tabbed pane for the geometrypieces
    geomObjectTabPane = new JTabbedPane();

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(geomObjectTabPane, gbc);
    add(geomObjectTabPane);
  }

  //---------------------------------------------------------------------
  // Update the usePartList flag
  //---------------------------------------------------------------------
  public void usePartList(boolean flag) {
    d_usePartList = flag;
  }

  //---------------------------------------------------------------------
  // Disable the create and delete buttons
  //---------------------------------------------------------------------
  public void disableCreate() {
    addButton.setEnabled(false);
  }
  public void disableDelete() {
    delButton.setEnabled(false);
  }

  //---------------------------------------------------------------------
  // Enable the create button
  //---------------------------------------------------------------------
  public void enableCreate() {
    addButton.setEnabled(true);
  }
  public void enableDelete() {
    delButton.setEnabled(true);
  }

  //---------------------------------------------------------------------
  // Create geom object panels if a particle list is input
  //---------------------------------------------------------------------
  public void addPartListGeomObjectPanel() {

    // Set the usePartList flag to true
    d_usePartList = true;

    // Add the particles
    String particleTabName = new String("Particles");
    GeomObjectPanel particleGeomObjectPanel = 
      new GeomObjectPanel(d_usePartList, d_partList, 
                          d_geomObj, d_geomPiece, this);
    int numPart = d_partList.size();
    for (int ii=0; ii < numPart; ++ii) {
      GeomPiece gp = (GeomPiece) d_geomPiece.elementAt(ii);
      particleGeomObjectPanel.addGeomPiece(gp);
    }
    particleGeomObjectPanel.selectAllGeomPiece();
    geomObjectTabPane.addTab(particleTabName, particleGeomObjectPanel);

    // Add the remainder
    String remainderTabName = new String("Outside Particles");
    GeomObjectPanel remainderGeomObjectPanel = 
      new GeomObjectPanel(d_usePartList, d_partList, 
                          d_geomObj, d_geomPiece, this);
    GeomPiece gpOuter = (GeomPiece) d_geomPiece.elementAt(numPart);
    remainderGeomObjectPanel.addGeomPiece(gpOuter);
    remainderGeomObjectPanel.selectAllGeomPiece();
    geomObjectTabPane.addTab(remainderTabName, remainderGeomObjectPanel);

    // Add the inner stuff
    double partThick = ((Particle) d_partList.getParticle(0)).getThickness();
    if (partThick > 0.0) {
      String insideTabName = new String("Inside Particles");
      GeomObjectPanel insideGeomObjectPanel = 
        new GeomObjectPanel(d_usePartList, d_partList, 
                            d_geomObj, d_geomPiece, this);
      GeomPiece gpInner = (GeomPiece) d_geomPiece.elementAt(numPart+1);
      insideGeomObjectPanel.addGeomPiece(gpInner);
      insideGeomObjectPanel.selectAllGeomPiece();
      geomObjectTabPane.addTab(insideTabName, insideGeomObjectPanel);
    }

    // Update
    validate();
    updatePanels();
  }

  //---------------------------------------------------------------------
  // Remove geom object panels for a particle list
  //---------------------------------------------------------------------
  public void removePartListGeomObjectPanel() {

    // Set the usePartList flag to true
    d_usePartList = false;

    // Remove any existing geom objects
    if (d_geomObj.size() > 0) {
      d_geomObj.removeAllElements();
    }

    // Remove any geom object panel tabs
    geomObjectTabPane.removeAll(); 

    // Update
    validate();
    updatePanels();
  }

  //-------------------------------------------------------------------------
  // Actions performed when a button is pressed
  //-------------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    if (e.getActionCommand() == "add") {
      String tabName = new String("Object");
      GeomObjectPanel geomObjectPanel = 
        new GeomObjectPanel(d_usePartList, d_partList, 
                            d_geomObj, d_geomPiece, this);
      geomObjectTabPane.addTab(tabName, geomObjectPanel);
      validate();
      updatePanels();
    } else if (e.getActionCommand() == "del") {
      int index = geomObjectTabPane.getSelectedIndex();
      geomObjectTabPane.removeTabAt(index);
      if (d_geomObj.size() > 0) {
        d_geomObj.removeElementAt(index);
      }
      validate();
      updatePanels();
    }
  }

  //-------------------------------------------------------------------------
  // Update the components
  //-------------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }

}
