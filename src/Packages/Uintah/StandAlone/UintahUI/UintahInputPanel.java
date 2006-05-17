//**************************************************************************
// Program : UintahInputPanel.java
// Purpose : Input data for Uintah input ups file.
// Author  : Biswajit Banerjee
// Date    : 04/05/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.Vector;
import java.io.PrintWriter;

//**************************************************************************
// Class:  UintahInputPanel
// Purpose : Creates a panel for entering Uintah input data
//**************************************************************************
public class UintahInputPanel extends JPanel {

  // Data
  private ParticleList d_partList;
  private UintahGui d_parent;
  private Vector d_mpmMat;
  private Vector d_iceMat;
  private Vector d_geomObj;

  // Data that may be needed later
  private JTabbedPane uintahTabbedPane = null;
  private GeneralInputsPanel generalInpPanel = null;
  private MPMInputsPanel mpmInpPanel = null;
  private MPMMaterialsPanel mpmMatPanel = null;
  private ICEInputsPanel iceInpPanel = null;
  private ICEMaterialsPanel iceMatPanel = null;
  private GeometryPanel geomPanel = null;

  // Constructor
  public UintahInputPanel(ParticleList particleList,
                          UintahGui parent) {

    // Initialize
    d_mpmMat = new Vector();
    d_iceMat = new Vector();
    d_geomObj = new Vector();

    // Copy the arguments
    d_partList = particleList;
    d_parent = parent;

    // Create a tabbed pane
    uintahTabbedPane = new JTabbedPane();

    // Create the panels to be added to the tabbed pane
    generalInpPanel = new GeneralInputsPanel(this);
    mpmInpPanel = new MPMInputsPanel(this); 
    mpmMatPanel = new MPMMaterialsPanel(d_mpmMat, this); 
    iceInpPanel = new ICEInputsPanel(this); 
    iceMatPanel = new ICEMaterialsPanel(d_iceMat, this); 
    geomPanel = new GeometryPanel(d_partList, d_geomObj, d_mpmMat, d_iceMat, 
                                  this); 
    /*
    GridBCPanel gridBCPanel = new GridBCPanel(); 
    */

    // Add the tabs
    uintahTabbedPane.addTab("General Inputs", null,
                          generalInpPanel, null);
    uintahTabbedPane.addTab("MPM Parameters", null,
                          mpmInpPanel, null);
    uintahTabbedPane.addTab("MPM Materials", null,
                          mpmMatPanel, null);
    uintahTabbedPane.addTab("ICE Parameters", null,
                          iceInpPanel, null);
    uintahTabbedPane.addTab("ICE Materials", null,
                          iceMatPanel, null);
    uintahTabbedPane.addTab("Geometry", null,
                          geomPanel, null);
    /*
    uintahTabbedPane.addTab("Grid and BC Inputs", null,
                          gridBCPanel, null);
    */
    uintahTabbedPane.setSelectedIndex(0);

    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Set the constraints for the tabbed pane
    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 0,1, 1,1,5);
    gb.setConstraints(uintahTabbedPane, gbc);
    add(uintahTabbedPane);
    
    // Add tab listener
    TabListener tabListener = new TabListener();
    uintahTabbedPane.addChangeListener(tabListener);
  }

  // Tab listener
  private class TabListener implements ChangeListener {
    public void stateChanged(ChangeEvent e) {
    }
  }

  // Update Panels
  public void updatePanels() {
    validate();
    d_parent.updatePanels();
  }

  // Write in Uintah format
  public void writeUintah(PrintWriter pw) {

    if (pw == null) return;

    String tab = new String("  ");
    String tab1 = new String(tab+"  ");
    String tab2 = new String(tab1+"  ");

    pw.println("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    pw.println("<!-- <!DOCTYPE Uintah_specification SYSTEM \"input.dtd\"> -->");
    pw.println("<Uintah_specification>");
    pw.println(tab);

    generalInpPanel.writeUintah(pw, tab);
    mpmInpPanel.writeUintah(pw, tab);
    iceInpPanel.writeUintah(pw,tab);

    pw.println(tab+"<MaterialProperties>");
    pw.println(tab);
    
    pw.println(tab1+"<MPM>");
    int numMPMMat = d_mpmMat.size();
    for (int ii = 0; ii < numMPMMat; ++ii) {
      mpmMatPanel.writeUintah(pw, tab2, ii);
    }
    pw.println(tab1+"</MPM>");
    pw.println(tab);

    pw.println(tab1+"<ICE>");
    int numICEMat = d_iceMat.size();
    for (int ii = 0; ii < numICEMat; ++ii) {
    //iceMatPanel = new ICEMaterialsPanel(d_iceMat, this); 
    }
    pw.println(tab1+"</ICE>");
    pw.println(tab);

    pw.println(tab+"</MaterialProperties>");
    pw.println(tab);

    pw.println(tab+"<PhysicalBC>");
    pw.println(tab1+"<MPM>");
    pw.println(tab1+"</MPM>");
    pw.println(tab+"</PhysicalBC>");
    pw.println(tab);

    pw.println("</Uintah_specification>");
  }
}
