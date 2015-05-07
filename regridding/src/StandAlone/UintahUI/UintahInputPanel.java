//**************************************************************************
// Program : UintahInputPanel.java
// Purpose : Input data for Uintah input ups file.
// Author  : Biswajit Banerjee
// Date    : 04/05/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.Vector;
import java.io.PrintWriter;

//**************************************************************************
// Class:  UintahInputPanel
// Purpose : Creates a panel for entering Uintah input data
//**************************************************************************
public class UintahInputPanel extends JPanel 
                              implements ChangeListener {

  // Data
  private ParticleList d_partList = null;
  private UintahGui d_parent = null;
  private Vector d_mpmMat = null;
  private Vector d_iceMat = null;
  private Vector d_geomObj = null;
  private String d_simComponent = null;

  // Data that may be needed later
  private JTabbedPane uintahTabbedPane = null;
  private GeneralInputsPanel generalInpPanel = null;
  private GeometryPanel geomPanel = null;
  private MPMInputsPanel mpmInpPanel = null;
  private MPMMaterialsPanel mpmMatPanel = null;
  private ICEInputsPanel iceInpPanel = null;
  private ICEMaterialsPanel iceMatPanel = null;
  private MPMICEExchangePanel exchangePanel = null;
  private GridBCPanel gridBCPanel = null;

  // Constructor
  public UintahInputPanel(ParticleList particleList,
                          UintahGui parent) {

    // Initialize
    d_mpmMat = new Vector();
    d_iceMat = new Vector();
    d_geomObj = new Vector();
    d_simComponent = new String("none");

    // Copy the arguments
    d_partList = particleList;
    d_parent = parent;

    // Create a tabbed pane
    uintahTabbedPane = new JTabbedPane();

    // Create the panels to be added to the tabbed pane
    generalInpPanel = new GeneralInputsPanel(d_simComponent, this);
    geomPanel = new GeometryPanel(d_partList, d_geomObj, this); 
    mpmInpPanel = new MPMInputsPanel(this); 
    mpmMatPanel = new MPMMaterialsPanel(d_geomObj, d_mpmMat, this); 
    iceInpPanel = new ICEInputsPanel(d_mpmMat, d_iceMat, this); 
    iceMatPanel = new ICEMaterialsPanel(d_geomObj, d_iceMat, this); 
    exchangePanel = new MPMICEExchangePanel(d_mpmMat, d_iceMat, this);
    gridBCPanel = new GridBCPanel(this); 

    // Add the tabs
    uintahTabbedPane.addTab("General Inputs", null,
                          generalInpPanel, null);
    uintahTabbedPane.addTab("Geometry", null,
                          geomPanel, null);
    uintahTabbedPane.addTab("MPM Parameters", null,
                          mpmInpPanel, null);
    uintahTabbedPane.addTab("MPM Materials", null,
                          mpmMatPanel, null);
    uintahTabbedPane.addTab("ICE Parameters", null,
                          iceInpPanel, null);
    uintahTabbedPane.addTab("ICE Materials", null,
                          iceMatPanel, null);
    uintahTabbedPane.addTab("Exchange", null,
                          exchangePanel, null);
    uintahTabbedPane.addTab("Grid and BC", null,
                          gridBCPanel, null);
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
    
    // Set all except general inputs as disabled.  The relevant tabs
    // are enabled once the simulation component is known
    int numTabs = uintahTabbedPane.getTabCount();
    for (int ii=1; ii < numTabs; ++ii) {
      uintahTabbedPane.setEnabledAt(ii, false);
    }

    // Add tab listener
    uintahTabbedPane.addChangeListener(this);

  }

  // Refresh when tabs are selected
  public void stateChanged(ChangeEvent e) {

    // If the selected tab is the geom panel then turn the display 
    // frame visible
    if (uintahTabbedPane.getSelectedIndex() == 1) {
      geomPanel.setVisibleDisplayGeometryFrame(true);
    } 
    
    // When a tab is changed, create MPM materials or MPM+ICE
    // material if a particle list is being used
    int numParticles = d_partList.size();
    if (numParticles > 0) {
      if (d_simComponent.equals("mpm")) {
        mpmMatPanel.createPartListMPMMaterial(d_simComponent);
      } else if (d_simComponent.equals("mpmice")) {
        iceMatPanel.createPartListICEMaterial(d_simComponent);
      }
    }
    generalInpPanel.refresh();
    geomPanel.refresh();
    mpmInpPanel.refresh();
    mpmMatPanel.refresh();
    iceInpPanel.refresh();
    iceMatPanel.refresh();

    // Update the materials in the exchange panel
    exchangePanel.updateMaterials(d_mpmMat, d_iceMat);
  }

  // Make display frame visible
  public void setVisibleDisplayFrame(boolean visible) {
    geomPanel.setVisibleDisplayGeometryFrame(visible);
  }

  // Get the simulation component
  public String getSimComponent() {
    return d_simComponent;
  }

  // Enable the tabs depending on the chosen simulation component
  public void enableTabs(String simComponent) {
    
    d_simComponent = simComponent;
    if (d_simComponent.equals("mpm")) {
      uintahTabbedPane.setEnabledAt(1, true);
      uintahTabbedPane.setEnabledAt(2, true);
      uintahTabbedPane.setEnabledAt(3, true);
      uintahTabbedPane.setEnabledAt(4, false);
      uintahTabbedPane.setEnabledAt(5, false);
      uintahTabbedPane.setEnabledAt(7, true);
    } else if (d_simComponent.equals("ice")) {
      uintahTabbedPane.setEnabledAt(1, true);
      uintahTabbedPane.setEnabledAt(2, false);
      uintahTabbedPane.setEnabledAt(3, false);
      uintahTabbedPane.setEnabledAt(4, true);
      uintahTabbedPane.setEnabledAt(5, true);
      uintahTabbedPane.setEnabledAt(6, true);
      uintahTabbedPane.setEnabledAt(7, true);
    } else if (d_simComponent.equals("mpmice")) {
      int numTabs = uintahTabbedPane.getTabCount();
      for (int ii=1; ii < numTabs; ++ii) {
        uintahTabbedPane.setEnabledAt(ii, true);
      }
    } else {
      int numTabs = uintahTabbedPane.getTabCount();
      for (int ii=1; ii < numTabs; ++ii) {
        uintahTabbedPane.setEnabledAt(ii, false);
      }
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
    mpmMatPanel.writeUintahContact(pw, tab2);
    pw.println(tab1+"</MPM>");
    pw.println(tab);

    pw.println(tab1+"<ICE>");
    int numICEMat = d_iceMat.size();
    for (int ii = 0; ii < numICEMat; ++ii) {
      iceMatPanel.writeUintah(pw, tab2, ii);
    }
    pw.println(tab1+"</ICE>");
    pw.println(tab);
    
    pw.println(tab1+"<exchange_properties>");
    exchangePanel.writeUintah(pw, tab2);
    pw.println(tab1+"</exchange_properties>");

    pw.println(tab+"</MaterialProperties>");
    pw.println(tab);

    pw.println(tab+"<Grid>");
    gridBCPanel.writeUintah(pw, tab1);
    pw.println(tab+"</Grid>");

    pw.println(tab+"<PhysicalBC>");
    pw.println(tab1+"<MPM>");
    pw.println(tab1+"</MPM>");
    pw.println(tab+"</PhysicalBC>");
    pw.println(tab);

    pw.println("</Uintah_specification>");
  }
}
