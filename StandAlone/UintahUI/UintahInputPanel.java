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

//**************************************************************************
// Class   : UintahInputPanel
// Purpose : Creates a panel for entering Uintah input data
//**************************************************************************
public class UintahInputPanel extends JPanel {

  // Data
  private ParticleList d_particleList;

  // Data that may be needed later
  JTabbedPane uintahTabbedPane = null;

  // Constructor
  public UintahInputPanel(ParticleList particleList) {

    // Copy the arguments
    d_particleList = particleList;

    // Create a tabbed pane
    uintahTabbedPane = new JTabbedPane();

    // Create the panels to be added to the tabbed pane
    GeneralInputsPanel generalInpPanel = new GeneralInputsPanel(this);
    MPMInputsPanel mpmInpPanel = new MPMInputsPanel(this); 
    MPMMaterialsPanel mpmMatPanel = new MPMMaterialsPanel(this); 
    ICEInputsPanel iceInpPanel = new ICEInputsPanel(this); 
    ICEMaterialsPanel iceMatPanel = new ICEMaterialsPanel(this); 
    /*
    MaterialPropICEPanel matPropICEPanel = new MaterialPropICEPanel(); 
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
    /*
    uintahTabbedPane.addTab("Grid and BC Inputs", null,
                          gridBCPanel, null);
    uintahTabbedPane.setSelectedIndex(0);
    */

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

  // Methods for getting the canvases
  public ParticleList getParticleList() {
    return d_particleList;
  }

  // Tab listener
  class TabListener implements ChangeListener {
    public void stateChanged(ChangeEvent e) {
    }
  }
}
