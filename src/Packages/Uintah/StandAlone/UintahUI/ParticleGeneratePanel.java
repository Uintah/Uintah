//**************************************************************************
// Program : ParticleGeneratePanel.java
// Purpose : Tabbed pane for particle generation.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import javax.swing.*;
import javax.swing.event.*;

//**************************************************************************
// Class   : ParticleGeneratePanel
// Purpose : Creates a panel for particle generation
//**************************************************************************
public class ParticleGeneratePanel extends JPanel {

  // Data
  private ParticleList d_particleList = null;
  private ParticleSize d_partSizeDist = null;

  // Data that may be needed later
  JTabbedPane partTabbedPane= null;

  // Constructor
  public ParticleGeneratePanel(ParticleList particleList) {

    // Copy the arguments
    d_particleList = particleList;

    // Initialize particle size distribution
    d_partSizeDist = new ParticleSize();

    // Create the tabbed pane
    partTabbedPane = new JTabbedPane();

    // Create the panels to be added to the tabbed pane
    ParticleSizeDistInputPanel particleSizeInputPanel = 
      new ParticleSizeDistInputPanel(d_partSizeDist, this);
    ParticleLocGeneratePanel particleLocGenPanel = 
      new ParticleLocGeneratePanel(d_particleList, d_partSizeDist, this);

    // Add the tabs
    partTabbedPane.addTab("Size Distribution", null,
                          particleSizeInputPanel, null);
    partTabbedPane.addTab("Generate Locations", null,
                          particleLocGenPanel, null);
    partTabbedPane.setSelectedIndex(0);

    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Set the constraints for the tabbed pane
    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 0,1, 1,1,5);
    gb.setConstraints(partTabbedPane, gbc);
    add(partTabbedPane);
    
    // Add tab listener
    TabListener tabListener = new TabListener();
    partTabbedPane.addChangeListener(tabListener);
  }

  public ParticleList getParticleList() {
    return d_particleList;
  }

  // Tab listener
  class TabListener implements ChangeListener {
    public void stateChanged(ChangeEvent e) {
    }
  }
}
