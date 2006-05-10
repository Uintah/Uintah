//**************************************************************************
// Program : ParticleSizeDistInputPanel.java
// Purpose : Input particle size distribution and plot size distribution.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

//**************************************************************************
// Class   : ParticleSizeDistInputPanel
// Purpose : Inputs particle size distribution and plot size histogram.
//**************************************************************************
public class ParticleSizeDistInputPanel extends JPanel {

  // Data
  private ParticleGeneratePanel d_parentPanel = null;
  private ParticleSize d_partSizeDist = null;

  private InputPartDistPanel inputPanel = null;
  private DisplayPartDistPanel displayPanel = null;

  // Constructor
  public ParticleSizeDistInputPanel(ParticleSize partSizeDist,
                                    ParticleGeneratePanel parentPanel) {

    // Copy the arguments
    d_partSizeDist = partSizeDist;
    d_parentPanel = parentPanel;

    // Create and add the relevant panels
    inputPanel = new InputPartDistPanel(d_partSizeDist, this);
    displayPanel = new DisplayPartDistPanel(d_partSizeDist, this);

    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Set the constraints
    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 0,0, 1,1,5);
    gb.setConstraints(inputPanel,gbc);
    add(inputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 1,0, 1,1,5);
    gb.setConstraints(displayPanel,gbc);
    add(displayPanel);
  }

  public ParticleGeneratePanel getSuper() {
    return d_parentPanel;
  }

  public void refreshDisplayPartDistPanel() {
    displayPanel.refresh();
  }
}
