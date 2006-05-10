//**************************************************************************
// Program : ParticleLocGeneratePanel.java
// Purpose : Generate particle distribution and plot distribution.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

//**************************************************************************
// Class   : ParticleLocGeneratePanel
// Purpose : Generate particle distribution and plot locations.
//**************************************************************************
public class ParticleLocGeneratePanel extends JPanel {

  // Data
  private ParticleGeneratePanel d_parentPanel = null;
  private ParticleList d_partList = null;
  private ParticleSize d_partSizeDist = null;

  private ComputeParticleLocPanel computePanel = null;
  private DisplayParticleLocPanel displayPanel = null;

  private double d_rveSize = 100.0;

  // Constructor
  public ParticleLocGeneratePanel(ParticleList partList, 
                                  ParticleSize partSizeDist, 
				  ParticleGeneratePanel parentPanel) {

    // Copy the arguments
    d_partList = partList;
    d_partSizeDist = partSizeDist;
    d_parentPanel = parentPanel;

    // Create and add the relevant panels
    computePanel = new ComputeParticleLocPanel(partList, partSizeDist, this);
    displayPanel = new DisplayParticleLocPanel(partList, this);
 
    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Set the constraints for the label
    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 0,0, 1,1,5);
    gb.setConstraints(computePanel,gbc);
    add(computePanel);

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

  public void setRVESize(double rveSize) {
    d_rveSize = rveSize;
  }

  public double getRVESize() {
    return d_rveSize;
  }
}
