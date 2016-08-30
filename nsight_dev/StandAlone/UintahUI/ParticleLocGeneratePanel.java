//**************************************************************************
// Program : ParticleLocGeneratePanel.java
// Purpose : Generate particle distribution and plot distribution.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import javax.swing.*;

//**************************************************************************
// Class   : ParticleLocGeneratePanel
// Purpose : Generate particle distribution and plot locations.
//**************************************************************************
public class ParticleLocGeneratePanel extends JPanel {

  // Data
  private ParticleGeneratePanel d_parentPanel = null;

  private ComputeParticleLocPanel computePanel = null;
  private DisplayParticleLocFrame displayFrame = null;

  private double d_rveSize = 100.0;

  // Constructor
  public ParticleLocGeneratePanel(ParticleList partList, 
                                  ParticleSize partSizeDist, 
				  ParticleGeneratePanel parentPanel) {

    // Copy the arguments
    d_parentPanel = parentPanel;

    // Create and add the relevant panels
    computePanel = new ComputeParticleLocPanel(partList, partSizeDist, this);
    displayFrame = new DisplayParticleLocFrame(partList, this);
    displayFrame.pack();
    displayFrame.setVisible(false);
 
    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Set the constraints for the label
    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 0,0, 1,1,5);
    gb.setConstraints(computePanel,gbc);
    add(computePanel);
  }

  public ParticleGeneratePanel getSuper() {
    return d_parentPanel;
  }

  public void refreshDisplayPartLocFrame() {
    displayFrame.refresh();
  }

  public void setVisibleDisplayFrame(boolean visible) {
    displayFrame.setVisible(visible);
  }

  public void setRVESize(double rveSize) {
    d_rveSize = rveSize;
  }

  public double getRVESize() {
    return d_rveSize;
  }
}
