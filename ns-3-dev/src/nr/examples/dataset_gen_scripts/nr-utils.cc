/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License version 2 as
 *   published by the Free Software Foundation;
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
#include "nr-utils.h"
#include <ns3/enum.h>
//#include "flow-monitor-output-stats.h"
#include <ns3/nr-spectrum-value-helper.h>
#include <ns3/antenna-module.h>

#include "ns3/log.h"

NS_LOG_COMPONENT_DEFINE ("NrUtils");

namespace ns3 {

void
NrUtils::ConfigureBwpTo (BandwidthPartInfoPtr & bwp, double centerFreq, double bwpBw)
{
  bwp->m_centralFrequency = centerFreq;
  bwp->m_higherFrequency = centerFreq + (bwpBw / 2);
  bwp->m_lowerFrequency = centerFreq - (bwpBw / 2);
  bwp->m_channelBandwidth = bwpBw;
}

//  unnamed namespace
namespace {

void ConfigurePhy (Ptr<NrHelper> &nrHelper,
                   Ptr<NetDevice> gnb,
                   double orientationRads,
                   uint16_t numerology,
                   double txPowerBs,
                   const std::string & pattern,
                   uint32_t bwpIndex)
{
  // Change the antenna orientation
  Ptr<NrGnbPhy> phy0 = nrHelper->GetGnbPhy (gnb, 0);  // BWP 0
  Ptr<UniformPlanarArray> antenna0 =
    ConstCast<UniformPlanarArray> (phy0->GetSpectrumPhy ()->GetAntennaArray ());
  antenna0->SetAttribute ("BearingAngle", DoubleValue (orientationRads));

  // configure the beam that points toward the center of hexagonal
  // In case of beamforming, it will be overwritten.
  phy0->GetBeamManager ()->SetPredefinedBeam (3, 30);

  // Set numerology
  nrHelper->GetGnbPhy (gnb, 0)->SetAttribute ("Numerology", UintegerValue (numerology));      // BWP

  // Set TX power
  nrHelper->GetGnbPhy (gnb, 0)->SetAttribute ("TxPower", DoubleValue (txPowerBs));

  // Set TDD pattern
  nrHelper->GetGnbPhy (gnb, 0)->SetAttribute ("Pattern", StringValue (pattern));
}

}  // unnamed namespace

void
NrUtils::SetNrSimulatorParameters (const double sector0AngleRad,
                                           const std::string &scenario,
                                           const std::string &operationMode,
                                           uint16_t numerology,
                                           const std::string &pattern,
                                           const NodeContainer &gnbSector1Container,
                                           const NodeContainer &gnbSector2Container,
                                           const NodeContainer &gnbSector3Container,
                                           //const NodeContainer &ueSector1Container,
                                           //const NodeContainer &ueSector2Container,
                                           //const NodeContainer &ueSector3Container,
                                           const NodeContainer &ueNodesContainer,
                                           const Ptr<PointToPointEpcHelper> &baseEpcHelper,
                                           Ptr<NrHelper> &nrHelper,
                                           NetDeviceContainer &gnbSector1NetDev,
                                           NetDeviceContainer &gnbSector2NetDev,
                                           NetDeviceContainer &gnbSector3NetDev,
                                           //NetDeviceContainer &ueSector1NetDev,
                                           //NetDeviceContainer &ueSector2NetDev,
                                           //NetDeviceContainer &ueSector3NetDev,
                                           NetDeviceContainer &ueNetDevsContainer,
                                           bool enableUlPc,
                                           const std::string &scheduler,
                                           uint32_t bandwidthMHz, uint32_t freqScenario,
                                           double downtiltAngle)
{
  /*
   * Create the radio network related parameters
   */

  /*
   * Setup the NR module. We create the various helpers needed for the
   * NR simulation:
   * - IdealBeamformingHelper, which takes care of the beamforming part
   * - NrHelper, which takes care of creating and connecting the various
   * part of the NR stack
   */

  nrHelper = CreateObject<NrHelper> ();

  Ptr<IdealBeamformingHelper> idealBeamformingHelper;
  idealBeamformingHelper = CreateObject<IdealBeamformingHelper> ();
  nrHelper->SetBeamformingHelper (idealBeamformingHelper);

  Ptr<NrPointToPointEpcHelper> epcHelper = DynamicCast<NrPointToPointEpcHelper> (baseEpcHelper);
  nrHelper->SetEpcHelper (epcHelper);

  double txPowerBs = 0.0;

  BandwidthPartInfo::Scenario scene;
  if (scenario == "UMi")
    {
      txPowerBs = 30;
      scene =  BandwidthPartInfo::UMi_StreetCanyon_LoS;
    }
  else if (scenario == "UMa")
    {
      txPowerBs = 43;
      scene = BandwidthPartInfo::UMa_LoS;
    }
  else if (scenario == "RMa")
    {
      txPowerBs = 43;
      scene = BandwidthPartInfo::RMa_LoS;
    }
  else
    {
      NS_ABORT_MSG ("Unsupported scenario " << scenario << ". Supported values: UMi, UMa, RMa");
    }

  /*
   * Attributes of ThreeGppChannelModel still cannot be set in our way.
   * TODO: Coordinate with Tommaso
   */
  Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod",TimeValue (MilliSeconds (500)));
  nrHelper->SetChannelConditionModelAttribute ("UpdatePeriod", TimeValue (MilliSeconds (0)));

  nrHelper->SetPathlossAttribute ("ShadowingEnabled", BooleanValue (true));

  nrHelper->SetUePhyAttribute ("EnableUplinkPowerControl", BooleanValue (enableUlPc));
  /*
   * Create the necessary operation bands.
   *
   * In the 0 frequency scenario, each sector operates, in a separate band,
   * while for scenario 1 all the sectors are in the same band. Please note that
   * a single BWP in FDD is half the size of the corresponding TDD BWP, and the
   * parameter bandwidthMHz refers to the size of the FDD BWP.
   *
   * Scenario 0:  sectors NON_OVERLAPPING in frequency
   *
   * FDD scenario 0:
   *
   * |--------Band0--------|--------Band1--------|--------Band2--------|
   * |---------CC0---------|---------CC1---------|---------CC2---------|
   * |---BWP0---|---BWP1---|---BWP2---|---BWP3---|---BWP4---|---BWP5---|
   *
   *   Sector i will go in Bandi
   *   DL in the first BWP, UL in the second BWP
   *
   * TDD scenario 0:
   *
   * |--------Band0--------|--------Band1--------|--------Band2--------|
   * |---------CC0---------|---------CC2---------|---------CC2---------|
   * |---------BWP0--------|---------BWP1--------|---------BWP2--------|
   *
   *   Sector i will go in BWPi
   *
   *
   * Scenario 1:  sectors in OVERLAPPING bands
   *
   * Note that this configuration has 1/3 the total bandwidth of the
   * NON_OVERLAPPING configuration.
   *
   * FDD scenario 1:
   *
   * |--------Band0--------|
   * |---------CC0---------|
   * |---BWP0---|---BWP1---|
   *
   *   Sector i will go in BWPi
   *
   * TDD scenario 1:
   *
   * |--------Band0--------|
   * |---------CC0---------|
   * |---------BWP0--------|
   *
   * This is tightly coupled with what happens in lte-utils.cc
   *
   */
  // \todo: set band 0 start frequency from the command line
  const double band0Start = 2110e6;
  double bandwidthBwp = bandwidthMHz * 1e6;

  OperationBandInfo band0, band1, band2;
  band0.m_bandId = 0;
  band1.m_bandId = 1;
  band2.m_bandId = 2;

  if (freqScenario == 0) // NON_OVERLAPPING
    {
      uint8_t numBwp;

      if (operationMode == "FDD")
        {
          // FDD uses two BWPs per CC, one CC per band
          numBwp = 2;
        }
      else // if (operationMode = "TDD")
        {
          // Use double with BWP, to match total bandwidth for FDD in UL and DL
          bandwidthBwp *= 2;
          numBwp = 1;
        }

      double bandwidthCc = numBwp * bandwidthBwp;
      uint8_t numCcPerBand = 1;
      double bandwidthBand = numCcPerBand * bandwidthCc;
      double bandCenter = band0Start + bandwidthBand / 2.0;

      NS_LOG_LOGIC ("NON_OVERLAPPING, " << operationMode << ": "
                                        << bandwidthBand << ":" << bandwidthCc << ":"
                                        << bandwidthBwp << ", "
                                        << (int)numCcPerBand << ", " << (int)numBwp);

      NS_LOG_LOGIC ("bandConf0: " << bandCenter << " " << bandwidthBand);
      CcBwpCreator::SimpleOperationBandConf
        bandConf0 (bandCenter, bandwidthBand, numCcPerBand, scene);
      bandConf0.m_numBwp = numBwp;
      bandCenter += bandwidthBand;

      NS_LOG_LOGIC ("bandConf1: " << bandCenter << " " << bandwidthBand);
      CcBwpCreator::SimpleOperationBandConf
        bandConf1 (bandCenter, bandwidthBand, numCcPerBand, scene);
      bandConf1.m_numBwp = numBwp;
      bandCenter += bandwidthBand;

      NS_LOG_LOGIC ("bandConf2: " << bandCenter << " " << bandwidthBand);
      CcBwpCreator::SimpleOperationBandConf
        bandConf2 (bandCenter, bandwidthBand, numCcPerBand, scene);
      bandConf2.m_numBwp = numBwp;

      // Create, then configure
      CcBwpCreator ccBwpCreator;
      band0 = ccBwpCreator.CreateOperationBandContiguousCc (bandConf0);
      band0.m_bandId = 0;

      band1 = ccBwpCreator.CreateOperationBandContiguousCc (bandConf1);
      band1.m_bandId = 1;

      band2 = ccBwpCreator.CreateOperationBandContiguousCc (bandConf2);
      band2.m_bandId = 2;

      bandCenter = band0Start + bandwidthBwp / 2.0;

      NS_LOG_LOGIC ("band0[0][0]: " << bandCenter << " " << bandwidthBwp);
      ConfigureBwpTo (band0.m_cc[0]->m_bwp[0], bandCenter, bandwidthBwp);
      bandCenter += bandwidthBwp;

      if (operationMode == "FDD")
        {
          NS_LOG_LOGIC ("band0[0][1]: " << bandCenter << " " << bandwidthBwp);
          ConfigureBwpTo (band0.m_cc[0]->m_bwp[1], bandCenter, bandwidthBwp);
          bandCenter += bandwidthBwp;
        }

      NS_LOG_LOGIC ("band1[0][0]: " << bandCenter << " " << bandwidthBwp);
      ConfigureBwpTo (band1.m_cc[0]->m_bwp[0], bandCenter, bandwidthBwp);
      bandCenter += bandwidthBwp;

      if (operationMode == "FDD")
        {
          NS_LOG_LOGIC ("band1[0][1]: " << bandCenter << " " << bandwidthBwp);
          ConfigureBwpTo (band1.m_cc[0]->m_bwp[1], bandCenter, bandwidthBwp);
          bandCenter += bandwidthBwp;
        }

      NS_LOG_LOGIC ("band2[0][0]: " << bandCenter << " " << bandwidthBwp);
      ConfigureBwpTo (band2.m_cc[0]->m_bwp[0], bandCenter, bandwidthBwp);
      bandCenter += bandwidthBwp;

      if (operationMode == "FDD")
        {
          NS_LOG_LOGIC ("band2[0][1]: " << bandCenter << " " << bandwidthBwp);
          ConfigureBwpTo (band2.m_cc[0]->m_bwp[1], bandCenter, bandwidthBwp);
        }

      std::cout << "BWP Configuration for NON_OVERLAPPING case, mode "
                << operationMode << "\n"
                << band0 << band1 << band2;
    }


  else if (freqScenario == 1) // OVERLAPPING
    {
      uint8_t numBwp;

      if (operationMode == "FDD")
        {
          // FDD uses two BWPs per CC, one CC per band
          numBwp = 2;
        }
      else // if (operationMode = "TDD")
        {
          // Use double with BWP, to match total bandwidth for FDD in UL and DL
          bandwidthBwp *= 2;
          numBwp = 1;
        }

      double bandwidthCc = numBwp * bandwidthBwp;
      uint8_t numCcPerBand = 1;
      double bandwidthBand = numCcPerBand * bandwidthCc;
      double bandCenter = band0Start + bandwidthBand / 2.0;

      NS_LOG_LOGIC ("OVERLAPPING, " << operationMode << ": "
                                    << bandwidthBand << ":" << bandwidthCc << ":"
                                    << bandwidthBwp << ", "
                                    << (int)numCcPerBand << ", " << (int)numBwp);

      NS_LOG_LOGIC ("bandConf0: " << bandCenter << " " << bandwidthBand);
      CcBwpCreator::SimpleOperationBandConf
        bandConf0 (bandCenter, bandwidthBand, numCcPerBand, scene);
      bandConf0.m_numBwp = numBwp;
      bandCenter += bandwidthBand;

      // Create, then configure
      CcBwpCreator ccBwpCreator;
      band0 = ccBwpCreator.CreateOperationBandContiguousCc (bandConf0);
      band0.m_bandId = 0;

      bandCenter = band0Start + bandwidthBwp / 2.0;

      NS_LOG_LOGIC ("band0[0][0]: " << bandCenter << " " << bandwidthBwp);
      ConfigureBwpTo (band0.m_cc[0]->m_bwp[0], bandCenter, bandwidthBwp);
      bandCenter += bandwidthBwp;

      if (operationMode == "FDD")
        {
          NS_LOG_LOGIC ("band0[0][1]: " << bandCenter << " " << bandwidthBwp);
          ConfigureBwpTo (band0.m_cc[0]->m_bwp[1], bandCenter, bandwidthBwp);
        }

      std::cout << "BWP Configuration for OVERLAPPING case, mode "
                << operationMode << "\n" << band0;
    }

  else
    {
      std::cerr << "unknown combination of freqScenario = " << freqScenario
                << " and operationMode = " << operationMode
                << std::endl;
      exit (1);
    }

   nrHelper->InitializeOperationBand (&band0);
   nrHelper->InitializeOperationBand (&band1);
   nrHelper->InitializeOperationBand (&band2);

  BandwidthPartInfoPtrVector sector1Bwps, sector2Bwps, sector3Bwps;
  if (freqScenario == 0) // NON_OVERLAPPING
    {
      sector1Bwps = CcBwpCreator::GetAllBwps ({band0});
      sector2Bwps = CcBwpCreator::GetAllBwps ({band1});
      sector3Bwps = CcBwpCreator::GetAllBwps ({band2});
    }
  else // OVERLAPPING
    {
      sector1Bwps = CcBwpCreator::GetAllBwps ({band0});
      sector2Bwps = CcBwpCreator::GetAllBwps ({band0});
      sector3Bwps = CcBwpCreator::GetAllBwps ({band0});
    }


  /*
   * Now, we can setup the attributes. We can have three kind of attributes:
   * (i) parameters that are valid for all the bandwidth parts and applies to
   * all nodes, (ii) parameters that are valid for all the bandwidth parts
   * and applies to some node only, and (iii) parameters that are different for
   * every bandwidth parts. The approach is:
   *
   * - for (i): Configure the attribute through the helper, and then install;
   * - for (ii): Configure the attribute through the helper, and then install
   * for the first set of nodes. Then, change the attribute through the helper,
   * and install again;
   * - for (iii): Install, and then configure the attributes by retrieving
   * the pointer needed, and calling "SetAttribute" on top of such pointer.
   *
   */

  /*
   *  Case (i): Attributes valid for all the nodes
   */
  // Beamforming method

  idealBeamformingHelper->SetAttribute ("BeamformingMethod", TypeIdValue (DirectPathBeamforming::GetTypeId ()));
  // Scheduler type

  if (scheduler == "PF")
    {
      nrHelper->SetSchedulerTypeId (TypeId::LookupByName ("ns3::NrMacSchedulerOfdmaPF"));
    }
  else if (scheduler == "RR")
    {
      nrHelper->SetSchedulerTypeId (TypeId::LookupByName ("ns3::NrMacSchedulerOfdmaRR"));
    }

  // configure SRS symbols
  /*
  nrHelper->SetSchedulerAttribute ("SrsSymbols", UintegerValue (1));
  nrHelper->SetSchedulerAttribute ("EnableSrsInUlSlots", BooleanValue (false));
  nrHelper->SetSchedulerAttribute ("EnableSrsInFSlots", BooleanValue (false));

  // configure CTRL symbols
  nrHelper->SetSchedulerAttribute ("DlCtrlSymbols", UintegerValue (1));

  // Core latency
  epcHelper->SetAttribute ("S1uLinkDelay", TimeValue (MilliSeconds (0)));
*/


  // Antennas for all the UEs
  nrHelper->SetUeAntennaAttribute ("NumRows", UintegerValue (1));
  nrHelper->SetUeAntennaAttribute ("NumColumns", UintegerValue (1));
  Ptr<IsotropicAntennaModel> ueIsotropicAntenna = CreateObject<IsotropicAntennaModel> ();
  ueIsotropicAntenna->SetAttribute ("Gain", DoubleValue (0.0));
  nrHelper->SetUeAntennaAttribute ("AntennaElement", PointerValue (ueIsotropicAntenna));

  nrHelper->SetGnbAntennaAttribute ("NumRows", UintegerValue (5));
  nrHelper->SetGnbAntennaAttribute ("NumColumns", UintegerValue (2));

  nrHelper->SetGnbAntennaAttribute ("AntennaElement", PointerValue (CreateObject<ThreeGppAntennaModel> ()));
  nrHelper->SetGnbAntennaAttribute ("DowntiltAngle", DoubleValue (downtiltAngle * M_PI / 180.0));

  // UE transmit power
  nrHelper->SetUePhyAttribute ("TxPower", DoubleValue (23.0));

  // We assume a common traffic pattern for all UEs
  uint32_t bwpIdForLowLat;
/*  if (operationMode == "FDD" && direction == "UL")
    {
      bwpIdForLowLat = 1;
    }*/

  // unsure if this code works as I think it does. Havent checked it since I have not tried FDD in 5G 
  if (operationMode == "FDD")
    {
      bwpIdForLowLat = 0;
      // gNb routing between Bearer and bandwidth part
      nrHelper->SetGnbBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForLowLat));

      // Ue routing between Bearer and bandwidth part
      nrHelper->SetUeBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForLowLat));
       bwpIdForLowLat = 1;
      // gNb routing between Bearer and bandwidth part
      nrHelper->SetGnbBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForLowLat));

      // Ue routing between Bearer and bandwidth part
      nrHelper->SetUeBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForLowLat));    
    }
  else { // TDD
      bwpIdForLowLat = 0;
      // gNb routing between Bearer and bandwidth part
      nrHelper->SetGnbBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForLowLat));

      // Ue routing between Bearer and bandwidth part
      nrHelper->SetUeBwpManagerAlgorithmAttribute ("NGBR_VIDEO_TCP_DEFAULT", UintegerValue (bwpIdForLowLat));
   }
  /*
   * We miss many other parameters. By default, not configuring them is equivalent
   * to use the default values. Please, have a look at the documentation to see
   * what are the default values for all the attributes you are not seeing here.
   */

  /*
   * Case (ii): Attributes valid for a subset of the nodes
   */

  // NOT PRESENT IN THIS SIMPLE EXAMPLE

  /*
   * We have configured the attributes we needed. Now, install and get the pointers
   * to the NetDevices, which contains all the NR stack:
   */

  //  NetDeviceContainer enbNetDev = nrHelper->InstallGnbDevice (gridScenario.GetBaseStations (), allBwps);
  gnbSector1NetDev = nrHelper->InstallGnbDevice (gnbSector1Container, sector1Bwps);
  NetDeviceContainer gnbNetDevs (gnbSector1NetDev);
  gnbSector2NetDev = nrHelper->InstallGnbDevice (gnbSector2Container, sector2Bwps);
  gnbNetDevs.Add (gnbSector2NetDev);
  gnbSector3NetDev = nrHelper->InstallGnbDevice (gnbSector3Container, sector3Bwps);
  gnbNetDevs.Add (gnbSector3NetDev);
  /*
  ueSector1NetDev = nrHelper->InstallUeDevice (ueSector1Container, sector1Bwps);
  NetDeviceContainer ueNetDevs (ueSector1NetDev);
  ueSector2NetDev = nrHelper->InstallUeDevice (ueSector2Container, sector2Bwps);
  ueNetDevs.Add (ueSector2NetDev);
  ueSector3NetDev = nrHelper->InstallUeDevice (ueSector3Container, sector3Bwps);
  ueNetDevs.Add (ueSector3NetDev);*/
  // all BWPs are the same when using the overlapping frequency scenario 
  ueNetDevsContainer = nrHelper->InstallUeDevice (ueNodesContainer, sector1Bwps);  

  int64_t randomStream = 1;
  randomStream += nrHelper->AssignStreams (gnbSector1NetDev, randomStream);
  randomStream += nrHelper->AssignStreams (gnbSector2NetDev, randomStream);
  randomStream += nrHelper->AssignStreams (gnbSector3NetDev, randomStream);
  randomStream += nrHelper->AssignStreams (ueNetDevsContainer, randomStream);
  //randomStream += nrHelper->AssignStreams (ueSector1NetDev, randomStream);
  //randomStream += nrHelper->AssignStreams (ueSector2NetDev, randomStream);
  //randomStream += nrHelper->AssignStreams (ueSector3NetDev, randomStream);

  /*
   * Case (iii): Go node for node and change the attributes we have to setup
   * per-node.
   */

  // Sectors (cells) of a site are pointing at different directions
  std::vector<double> sectorOrientationRad {
    sector0AngleRad,
    sector0AngleRad + 2.0 * M_PI / 3.0,   // + 120 deg
    sector0AngleRad - 2.0 * M_PI / 3.0   // - 120 deg
  };

  for (uint32_t cellId = 0; cellId < gnbNetDevs.GetN (); ++cellId)
    {
      Ptr<NetDevice> gnb = gnbNetDevs.Get (cellId);
      uint32_t numBwps = nrHelper->GetNumberBwp (gnb);
      if (numBwps > 2)
        {
          NS_ABORT_MSG ("Incorrect number of BWPs per CC");
        }

      uint32_t sector = cellId % (gnbSector3NetDev.GetN () == 0 ? 1 : 3);
      double orientation = sectorOrientationRad[sector];

      // First BWP (in case of FDD) or only BWP (in case of TDD)
      ConfigurePhy (nrHelper, gnb, orientation, numerology, txPowerBs, pattern, 0);

      if (numBwps == 2)  //FDD
        {
          ConfigurePhy (nrHelper, gnb, orientation, numerology, txPowerBs, pattern, 1);
          // Link the two FDD BWP
          nrHelper->GetBwpManagerGnb (gnb)->SetOutputLink (1, 0);
        }

    }


  // Set the UE routing:
  for (auto nd = ueNetDevsContainer.Begin (); nd != ueNetDevsContainer.End (); ++nd)
    {
      auto uePhyFirst = nrHelper->GetUePhy (*nd, 0);
      auto uePhySecond {uePhyFirst};
      if (operationMode == "FDD")
        {
          nrHelper->GetBwpManagerUe (*nd)->SetOutputLink (0, 1);
          uePhySecond = nrHelper->GetUePhy (*nd, 1);
          uePhySecond->SetUplinkPowerControl (uePhyFirst->GetUplinkPowerControl ());
        }
      //uePhyFirst->TraceConnectWithoutContext ("ReportCurrentCellRsrpSinr",
      //                                        MakeBoundCallback (&ReportSinrNr, sinrStats));
      //uePhySecond->TraceConnectWithoutContext ("ReportPowerSpectralDensity",
      //                                         MakeBoundCallback (&ReportPowerNr, ueTxPowerStats));

    }


  // When all the configuration is done, explicitly call UpdateConfig ()
  for (auto nd = gnbNetDevs.Begin (); nd != gnbNetDevs.End (); ++nd)
    {
      /*uint32_t bwpId = 0;
      if (operationMode == "FDD" && direction == "UL")
        {
          bwpId = 1;
        }
      auto gnbPhy = nrHelper->GetGnbPhy (*nd, bwpId);
      gnbPhy->TraceConnectWithoutContext ("SlotDataStats",
                                          MakeBoundCallback (&ReportSlotStatsNr, slotStats));
      gnbPhy->TraceConnectWithoutContext ("RBDataStats",
                                          MakeBoundCallback (&ReportRbStatsNr, rbStats));
      gnbPhy->GetSpectrumPhy ()->TraceConnectWithoutContext ("RxDataTrace",
                                                             MakeBoundCallback (&ReportGnbRxDataNr, gnbRxPowerStats)); */

      DynamicCast<NrGnbNetDevice> (*nd)->UpdateConfig ();
    }

  for (auto nd = ueNetDevsContainer.Begin (); nd != ueNetDevsContainer.End (); ++nd)
    {
      DynamicCast<NrUeNetDevice> (*nd)->UpdateConfig ();
    }

}

} // namespace ns3
