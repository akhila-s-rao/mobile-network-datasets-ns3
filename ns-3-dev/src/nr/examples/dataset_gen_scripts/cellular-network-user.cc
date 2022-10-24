/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include <ns3/command-line.h>
#include <ns3/show-progress.h>
#include "cellular-network.h"

using namespace ns3;

/**
 * \ingroup examples
 * \file lena-lte-comparison-user.cc
 * \brief A multi-cell network deployment with site sectorization
 *
 * This example describes how to setup a simulation using the 3GPP channel model
 * from TR 38.900. This example consists of an hexagonal grid deployment
 * consisting on a central site and a number of outer rings of sites around this
 * central site. Each site is sectorized, meaning that a number of three antenna
 * arrays or panels are deployed per gNB. These three antennas are pointing to
 * 30º, 150º and 270º w.r.t. the horizontal axis. We allocate a band to each
 * sector of a site, and the bands are contiguous in frequency.
 *
 * We provide a number of simulation parameters that can be configured in the
 * command line, such as the number of UEs per cell or the number of outer rings.
 * Please have a look at the possible parameters to know what you can configure
 * through the command line.
 *
 * With the default configuration, the example will create one DL flow per UE.
 * The example will print on-screen the end-to-end result of each flow,
 * as well as writing them on a file.
 *
 * \code{.unparsed}
$ ./waf --run "lena-lte-comparison-user --Help"
    \endcode
 *
 */
int
main (int argc, char *argv[])
{
    Parameters params;
    /*
    * From here, we instruct the ns3::CommandLine class of all the input parameters
    * that we may accept as input, as well as their description, and the storage
    * variable.
    */
    CommandLine cmd;

    cmd.AddValue ("scenario",
                "The urban scenario string (UMa,UMi,RMa)",
                params.scenario);
    cmd.AddValue ("numRings",
                "The number of rings around the central site",
                params.numOuterRings);
    cmd.AddValue ("useMicroLayer",
                "If true, a micro cell layer of base stations will be created over the macro cell layer",
                params.useMicroLayer);
    cmd.AddValue ("numMicroCells",
                "Number of micro cells that should be random dropped into the deployment",
                params.numMicroCells);  
    cmd.AddValue ("microCellTxPower",
                "Transmit power of the base stations in the micro cell layer",
                params.microCellTxPower); 
    cmd.AddValue ("ueNumPergNb",
                "The number of UE per cell or gNB in multiple-ue topology",
                params.ueNumPergNb);
    cmd.AddValue ("siteFile",
                "Path to file of tower coordinates (instead of hexagonal grid)",
                params.baseStationFile);
    cmd.AddValue ("useSiteFile",
                "If true, it will be used site file, otherwise it will be used "
                "numRings parameter to create scenario.",
                params.useSiteFile);
    cmd.AddValue ("appGenerationTime",
                "Duration applications will generate traffic.",
                params.appGenerationTime);
    cmd.AddValue ("rat",
                "The radio access technology to use: LTE or NR",
                params.rat);
    cmd.AddValue ("operationMode",
                "The network operation mode can be TDD or FDD",
                params.operationMode);
    cmd.AddValue ("trafficLoadFrac",
                "The fraction of base station bandwidth to load with UDP flow traffic",
                params.trafficLoadFrac);
    cmd.AddValue ("scheduler",
                "PF: Proportional Fair, RR: Round-Robin",
                params.scheduler);
    cmd.AddValue ("bandwidth",
                "BW in MHz for each BWP (integer value): valid values are 20, 10, 5",
                params.bandwidthMHz);
    cmd.AddValue ("freqScenario",
                "0: NON_OVERLAPPING (each sector in different freq), 1: OVERLAPPING (same freq for all sectors)",
                params.freqScenario);
    cmd.AddValue ("enableUlPc",
                "Whether to enable or disable UL power control",
                params.enableUlPc);
    cmd.AddValue ("progressInterval",
                "Progress reporting interval",
                params.progressInterval);
    cmd.AddValue ("randomSeed",
                "Random seed to create repeatable or different runs",
                params.randSeed);
    cmd.AddValue ("appDelay",
                "Install the uplink and downlink one way delay measurement application",
                params.traceDelay);
    cmd.AddValue ("appRtt",
                "Install the round trip time measurement application",
                params.traceRtt);
    cmd.AddValue ("appDlThput",
                "Install the downlink throughput measurement application",
                params.traceDlThput);
    cmd.AddValue ("appUlThput",
                "Install the uplink throughput measurement application",
                params.traceUlThput);
    cmd.AddValue ("appDash",
                "Install the adaptive video streaming application",
                params.traceDash);
    cmd.AddValue ("appHttp",
                "Install the web browsing activity application",
                params.traceHttp);
    cmd.AddValue ("appVr",
                "Install the virtual reality (VR) streaming application",
                params.traceUlThput);
    cmd.AddValue ("appFlow",
                "Install the uplink and downlink UDP flow generation application",
                params.traceFlow);


    // Parse the command line
    cmd.Parse (argc, argv);
    params.Validate ();

    std::cout << params;

    ShowProgress spinner (params.progressInterval);

    CellularNetwork (params);

    return 0;
}
