import time
from pathlib import Path
import acts
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV
import traceback
from acts.examples.odd import getOpenDataDetectorDirectory
from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("DD4hep simulation for ACTS")
    parser.add_argument(
        "--input-file",
        help="Input HepMC3 file (overrides config input_filename; default: merged_events.hepmc3)",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--pdg-file",
        help="Path to particle.tbl file containing PDG data",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--single-particle",
        help="Enable single particle simulation mode (particle gun)",
        action="store_true",
        default=None
    )
    parser.add_argument(
        "--threads",
        help="Number of threads to use",
        type=int,
        default=None
    )
    return parser.parse_args()

def configure_particle_gun(ddsim, config, logger):
    """Configure the particle gun based on configuration parameters
    
    Args:
        ddsim: DD4hepSimulation instance
        config: Configuration object
        logger: Logger instance
    """
    logger.info("Configuring particle gun")
    ddsim.enableGun = True
    
    # Configure particle type
    ddsim.gun.particle = getattr(config, 'gun_particle', 'e-')
    
    # Configure energy or momentum
    if hasattr(config, 'gun_energy'):
        ddsim.gun.energy = config.gun_energy * GeV
    else:
        ddsim.gun.momentumMin = getattr(config, 'gun_momentum_min', 0.0) * GeV
        ddsim.gun.momentumMax = getattr(config, 'gun_momentum_max', 10.0) * GeV
    
    # Configure direction only if explicitly provided in config
    if hasattr(config, 'gun_direction'):
        ddsim.gun.direction = config.gun_direction
    
    # Configure position
    ddsim.gun.position = getattr(config, 'gun_position', (0.0, 0.0, 0.0))
    
    # Configure angular distribution if specified
    if hasattr(config, 'gun_distribution'):
        ddsim.gun.distribution = config.gun_distribution
        ddsim.gun.isotrop = True
        
        # Configure angular limits
        if hasattr(config, 'gun_theta_min'):
            ddsim.gun.thetaMin = config.gun_theta_min
        if hasattr(config, 'gun_theta_max'):
            ddsim.gun.thetaMax = config.gun_theta_max
        if hasattr(config, 'gun_phi_min'):
            ddsim.gun.phiMin = config.gun_phi_min
        if hasattr(config, 'gun_phi_max'):
            ddsim.gun.phiMax = config.gun_phi_max
        # Map eta bounds as well (used by 'eta'/'pseudorapidity' distribution)
        if hasattr(config, 'gun_eta_min'):
            ddsim.gun.etaMin = config.gun_eta_min
        if hasattr(config, 'gun_eta_max'):
            ddsim.gun.etaMax = config.gun_eta_max
    
    # Configure multiplicity
    ddsim.gun.multiplicity = getattr(config, 'gun_multiplicity', 1)
    
    # Log configuration
    log_particle_gun_config(ddsim, logger)
    
    return ddsim

def configure_vertex_smearing(ddsim, config, logger):
    """Configure vertex smearing for all simulation modes (particle gun and HepMC3 input)
    
    Args:
        ddsim: DD4hepSimulation instance
        config: Configuration object
        logger: Logger instance
        
    Returns:
        DD4hepSimulation: Configured DD4hepSimulation instance
    """
    if hasattr(config, 'vertexOffset'):
        ddsim.vertexOffset = config.vertexOffset
        logger.info(f"Setting vertex offset: {ddsim.vertexOffset}")
    
    if hasattr(config, 'vertexSigma'):
        ddsim.vertexSigma = config.vertexSigma
        if any(x != 0 for x in ddsim.vertexSigma):
            logger.info(f"Setting vertex smearing sigma: {ddsim.vertexSigma}")
    
    return ddsim

def log_particle_gun_config(ddsim, logger):
    """Log the particle gun configuration
    
    Args:
        ddsim: DD4hepSimulation instance
        logger: Logger instance
    """
    logger.info(f"Particle gun configuration:")
    logger.info(f"  Particle: {ddsim.gun.particle}")
    if ddsim.gun.energy is not None:
        logger.info(f"  Energy: {ddsim.gun.energy}")
    else:
        logger.info(f"  Momentum range: {ddsim.gun.momentumMin} - {ddsim.gun.momentumMax}")
    logger.info(f"  Direction: {ddsim.gun.direction}")
    logger.info(f"  Position: {ddsim.gun.position}")
    logger.info(f"  Distribution: {ddsim.gun.distribution}")
    logger.info(f"  Multiplicity: {ddsim.gun.multiplicity}")

def configure_detector(ddsim):
    """Configure the detector for simulation
    
    Args:
        ddsim: DD4hepSimulation instance
        
    Returns:
        DD4hepSimulation: Configured DD4hepSimulation instance
    """
    # Get detector XML
    odd_dir = getOpenDataDetectorDirectory()
    odd_xml = odd_dir / "xml" / "OpenDataDetector.xml"
    
    # Configure DD4hep # TODO: This logic is probably backwards!!
    if isinstance(ddsim.compactFile, list):
        ddsim.compactFile = [str(odd_xml)]
    else:
        ddsim.compactFile = str(odd_xml)
    
    return ddsim

def configure_physics(ddsim, config, logger):
    """Configure physics for simulation
    
    Args:
        ddsim: DD4hepSimulation instance
        config: Configuration object
        
    Returns:
        DD4hepSimulation: Configured DD4hepSimulation instance
    """
    # Set PDG file if specified
    if hasattr(config, 'pdg_file') and config.pdg_file:
        ddsim.physics.pdgfile = str(config.pdg_file)
    
    # Set physics list if specified
    if hasattr(config, 'physics_list'):
        ddsim.physics.list = config.physics_list

    # Preserve preassigned displaced decays instead of promoting their daughters
    # to particles originating at the hard-scatter vertex.
    if getattr(config, 'alternative_decay_statuses', None) is not None:
        ddsim.physics.alternativeDecayStatuses = set(config.alternative_decay_statuses)
        logger.info(f"Alternative decay statuses: {ddsim.physics.alternativeDecayStatuses}")
    if getattr(config, 'extra_reject_pdgs', None) is not None:
        ddsim.physics.rejectPDGs = set(ddsim.physics.rejectPDGs) | set(config.extra_reject_pdgs)
        logger.info(f"Rejected generator PDGs: {ddsim.physics.rejectPDGs}")
    
    # Set truth particle handler
    if hasattr(config, 'truthParticleHandler'):
        logger.info(f"Setting truth particle handler to {config.truthParticleHandler}")
        ddsim.part.userParticleHandler = config.truthParticleHandler
    else:
        logger.info("Setting truth particle handler to default Geant4TCUserParticleHandler")
        ddsim.part.userParticleHandler = "Geant4TCUserParticleHandler"

    if hasattr(config, 'minimalKineticEnergy'):
        ddsim.part.minimalKineticEnergy = config.minimalKineticEnergy * GeV
    else:
        ddsim.part.minimalKineticEnergy = 1.0 * GeV

    if hasattr(config, 'keepAllParticles'):
        ddsim.part.keepAllParticles = config.keepAllParticles
    else:
        ddsim.part.keepAllParticles = False

    return ddsim

def configure_verbosity_and_ui(ddsim, config, logger):
    """Configure DDSim verbosity and Geant4 UI commands to reduce log volume
    
    Args:
        ddsim: DD4hepSimulation instance
        config: Configuration object
        logger: Logger instance
        
    Returns:
        DD4hepSimulation: Configured DD4hepSimulation instance
    """
    import logging as py_logging
    
    # Set DDSim print level (default to WARNING=4 for quieter logs)
    if hasattr(config, 'ddsim_printLevel'):
        ddsim.printLevel = config.ddsim_printLevel
        logger.info(f"Setting DDSim printLevel to {config.ddsim_printLevel}")
    else:
        ddsim.printLevel = 4  # WARNING level by default (was 3=INFO)
        logger.info(f"Setting DDSim printLevel to default WARNING level (4)")
    
    # Map DDSim printLevel to Python logging levels and configure DDSim loggers
    ddsim_to_python_level = {
        1: py_logging.DEBUG,    # VERBOSE
        2: py_logging.DEBUG,    # DEBUG  
        3: py_logging.INFO,     # INFO
        4: py_logging.WARNING,  # WARNING
        5: py_logging.ERROR,    # ERROR
        6: py_logging.CRITICAL, # FATAL
        7: py_logging.CRITICAL  # ALWAYS
    }
    
    python_level = ddsim_to_python_level.get(ddsim.printLevel, py_logging.WARNING)
    
    # Configure DDSim-related loggers using existing setup_logging functionality
    ddsim_loggers = ['DDSim', 'DDSim.Helper.Filter', 'DDG4']
    for logger_name in ddsim_loggers:
        setup_logging(logger_name, python_level)
    
    logger.info(f"Set Python logging level to {py_logging.getLevelName(python_level)} for DDSim modules")
    
    # Add Geant4 UI commands to reduce verbosity
    ui_commands = [
        '/run/verbose 0',        # Reduce run manager verbosity
        '/event/verbose 0',      # Reduce event manager verbosity
        '/tracking/verbose 0'    # Reduce tracking verbosity
    ]
    
    # Add any additional UI commands from config
    if hasattr(config, 'ui_commands') and config.ui_commands:
        ui_commands.extend(config.ui_commands)
    
    ddsim.ui.commandsConfigure = ui_commands
    logger.info(f"Added {len(ui_commands)} Geant4 UI commands to reduce verbosity")
    
    return ddsim

def single_thread_edm4hep_output(simulation):
    """Avoid the shared writer's duplicate event callbacks in sequential DDG4."""
    import DDG4
    kernel = DDG4.Kernel().worker()
    output = DDG4.EventAction(kernel, "Geant4Output2EDM4hep/EDM4hepOutput", False)
    output.Control = True
    output.Output = simulation.outputFile
    output.enableUI()
    output.RunHeader = simulation.meta.addParametersToRunHeader(simulation)
    parameters = simulation.meta.parseEventParameters()
    output.EventParametersString, output.EventParametersInt, output.EventParametersFloat = parameters
    output.RunNumberOffset = max(0, simulation.meta.runNumberOffset)
    output.EventNumberOffset = max(0, simulation.meta.eventNumberOffset)
    kernel.eventAction().add(output)


def run_ddsim(input_path, output_path, config, logger=None):
    """Run DD4hep simulation
    
    Args:
        input_path: Path to input HepMC3 file (or None for particle gun)
        output_path: Path to output EDM4hep file
        config: Configuration object
        logger: Logger instance (optional)
    """
    logger = logger or setup_logging("DD4hepStage")
    
    # Create and configure DD4hep simulation
    ddsim = DD4hepSimulation()
    
    # Configure detector
    ddsim = configure_detector(ddsim)
    
    # Check if we're using single particle mode
    use_single_particle = getattr(config, 'single_particle', False)
    if use_single_particle:
        logger.info("Using single particle simulation mode (particle gun)")
        ddsim = configure_particle_gun(ddsim, config, logger)
    else:
        # Standard HepMC3 input mode
        logger.info(f"Using HepMC3 input file: {input_path}")
        ddsim.inputFiles = [str(input_path)]
    
    # Configure common settings
    ddsim.outputFile = str(output_path)
    ddsim.numberOfEvents = config.events if config.events is not None else 10
    ddsim.numberOfThreads = config.threads if config.threads is not None else 1
    if ddsim.numberOfThreads <= 1 and str(output_path).endswith(".root"):
        ddsim.outputConfig.userOutputPlugin = single_thread_edm4hep_output
    ddsim.random.enableEventSeed = True
    ddsim.random.seed = getattr(config, 'seed', None) or int(time.time())
    
    # Configure vertex smearing (applies to both particle gun and HepMC3 input)
    ddsim = configure_vertex_smearing(ddsim, config, logger)
    
    # Configure physics
    ddsim = configure_physics(ddsim, config, logger)
    
    # Configure verbosity and UI commands
    ddsim = configure_verbosity_and_ui(ddsim, config, logger)
    
    # Log configuration
    logger.info(f"Running DD4hep simulation with {ddsim.numberOfEvents} events")
    if not use_single_particle:
        logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Random seed: {ddsim.random.seed}")
 
    exit_code = ddsim.run()
    if exit_code:
        raise RuntimeError(f"DDSim failed with exit code {exit_code}")

    

def main():
    timer = None  # Initialize timer to None
    logger = setup_logging() # Setup logger early
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)

        # Create output directory structure
        output_dir = Path(args.output)
        if hasattr(config, 'output_subdir') and config.output_subdir:
            output_dir = output_dir / config.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default input path if not specified and not in single particle mode
        input_path = None
        if not getattr(config, 'single_particle', False):
            # Priority: 1) command-line arg, 2) config file, 3) default
            input_filename = getattr(config, 'input_filename', 'merged_events.hepmc3')
            input_path = args.input_file or output_dir / input_filename
            logger.info(f"Using input filename: {input_filename} (from {'command-line' if args.input_file else 'config' if hasattr(config, 'input_filename') else 'default'})")
        output_path = output_dir / "edm4hep.root"
        
        # Initialize timing recorder
        timer = TimingRecorder(output_dir) # Assign here

        # Run DD4hep simulation
        with timer.record("DD4hep Simulation"):
            run_ddsim(input_path, output_path, config, logger)

        logger.info("DD4hep simulation completed successfully")
        logger.info(f"Output file: {output_path}")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Ensure the report is written even if errors occur
        if timer:
            try:
                timer.write_report()
            except Exception as report_e:
                logger.error(f"Error writing timing report: {str(report_e)}")
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
